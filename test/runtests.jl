using Test, Random, FFTW
using LNMock: LNConfig, compute_aHf, FFTPlanSpec, forwardFT!, inverseFT!, make_plan, allocate_fields, run_lognormal, BackendDispatch
using CosmoFFTs: set_backend!, reinitialize!
import LNMock

# MPI and PencilFFTs are optional - only load if available
const HAS_MPI = try
    using MPI, PencilFFTs
    true
catch
    @warn "MPI/PencilFFTs not available - skipping MPI tests"
    false
end

function run_tests()

    @testset "LNMock.jl" begin
        @testset "Configuration" begin
            cfg = LNConfig(
                           dims=(64, 64, 64),
                           boxsize=(100.0, 100.0, 100.0),
                           Ngals=1000,
                           pkA_file=joinpath(@__DIR__,"../../Lognormal/Planck2018_pkA.dat"),
                           pkgA_file=joinpath(@__DIR__,"../../Lognormal/Planck2018_pkgA.dat"),
                           outdir=joinpath(@__DIR__,"test_output"),
                           outhead="test"
                          )

            @test cfg.dims == (64, 64, 64)
            @test cfg.boxsize == (100.0, 100.0, 100.0)
            @test cfg.Ngals == 1000
            @test cfg.z == 2.7  # default value
            @test cfg.nsub == (1, 1, 1)  # default
            @test cfg.computeGRF == false  # default
            @test cfg.comm === nothing  # default

            println("✓ Configuration tests passed")
        end

        @testset "Utils - Cosmology" begin
            # 예시: Cosmology 함수 테스트
            z = 2.5
            aHfz = compute_aHf(z)
            @info "The value of aHf at z=$z is $aHfz !!"
            @test aHfz > 0
            @test typeof(aHfz) <: AbstractFloat  # 타입도 체크
            println("✓ Cosmology tests passed")
        end 

        @testset "Backend - FFTW Single" begin
            dims      = (32, 32, 32)
            Lbox      = 100
            boxsize   = (Lbox,Lbox,Lbox)
            fftspec = FFTPlanSpec(dims,boxsize)
            fftplan = make_plan(fftspec)

            A,B = allocate_fields(fftspec)
            A2  = similar(A)
            rng = MersenneTwister(5128797806)
            randn!(rng, A)
            @. A  *= Lbox
            forwardFT!(B,fftplan,A)
            inverseFT!(A2,fftplan,B)

            rel_err = sum(abs.(A2 - A))/sum(abs.(A))
            println("Relative error: $rel_err")
            @test rel_err < 1e-8
            # 작은 그리드로 FFT 테스트
            println("✓ FFTW Single tests passed")
        end

        @testset "Backend - FFTW Threads" begin
            dims      = (32, 32, 32)
            Lbox      = 100
            boxsize   = (Lbox,Lbox,Lbox)

            # Set number of threads for FFTW
            nthreads  = 3
            FFTW.set_num_threads(nthreads)

            fftspec = FFTPlanSpec(dims,boxsize,backend=:fftw_threads)
            fftplan = make_plan(fftspec)

            A,B = allocate_fields(fftspec)
            A2  = similar(A)
            rng = MersenneTwister(5128797806)
            randn!(rng, A)
            @. A  *= Lbox

            # Use BackendDispatch functions directly with backend keyword
            BackendDispatch.forwardFT!(B,fftplan,A,backend=:fftw_threads)
            BackendDispatch.inverseFT!(A2,fftplan,B,backend=:fftw_threads)

            rel_err = sum(abs.(A2 - A))/sum(abs.(A))
            println("Relative error (threads=$nthreads): $rel_err")
            @test rel_err < 1e-8
            println("✓ FFTW Threads tests passed")
        end

        @testset "Backend Consistency - Single vs Threads with Same GRF" begin
            # Test that single-threaded and multi-threaded backends produce
            # identical results when using the same Gaussian Random Field

            dims      = (32, 32, 32)
            Lbox      = 100
            boxsize   = (Lbox, Lbox, Lbox)
            seed      = UInt32(12345)

            # Create output directory for GRF
            grf_dir = joinpath(@__DIR__, "test_output")
            mkpath(grf_dir)
            grf_file = joinpath(grf_dir, "test_grf_consistency.h5")

            # ========================================
            # Step 1: Generate GRF with single-threaded backend
            # ========================================
            fftspec_single = FFTPlanSpec(dims, boxsize, backend=:fftw_single)

            # Allocate field for GRF
            grf_single, _ = allocate_fields(fftspec_single)

            # Generate Gaussian Random Field
            LNMock.Core.gen_GRand3d!(grf_single, seed, nothing)

            # Save GRF to file
            LNMock.Core.write_GRand3d(grf_single, grf_file)
            println("Generated and saved GRF with single-threaded backend")

            # ========================================
            # Step 2: Read GRF and use with threaded backend
            # ========================================
            nthreads = 3
            FFTW.set_num_threads(nthreads)

            fftspec_threads = FFTPlanSpec(dims, boxsize, backend=:fftw_threads)

            # Allocate field for GRF
            grf_threads, _ = allocate_fields(fftspec_threads)

            # Create FourierArrayInfo for read_GRand3d! (it needs aix1, aix2, aix3)
            Finfo = LNMock.BackendDispatch.FourierArrayInfo(fftspec_threads)

            # Read the same GRF
            LNMock.Core.read_GRand3d!(grf_threads, grf_file, Finfo)
            println("Read GRF with threaded backend")

            # ========================================
            # Step 3: Verify they are identical
            # ========================================
            diff = sum(abs.(grf_threads - grf_single))
            norm = sum(abs.(grf_single))
            rel_diff = diff / norm

            println("Relative difference between single and threads GRF: $rel_diff")
            @test rel_diff < 1e-15  # Should be essentially zero

            # Clean up
            rm(grf_file, force=true)

            println("✓ Backend consistency tests passed")
        end

        if HAS_MPI
            @testset "Backend - PencilFFTs MPI" begin
                # Initialize MPI if not already initialized
                if !MPI.Initialized()
                    MPI.Init()
                end

                # Switch backend to PencilFFTs MPI and reinitialize
                set_backend!(:pencil_mpi)
                reinitialize!()

                dims      = (32, 32, 32)
                Lbox      = 100
                boxsize   = (Lbox,Lbox,Lbox)
                comm      = MPI.COMM_WORLD

                fftspec = FFTPlanSpec(dims,boxsize,backend=:pencil_mpi,comm=comm)
                fftplan = make_plan(fftspec)

                # For PencilFFTs, need to pass the plan to allocate_fields
                A,B = allocate_fields(fftspec, fftplan=fftplan)
                A2  = similar(A)
                rng = MersenneTwister(5128797806)
                randn!(rng, A)
                @. A  *= Lbox

                # Use BackendDispatch functions directly with backend keyword
                BackendDispatch.forwardFT!(B,fftplan,A,backend=:pencil_mpi)
                BackendDispatch.inverseFT!(A2,fftplan,B,backend=:pencil_mpi)

                # Compute local relative error (avoid MPI reduction operators issue on ARM)
                local_err = sum(abs.(A2.data - A.data))
                local_norm = sum(abs.(A.data))

                # Manually reduce across MPI ranks
                total_err = MPI.Allreduce(local_err, +, comm)
                total_norm = MPI.Allreduce(local_norm, +, comm)
                rel_err = total_err / total_norm

                rank = MPI.Comm_rank(comm)
                if rank == 0
                    println("Relative error (MPI, rank=$rank): $rel_err")
                end
                @test rel_err < 1e-8
                if rank == 0
                    println("✓ PencilFFTs MPI tests passed")
                end

                # Reset backend to default for subsequent tests
                set_backend!(:fftw_single)
                reinitialize!()
            end
        else
            @testset "Backend - PencilFFTs MPI" begin
                @test_skip true  # Skip MPI tests when not available
                println("⊘ PencilFFTs MPI tests skipped (MPI not installed)")
            end
        end

        @testset "Full Workflow - Small Scale" begin
            # 작은 스케일로 전체 workflow 테스트
            mkpath(joinpath(@__DIR__,"test_output"))

            cfg = LNConfig(
                           dims=(32, 32, 32),
                           boxsize=(50.0, 50.0, 50.0),
                           Ngals=100,
                           pkA_file=joinpath(@__DIR__,"../../../Lognormal/Planck2018_pkA.dat"),
                           pkgA_file=joinpath(@__DIR__,"../../../Lognormal/Planck2018_pkgA.dat"),
                           outdir=joinpath(@__DIR__,"test_output"),
                           outhead="workflow_test",
                           nreal=1,
                           computeGRF=true,
                           GRFfilehead="grf_test"
                          )

            # 실제 실행 (파일이 준비되면)
            run_lognormal(cfg)
            @test isfile(joinpath(@__DIR__,"test_output/workflow_test_1.h5"))

            println("✓ Full workflow tests passed")
        end

    end
end

run_tests()

println("\n" * "="^60)
println("All LNMock tests completed successfully!")
println("="^60)
