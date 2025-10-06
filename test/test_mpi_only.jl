using Test, Random, MPI, PencilFFTs
using LNMock: FFTPlanSpec, forwardFT!, inverseFT!, make_plan, allocate_fields, BackendDispatch

# Initialize MPI
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

if rank == 0
    println("="^60)
    println("Testing PencilFFTs MPI backend with $nprocs processes")
    println("="^60)
end

@testset "PencilFFTs MPI Backend" begin
    dims      = (32, 32, 32)
    Lbox      = 100
    boxsize   = (Lbox, Lbox, Lbox)

    # Create FFT plan with MPI backend
    fftspec = FFTPlanSpec(dims, boxsize, backend=:pencil_mpi, comm=comm)
    fftplan = make_plan(fftspec)

    # Allocate arrays (each rank gets a portion of the data)
    A, B = allocate_fields(fftspec, fftplan=fftplan)
    A2 = similar(A)

    # Fill with random data (same seed on all ranks for testing)
    rng = MersenneTwister(5128797806)
    randn!(rng, A)
    @. A *= Lbox

    # Perform forward and inverse FFT
    BackendDispatch.forwardFT!(B, fftplan, A, backend=:pencil_mpi)
    BackendDispatch.inverseFT!(A2, fftplan, B, backend=:pencil_mpi)

    # Compute relative error (using manual reduction to avoid ARM MPI issue)
    local_err = sum(abs.(A2.data - A.data))
    local_norm = sum(abs.(A.data))

    total_err = MPI.Allreduce(local_err, +, comm)
    total_norm = MPI.Allreduce(local_norm, +, comm)
    rel_err = total_err / total_norm

    if rank == 0
        println("Relative roundtrip error: $rel_err")
        println("Local array size on rank $rank: $(size(A))")
    end

    @test rel_err < 1e-8

    if rank == 0
        println("✓ Test passed on $nprocs MPI processes")
    end
end

if rank == 0
    println("="^60)
end
