# src/Core.jl
# ============================================================================
# Module Core
# 
# Core numerical frontend for LNMock.
# This module provides the high-level interface to generate lognormal fields
# using different FFT backends (FFTW single-thread, threaded, or PencilFFTs MPI).
#
# Exports:
#   - LNConfig: configuration struct
#   - run_lognormal: main driver function
#
# 27 September 2025
# Donghui Jeong
# ============================================================================
module Core
# ============================================================================

export LNConfig, run_lognormal

using FFTW, Random
using FLoops
using Distributions: Poisson
using HDF5
using ..LNMock: DEFAULT_BACKEND, FFTPlanSpec, FourierArrayInfo, allocate_fields, compute_aHf, make_plan, forwardFT!, inverseFT!
using ..Spectra: get_fpk
using ..Cosmology: compute_aHf

# Conditional MPI support - imported only when pencil_mpi backend is used
# Access MPI through parent module to avoid hard dependency
function _get_MPI()
    # This will be defined in LNMock module when MPI backend is loaded
    if !isdefined(Main, :MPI)
        error("MPI not available. Please use :pencil_mpi backend or load MPI.jl")
    end
    return Main.MPI
end

# Include utility functions for MPI broadcasting
include(joinpath(@__DIR__,"Utils","Utils.jl"))


# ------------------------------------------------------------------------------
# Configuration structure
# ------------------------------------------------------------------------------
"""
    LNConfig

Configuration for generating a lognormal mock field.

# Fields
- `dims::NTuple{3,Int}` : grid dimensions (n₁, n₂, n₃)
- `boxsize::NTuple{3,Float64}` : physical box size [Mpc/h]
- `seed::Int` : RNG seed for reproducibility
- fGRF = file storing the Gaussian random field
- `comm` : MPI communicator (optional, used only for PencilMPI backend)
"""
Base.@kwdef struct LNConfig
    dims::NTuple{3,Int}
    boxsize::NTuple{3,Float64}
    Ngals::Int
    z::Float64 = 2.7
    pkA_file::String
    pkgA_file::String
    outdir::String
    outhead::String
    nsub::NTuple{3,Int} = (1,1,1)
    nreal::Int = 1
    seed::UInt64 = 3141592653
    computeGRF = true
    writeGRF = false
    GRFfilehead = nothing
    comm = nothing
end


# ------------------------------------------------------------------------------
# Write/Read the 3D Gaussian random field
# ------------------------------------------------------------------------------
function write_GRand3d(v::Array{T,3}, filename::String) where T
    """
    Write 3D Gaussian Random Field to HDF5 file.
    
    Parameters:
    - v: 3D array to save
    - filename: output HDF5 filename
    """
    h5open(filename, "w") do file
        write(file, "grf", v)
        # Save dimensions as attributes for verification
        attrs(file["grf"])["dims"] = collect(size(v))
    end
    @info "Wrote GRF to $(filename), size: $(size(v))"
end
# ------------------------------------------------------------------------------
function read_GRand3d!(v::Array{T,3}, filename::String, Finfo) where T
    """
    Read 3D Gaussian Random Field from HDF5 file and extract local portion.
    
    For PencilFFTs MPI: reads full array and extracts the portion 
    corresponding to this rank's local indices.
    
    Parameters:
    - v: pre-allocated local array to fill (will be modified in-place)
    - filename: input HDF5 filename
    - Finfo: structure containing local index ranges (aix1, aix2, aix3)
    """
    h5open(filename, "r") do file
        dset = file["grf"]
        
        # Convert to ranges if needed
        i1_range = Finfo.aix1 isa AbstractRange ? Finfo.aix1 : (minimum(Finfo.aix1):maximum(Finfo.aix1))
        i2_range = Finfo.aix2 isa AbstractRange ? Finfo.aix2 : (minimum(Finfo.aix2):maximum(Finfo.aix2))
        i3_range = Finfo.aix3 isa AbstractRange ? Finfo.aix3 : (minimum(Finfo.aix3):maximum(Finfo.aix3))
        
        # Read only the needed hyperslab from disk
        v[:, :, :] = dset[i1_range, i2_range, i3_range]
    end
    
    @info "Read GRF from $(filename), extracted local portion: $(size(v))"
end
# ------------------------------------------------------------------------------
# Generating the 3D Gaussian random field
# ------------------------------------------------------------------------------
"""
    gen_GRand3d!(deltaL, seed, comm=nothing)

Generate a 3D Gaussian random field of zero mean.
- `deltaL` : array to fill
- `seed`   : RNG seed
- `comm`   : MPI communicator (or `nothing` for serial/thread)

This works for:
- Serial
- Multithreaded (through @floop)
- MPI (global mean subtraction)
"""
function gen_GRand3d!(deltaL, seed, comm::Union{Nothing, Any}=nothing)
    # 1. Generate Gaussian random variables
    rngL = MersenneTwister(seed)
    randn!(rngL, deltaL)   # do not parallelize this part

    # 2. Compute mean
    # -- local sum
    local_sum = 0.0
    @floop for x in deltaL
        @reduce(local_sum += x)
    end
    local_n = length(deltaL)

    # -- global reduction if MPI
    if comm !== nothing
        MPI = _get_MPI()
        global_sum = MPI.Allreduce(local_sum, +, comm)
        global_n   = MPI.Allreduce(local_n, +, comm)
        global_avg = global_sum / global_n
    else
        global_avg = local_sum / local_n
    end

    # 3. Subtract the global mean
    broadcast!(-, deltaL, deltaL, global_avg)
    return nothing
end
# ------------------------------------------------------------------------------
# generate the 3D densit yfield
# ------------------------------------------------------------------------------
"""
 gen_deltaLk!(deltaL, deltak, fftplan, Finfo,fpk)

Generate the 3D density field following the power spectrum fpk.

"""
function gen_deltaLk!(deltaL, deltak, fftplan, Finfo, fpk)
    # First, compute the forward FFT to get the complex delta(k) field
    forwardFT!(deltak, fftplan, deltaL)

    scale = sqrt(Finfo.Ntotal / Finfo.Volume)
    multiply_pk!(deltak, Finfo.akmag, fpk, scale)
    return nothing
end
"""
 multiply_pk!(deltak, akmag, fpk, scale)

Multiply the power spectrum to give the proper variance of deltak.
The normalization factor for the GRand3D! for rfft is sqrt[VP(k)/N].
Because `forwardFT!` introduces an extra V/N, we multiply by sqrt[P(k)N/V].
"""
function multiply_pk!(deltak, akmag, fpk, scale)
    @inbounds for idx in CartesianIndices(deltak)
        knum = akmag[idx]
        if knum > 0
            deltak[idx] *= sqrt(fpk(knum)) * scale
        else
            deltak[idx] = 0.0
        end
    end
    return nothing
end
# ------------------------------------------------------------------------------
# Computing mean and variance
# ------------------------------------------------------------------------------
function calc_mean_var(array, comm=nothing)
    isempty(array) && error("Array cannot be empty")
    T = eltype(array)
    local_sum = zero(T)
    local_sq_sum = zero(T)

    @floop for x in array
        @reduce(local_sum += x)
        @reduce(local_sq_sum += x^2)
    end

    local_n = length(array)

    # When using MPI
    if !isnothing(comm)
        MPI = _get_MPI()
        rank = MPI.Comm_rank(comm)
        global_sum = MPI.Allreduce(local_sum, +, comm)
        global_sq_sum = MPI.Allreduce(local_sq_sum, +, comm)
        global_n = MPI.Allreduce(local_n, +, comm)

        global_avg = global_sum / global_n
        global_var = global_sq_sum / global_n - global_avg^2

        rank == 0 && @info "rank $rank global_n, global_avg, global_var = $global_n $global_avg $global_var"

        return global_avg, global_var
    else
        # Single process case
        mean = local_sum / local_n
        var = local_sq_sum / local_n - mean^2
        return mean, var
    end
end
# ------------------------------------------------------------------------------
# Generate the 3D position of galaxies
# ------------------------------------------------------------------------------
"""
    gen_LN_galaxy_mock(outfname,nbar,deltar,Finfo,Nseed,xyzseed)

    Initiate the random number generator and generate the random realization of galaxies.

"""
function gen_LN_galaxy_mock(outfname,nbar,deltar,Finfo,Nseed,xyzseed)
    # generate the random variables
    rngX   = MersenneTwister(Nseed)
    rngxyz = MersenneTwister(xyzseed)

    Nptls_this = gen_LN_galaxy_mock(outfname,nbar,deltar,
        Finfo.xH1,Finfo.xH2,Finfo.xH3,
        Finfo.aix1,Finfo.aix2,Finfo.aix3,
        rngX,rngxyz)
    return Nptls_this
end
# -----------------------------------------------------------------------------
function gen_LN_galaxy_mock(outfname,nbar,deltar,xH1,xH2,xH3,aix1,aix2,aix3,rngX,rngxyz)

    Nptl_this = 0
    rxyz = Vector{Float64}(undef,3)
    
    mkpath(dirname(outfname))
    open(outfname,"w") do fout
        @inbounds for (i3_local, i3_global) in enumerate(aix3)
            x3min = xH3*(i3_global-1)
            @inbounds for (i2_local, i2_global) in enumerate(aix2)
                x2min = xH2*(i2_global-1)
                @inbounds for (i1_local, i1_global) in enumerate(aix1)
                    x1min = xH1*(i1_global-1)

                    λgal_this_cell = nbar*(1+deltar[i1_local,i2_local,i3_local])
                    ngal_this_cell = rand(rngX,Poisson(λgal_this_cell))
                    Nptl_this += ngal_this_cell

                    if ngal_this_cell>0
                        # write to the temporal output file
                        # local index for this cell
                        write(fout,[i1_local,i2_local,i3_local])
                        write(fout,ngal_this_cell)
                        for igal in 1:ngal_this_cell
                            rand!(rngxyz,rxyz)
                            x1 = x1min + rxyz[1]*xH1
                            x2 = x2min + rxyz[2]*xH2
                            x3 = x3min + rxyz[3]*xH3
                            write(fout,[x1,x2,x3])
                        end
                    end
        end;end;end
    end
    
    return Nptl_this
end
# ------------------------------------------------------------------------------
# Generate the lognormal galaxy
# ------------------------------------------------------------------------------
""" 
  _Lognormal_galaxy(fpkgA,Ngals,Finfo,Nseed,xyzseed,gtempfname_this_rank,deltak,deltar,v1,fftplan,comm=nothing)

 This is the main function of the Lognormal to compute the discrete realization of points following the log-normal distribution with a given power spectrum of the log-transformed field.

 <Input parameters>
 fpkA       = function returning the log-transformed power spectrum
 Nptls      = target number of particles 
 Finfo      = Fourier information
 Nseed      = random seed for generating the number of particles (Poisson realization)
 xyzseed    = random seed for particle position in the grid
 gtempfname = temporary output file name
 deltak, deltar, v1 = 3D grid
 fftplan    = FFT plan
 comm       = nothing (single/threads), MPI.Comm (MPI)
"""
function _Lognormal_galaxy(fpkgA,Nptls,Finfo,Nseed,xyzseed,gtempfname,deltak,deltar,v1,fftplan,comm=nothing)

    rank = comm === nothing ? 0 : _get_MPI().Comm_rank(comm)
    
    # Generating Gaussian (log-transformed) density field
    gen_deltaLk!(v1,deltak,fftplan,Finfo,fpkgA)

    # calculate the real-space density contrast from the Fourier transformation
    rank==0 && println("log-transformed field inverse FFT back to real space ...")
    @time inverseFT!(deltar,fftplan,deltak)

    # Compute the variance of the log-transformed field
    meanA, sigsqA = calc_mean_var(deltar,comm)

    rank==0 && println("Square of the log-density field of galaxy = $sigsqA")

    # transform from log-transformed A field to the LN density field
    @. deltar = exp(deltar)*exp(-sigsqA/2) - 1.0

    # Target number density per each cell
    nbar_target = Nptls/Finfo.Ntotal
    Nptls_result = gen_LN_galaxy_mock(gtempfname,nbar_target,deltar,Finfo,Nseed,xyzseed)

    if !isnothing(comm)
        Nptls_total = _get_MPI().Reduce(Nptls_result, +, 0, comm)
    else
        Nptls_total = Nptls_result
    end
    rank==0 && @info("Nptl: target = $Nptls, result = $Nptls_total")

    return Nptls_result, gtempfname
end
# -----------------------------------------------------------------------------
# Compute the velocity
# -----------------------------------------------------------------------------
"""
    ddot_to_velocity!(deltar,deltak,deltak2,v1,v2,v3,aHf,Finfo,fftplan)

 Calculate vx,vy,vz from -(aHf)*deltar = ∇⋅(vx,vy,vz) for irrotational vel.
 Note the sign and amplitude difference between this theta_to_velocity! in GridSPT code.
"""
function ddot_to_velocity!(deltar,deltak,deltak2,v1,v2,v3,aHf,Finfo,fftplan)
    ddot_to_velocity!(deltar,deltak,deltak2,v1,v2,v3,aHf,Finfo.akmag,Finfo.vk1,Finfo.vk2,Finfo.vk3,Finfo.Ntotal,fftplan)
end
# -----------------------------------------------------------------------------
function ddot_to_velocity!(deltar,deltak,deltak2,v1,v2,v3,aHf,akmag,vk1,vk2,vk3,Ntotal,fftplan)

    # Fourier transform to get deltak
    forwardFT!(deltak2,fftplan,deltar)

    # For regular FFTW arrays and PencilArrays
    @. deltak2 = deltak2 * (im/akmag^2*aHf)

    # Force the DC mode to be zero
    deltak2[1,1,1] = 0.0im

    # for v1 =============================================================
    copy!(deltak,deltak2)
    @. deltak = deltak * vk1
    # inverse Fourier transformation
    inverseFT!(v1,fftplan,deltak)

    # for v2 =============================================================
    copy!(deltak,deltak2)
    @. deltak = deltak * vk2
    # inverse Fourier transformation
    inverseFT!(v2,fftplan,deltak)

    # for v3 =============================================================
    copy!(deltak,deltak2)
    @. deltak = deltak * vk3
    # inverse Fourier transformation
    inverseFT!(v3,fftplan,deltak)
end
# -----------------------------------------------------------------------------
# Save the galaxy position data with the velocity
# -----------------------------------------------------------------------------
function _save_galaxy_velocity(outbase, galdata,
                                   Lbox1, Lbox2, Lbox3,
                                   nsub1, nsub2, nsub3, iLN, comm=nothing)

    rank = comm === nothing ? 0 : _get_MPI().Comm_rank(comm)

    if nsub1*nsub2*nsub3>1
        # Pre-allocate sub_galdata with the maximum size
        # (overestimate to avoid reallocation)
        max_gals_per_sub = _get_max_sub_galnum(galdata,Lbox1,Lbox2,Lbox3,nsub1,nsub2,nsub3)
        sub_galdata = Matrix{Float64}(undef, 6, max_gals_per_sub)
    
        isubindx = 0
        for isub3 in 1:nsub3
            for isub2 in 1:nsub2
                for isub1 in 1:nsub1
                    isubindx += 1
                
                    # Extract galaxies for this sub-box
                    ngals_this = _get_sub_galdata!(sub_galdata, galdata, 
                                                  Lbox1, Lbox2, Lbox3,
                                                  isub1, isub2, isub3,
                                                  nsub1, nsub2, nsub3)
                
                    if ngals_this > 0
                        # Each rank saves its contribution with a different outbase
                        outfname = outbase * "_sub$(isubindx).h5"
                        save_sub_galdata(outfname, sub_galdata, ngals_this)
                    end
                end
            end
        end
    else
        # No sub-division, save all data
        outfname = outbase*".h5"
        save_full_galdata(outfname, galdata)
    end
    
    if !isnothing(comm)
        _get_MPI().Barrier(comm)
    end

    # Optionally, gather info about which sub-boxes have data
    if rank == 0
        msg = isnothing(comm) ? 
              "Realization $iLN: Sub-box files written." :
              "Realization $iLN: Sub-box files written. Merging may be needed."
        @info msg
    end
end
# -----------------------------------------------------------------------------
function _get_max_sub_galnum(galdata,LboxX,LboxY,LboxZ,nsub1,nsub2,nsub3)

    Nsubs = nsub1*nsub2*nsub3
    aNgals = Vector{Int64}(undef,Nsubs)
    mm = Vector{Bool}(undef,size(galdata,2))

    isubs = 1
    @inbounds for isub1 in 1:nsub1
        @inbounds for isub2 in 1:nsub2
            @inbounds for isub3 in 1:nsub3

                Xmin = LboxX/nsub1*(isub1-1)
                Xmax = LboxX/nsub1*(isub1)
                Ymin = LboxY/nsub2*(isub2-1)
                Ymax = LboxY/nsub2*(isub2)
                Zmin = LboxZ/nsub3*(isub3-1)
                Zmax = LboxZ/nsub3*(isub3)
                
                @. mm = (Xmin <= galdata[1,:]) & (galdata[1,:] < Xmax) &
                        (Ymin <= galdata[2,:]) & (galdata[2,:] < Ymax) &
                        (Zmin <= galdata[3,:]) & (galdata[3,:] < Zmax)

                aNgals[isubs] = sum(mm)
                isubs += 1
    end;end;end

    return maximum(aNgals)
end
# -----------------------------------------------------------------------------
function _get_sub_galdata!(sub_galdata,galdata,LboxX,LboxY,LboxZ,isubX,isubY,isubZ,Nx,Ny,Nz)
    
    Xmin = LboxX/Nx*(isubX-1)
    Xmax = LboxX/Nx*(isubX)
    Ymin = LboxY/Ny*(isubY-1)
    Ymax = LboxY/Ny*(isubY)
    Zmin = LboxZ/Nz*(isubZ-1)
    Zmax = LboxZ/Nz*(isubZ)

    Xcent = (Xmin+Xmax)/2
    Ycent = (Ymin+Ymax)/2
    
    indx_in = 1
    @inbounds for gindx in 1:size(galdata, 2)
        X = galdata[1,gindx]
        Y = galdata[2,gindx]
        Z = galdata[3,gindx]

        # first cut: those in the subbox
        if (Xmin<=X<Xmax) && (Ymin<=Y<Ymax) && (Zmin<=Z<Zmax)
            # Transform to sub-box coordinates
            Xsub = X - Xcent
            Ysub = Y - Ycent
            Zsub = Z - Zmin 

            # array overflow warning 
            if indx_in > size(sub_galdata, 2)
                error("sub_galdata overflow: attempted to write at column $indx_in (max=$(size(sub_galdata,2)))")
            end
            
            sub_galdata[1,indx_in] = Xsub
            sub_galdata[2,indx_in] = Ysub
            sub_galdata[3,indx_in] = Zsub
            sub_galdata[4,indx_in] = galdata[4,gindx]
            sub_galdata[5,indx_in] = galdata[5,gindx]
            sub_galdata[6,indx_in] = galdata[6,gindx]

            indx_in += 1
        end
    end

    return indx_in - 1  # Number of galaxies in this sub-box
end
# -----------------------------------------------------------------------------
function save_full_galdata(outfname, galdata)
    mkpath(dirname(outfname))
    h5open(outfname, "w") do fLNout
        LNdata = create_group(fLNout, "Mock")
        LNdata["xyzvxvyvz"] = galdata
        LNdata["Ngals"] = size(galdata, 2)
    end
end
# -----------------------------------------------------------------------------
function save_sub_galdata(outfname, sub_galdata, ngals_this)
    if ngals_this > 0  # Only save if there are galaxies
        mkpath(dirname(outfname))
        h5open(outfname, "w") do fLNout
            LNdata = create_group(fLNout, "Mock")
            # unit: xyz in [Mpc], vx,vy,vz in [km/s]
            LNdata["xyzvxvyvz"] = sub_galdata[:, 1:ngals_this]
            LNdata["Ngals"] = ngals_this
        end
        return true
    end
    return false  # No file created
end
# -----------------------------------------------------------------------------
# Attach the velocity field to the galaxies
# -----------------------------------------------------------------------------
"""
    _Lognormal_add_velocity_and_save(fpkA,Nptls,Finfo,galfname,outfbase,deltak,deltar,v1,fftplan,aHf,nsubX,nsubY,nsubZ,comm=nothing)

Compute the velocity field from the log-normal density field following the given density power specturm.
"""
function _Lognormal_add_velocity_and_save(fpkA,Nptls,Finfo,galfname,outfbase,deltak,deltar,v1,fftplan,aHf,nsubX,nsubY,nsubZ,iLN,comm=nothing)

    rank = comm === nothing ? 0 : _get_MPI().Comm_rank(comm)

    # Generating Gaussian (log-transformed) density field
    gen_deltaLk!(v1,deltak,fftplan,Finfo,fpkA)

    # calculate the real-space density contrast from the Fourier transformation
    rank==0 && println("log-transformed field inverse FFT back to real space ...")
    @time inverseFT!(deltar,fftplan,deltak)

    # Compute the variance of the log-transformed field
    meanA, sigsqA = calc_mean_var(deltar,comm)
    rank==0 && println("Square of the log-density field of matter density = $sigsqA")

    # transform from log-transformed A field to the LN density field
    @. deltar = exp(deltar)*exp(-sigsqA/2) - 1.0

    # prepare for the array to compute the velocity field
    deltak2 = similar(deltak)    
    v2 = similar(v1); v3 = similar(v1)

    @time ddot_to_velocity!(deltar,deltak,deltak2,v1,v2,v3,aHf,Finfo,fftplan)
    
    # Array for storing the galaxy data
    galdata  = Array{Float64,2}(undef,(6,Nptls))
    gindices = Vector{Int64}(undef,3)
    gxyz     = Vector{Float64}(undef,3)

    igal = 1
    
    open(galfname,"r") do fgal
        # While reading coordinates of all Nptls galaxies
        while igal <= Nptls
            read!(fgal,gindices)
            
            # Assign the same velocity for all galaxies in this cell
            # The stored gindices are the local indices, so we can directly call v1,v2,v3
            @inbounds v1this = v1[gindices[1],gindices[2],gindices[3]]
            @inbounds v2this = v2[gindices[1],gindices[2],gindices[3]]
            @inbounds v3this = v3[gindices[1],gindices[2],gindices[3]]
            Nptl_this = read(fgal,Int64)
            for iptl_this = 1:Nptl_this
                read!(fgal,gxyz)
                galdata[1,igal]   = gxyz[1]
                galdata[2,igal]   = gxyz[2]
                galdata[3,igal]   = gxyz[3]
                galdata[4,igal]   = v1this
                galdata[5,igal]   = v2this
                galdata[6,igal]   = v3this
                igal += 1
            end
        end
    end
    _save_galaxy_velocity(outfbase,galdata,Finfo.L1,Finfo.L2,Finfo.L3,nsubX,nsubY,nsubZ,iLN,comm)

    return true
end
# -----------------------------------------------------------------------------
"""
    merge_rank_files(merge_base, nranks, nsub_total)

    Parameters:
    - merge_base: filebase*"_iLN" (e.g., "output_5")
    - nranks: total number of MPI ranks
    - nsub_total: product of nsub dimensions
"""
function merge_rank_files(merge_base, nranks, nsub_total)

    function collect_mock_data(infname)
        h5open(infname, "r") do file
            haskey(file, "Mock") || return nothing
            mock_group = file["Mock"]
            try
                haskey(mock_group, "xyzvxvyvz") || return nothing
                data = read(mock_group["xyzvxvyvz"])
                ngals = haskey(mock_group, "Ngals") ? read(mock_group["Ngals"]) : size(data, 2)
                return data, ngals
            finally
                close(mock_group)
            end
        end
    end

    function write_merged_mock(outfname, arrays, total_ngals)
        mkpath(dirname(outfname))
        h5open(outfname, "w") do file
            mock_group = create_group(file, "Mock")
            try
                nrows = size(arrays[1], 1)
                @assert all(size(arr, 1) == nrows for arr in arrays) "Mock datasets with inconsistent row counts"
                merged = hcat(arrays...)
                merged_ngals = size(merged, 2)
                if merged_ngals != total_ngals
                    @warn "Mismatch between concatenated Ngals and stored counts" stored=total_ngals computed=merged_ngals
                    total_ngals = merged_ngals
                end
                mock_group["xyzvxvyvz"] = merged
                mock_group["Ngals"] = total_ngals
            finally
                close(mock_group)
            end
        end
    end

    if nsub_total > 1 # When the sub-division exists
        for isubindx in 1:nsub_total
            arrays = AbstractMatrix[]
            total_ngals = 0

            for rank in 0:(nranks-1)
                infname = merge_base * "_$(rank)_sub$(isubindx).h5"
                isfile(infname) || continue

                data_pair = collect_mock_data(infname)
                isnothing(data_pair) && continue

                data, ngals = data_pair
                push!(arrays, data)
                total_ngals += ngals
            end

            isempty(arrays) && continue

            outfname = merge_base * "_sub$(isubindx).h5"
            write_merged_mock(outfname, arrays, total_ngals)
            println("Merged sub-box $(isubindx): $(outfname)")
        end
    else
        arrays = AbstractMatrix[]
        total_ngals = 0

        for rank in 0:(nranks-1)
            infname = merge_base * "_$(rank).h5"
            isfile(infname) || continue

            data_pair = collect_mock_data(infname)
            isnothing(data_pair) && continue

            data, ngals = data_pair
            push!(arrays, data)
            total_ngals += ngals
        end

        isempty(arrays) && return

        outfname = merge_base * ".h5"
        write_merged_mock(outfname, arrays, total_ngals)
        println("✓ Merged full data: $(outfname)")
    end
end
# ------------------------------------------------------------------------------
# Main driver: lognormal mock generation
# ------------------------------------------------------------------------------

"""
    run_lognormal(cfg::LNConfig)

Generate a 3D lognormal density field using the configuration `cfg`.
The FFT backend is selected globally through `DEFAULT_BACKEND`.
"""
function run_lognormal(cfg::LNConfig)
    @info "Running lognormal mock generation with backend $(DEFAULT_BACKEND[])"
   
    # FFT plan spec
    spec = FFTPlanSpec(cfg.dims, cfg.boxsize; comm=cfg.comm, backend=DEFAULT_BACKEND[])
    # Make the FFT plan
    fftplan = make_plan(spec)

    # define the rank, for single and threads, rank=0
    if cfg.comm !== nothing
        MPI = _get_MPI()
        rank = MPI.Comm_rank(cfg.comm)
        println("Total number of processes = $(MPI.Comm_size(cfg.comm)), and my rank is $rank.")
    else
        rank = 0
    end
    
    # compute the aHf and broadcast the value
    val  = rank == 0 ? compute_aHf(cfg.z) : 0.0
    aHfzred = broadcast_scalar(val, cfg.comm)

    # make the output path
    mkpath(cfg.outdir)

    # -- Allocate arrays
    allocmem = @allocated etime = @elapsed begin
        if spec.backend == :pencil_mpi
            deltar, deltak = allocate_fields(spec; fftplan=fftplan)
            Finfo = FourierArrayInfo(spec; plan=fftplan)
        else
            deltar, deltak = allocate_fields(spec)
            Finfo = FourierArrayInfo(spec)
        end
        v1 = similar(deltar)
    end
    rank==0 && @info "setting up a total of $(allocmem/1024^3)GB Grids in $etime seconds!"

    # power spectrum function
    fpkA = rank == 0 ? get_fpk(cfg.pkA_file) : nothing
    fpkA = broadcast_object(fpkA, cfg.comm)
    fpkgA = rank == 0 ? get_fpk(cfg.pkgA_file) : nothing
    fpkgA = broadcast_object(fpkgA, cfg.comm)
    
    # generate the random variables: seed = seed + rank
    rng   = MersenneTwister(cfg.seed + rank)

    # filebase 
    filebase = joinpath(cfg.outdir,cfg.outhead)
    
    for iLN in 1:cfg.nreal
        # seed must be positive, so generate in UInt32 type
        rseed, Nseed, xyzseed = rand(rng,UInt32,3)
        
        # Generating the 3D Gaussian (log-transformed) density field
        # We will use this 3D field for both galaxy and matter (same cosmic variance)
        if cfg.computeGRF
            etime = @elapsed gen_GRand3d!(v1,rseed,cfg.comm)
            @info("generating grf in $etime seconds...")
            if cfg.writeGRF
                GRFfile = joinpath(cfg.outdir,cfg.GRFfilehead*"_$iLN.h5")
                write_GRand3d(v1,GRFfile)
            end
        else
            GRFfile = joinpath(cfg.outdir,cfg.GRFfilehead*"_$iLN.h5")
            read_GRand3d!(v1,GRFfile,Finfo)
        end

        gtempfname_this_rank = filebase*"_galtmp_$(iLN)_$rank.bin"
        Nptl_this, gtempfname_result = _Lognormal_galaxy(fpkgA,cfg.Ngals,Finfo,Nseed,xyzseed,gtempfname_this_rank,deltak,deltar,v1,fftplan,cfg.comm)

        # Of course, we have to use the same seed for the galaxy density and velocity field, to make them in phase.
        outbase = isnothing(cfg.comm) ? filebase*"_$iLN" : filebase*"_$(iLN)_$rank"

        success = _Lognormal_add_velocity_and_save(fpkA,Nptl_this,Finfo,gtempfname_result,outbase,deltak,deltar,v1,fftplan,aHfzred,cfg.nsub[1],cfg.nsub[2],cfg.nsub[3],iLN,cfg.comm)

        # remove the temp file
        if success
            rm(gtempfname_result)
        end

        # Merge files if necessary
        if !isnothing(cfg.comm)
            MPI = _get_MPI()
            MPI.Barrier(cfg.comm)
            if rank==0
                nranks = MPI.Comm_size(cfg.comm)
                nsub_total = prod(cfg.nsub)
                merge_rank_files(filebase*"_$iLN", nranks, nsub_total)
            end
        end
    end

    return
end

# ============================================================================
end # module
# ============================================================================
