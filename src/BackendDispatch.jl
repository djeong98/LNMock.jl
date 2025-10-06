# ==============================================================================
# >> BackendDispatch.jl
#
# This file handles all FFT-backend-dependent dispatching and utilities,
# including:
#   - make_plan()  (FFTW single, threaded, PencilMPI)
#   - forwardFT!, inverseFT!
#   - allocate_input/output() for PencilFFTs
#   - FourierArrayInfo struct and backend constructors
#
# By isolating these here, the Core physics code stays clean and backend-agnostic.
# ==============================================================================
module BackendDispatch
# ==============================================================================

export FFTPlanSpec, make_plan, forwardFT!, inverseFT!,
       FourierArrayInfo, allocate_fields


Base.@kwdef struct FFTPlanSpec
    dims::NTuple{3,Int}
    boxsize::NTuple{3,Real}
    volume::Real
    comm::Union{Nothing,Any} = nothing
    backend::Symbol = :fftw_single
end

# Always include FFTW backends (FFTW is a hard dependency)
include(joinpath(@__DIR__,"Backends","FFTW_Single.jl"))
include(joinpath(@__DIR__,"Backends","FFTW_Threads.jl"))
using .FFTW_Single
using .FFTW_Threads

# MPI/PencilFFTs backend will be loaded conditionally at __init__ time
# Only include the module definition here - it will be populated later if needed
module PencilFFTs_MPI
    # Empty placeholder - will be populated by include() if MPI is available
end

# Helper functions for PencilFFTs API compatibility
# These operate on the wrapped Plan struct and extract the inner PencilFFTPlan
# Will only work if PencilFFTs backend is loaded
function size_in end
function size_out end
function pencil_in end
function pencil_out end



function FFTPlanSpec(dims::NTuple{3,Int}, boxsize::NTuple{3,Real};
                     comm=nothing, backend=:fftw_single)
    return FFTPlanSpec(; dims=dims, boxsize=boxsize, volume=prod(boxsize),
                comm=comm, backend=backend)
end


function forwardFT!(out, plan, inp; backend=:fftw_single)
    if backend == :fftw_single
        return FFTW_Single.forwardFT!(out, plan, inp)
    elseif backend == :fftw_threads
        return FFTW_Threads.forwardFT!(out, plan, inp)
    elseif backend == :pencil_mpi
        return PencilFFTs_MPI.forwardFT!(out, plan, inp)
    end
end


function inverseFT!(out, plan, inp; backend=:fftw_single)
    if backend == :fftw_single
        return FFTW_Single.inverseFT!(out, plan, inp)
    elseif backend == :fftw_threads
        return FFTW_Threads.inverseFT!(out, plan, inp)
    elseif backend == :pencil_mpi
        return PencilFFTs_MPI.inverseFT!(out, plan, inp)
    end
end

function make_plan(spec::FFTPlanSpec; backend=:fftw_single)
    if spec.backend == :fftw_single
        return FFTW_Single.make_plan(spec)
    elseif spec.backend == :fftw_threads
        return FFTW_Threads.make_plan(spec)
    elseif spec.backend == :pencil_mpi
        return PencilFFTs_MPI.make_plan(spec)
    else
        error("Unknown backend $(spec.backend)")
    end
end
# ====================================================================
struct FourierArrayInfo
    n1::Integer
    n2::Integer
    n3::Integer
    Ntotal::Integer
    cn1::Integer
    cn2::Integer
    cn3::Integer
    L1::Real
    L2::Real
    L3::Real
    Volume::Real
    # Fourier-space arrays: k
    kF1::Real
    kF2::Real
    kF3::Real
    ak1::Vector{Real}
    ak2::Vector{Real}
    ak3::Vector{Real}
    aik1::Vector{Integer}
    aik2::Vector{Integer}
    aik3::Vector{Integer}
    # real-space arrays: x
    xH1::Real
    xH2::Real
    xH3::Real
    ax1::Vector{Real}
    ax2::Vector{Real}
    ax3::Vector{Real}
    aix1::Vector{Integer}
    aix2::Vector{Integer}
    aix3::Vector{Integer}
    vk1
    vk2
    vk3
    akmag
end
# -----------------------------------------------------------------------------
function calcWavenumbers!(ak,kF,n,cn)
    @inbounds for indx = 1:cn
        ak[indx] = kF*(indx-1)
    end
    if(n > cn)
        @inbounds for indx = cn+1:n
            ak[indx] = kF*(indx-1-n)
        end
    end
end
# -----------------------------------------------------------------------------
function calcWaveindices!(aik,n,cn)
    @inbounds for indx = 1:cn
        aik[indx] = (indx-1)
    end
    if(n > cn)
        @inbounds for indx = cn+1:n
            aik[indx] = (indx-1-n)
        end
    end
end
# -----------------------------------------------------------------------------
function FourierArrayInfo(spec::FFTPlanSpec;plan=nothing)
    n1,n2,n3 = spec.dims
    L1,L2,L3 = spec.boxsize

    Ntotal = prod(spec.dims)
    Volume = prod(spec.boxsize)
    
    # to compute the maximum 1D wavelength
    cn1 = div(n1,2)+1
    cn2 = div(n2,2)+1
    cn3 = div(n3,2)+1
    # Fundamental frequencies
    kF1 = 2π/L1; kF2 = 2π/L2; kF3 = 2π/L3
    # 1D Fourier arrays
    ak1 = Vector{Float64}(undef,cn1)
    ak2 = Vector{Float64}(undef,n2)
    ak3 = Vector{Float64}(undef,n3)
    calcWavenumbers!(ak1,kF1,cn1,cn1)
    calcWavenumbers!(ak2,kF2,n2,cn2)
    calcWavenumbers!(ak3,kF3,n3,cn3)
    aik1 = Vector{Int64}(undef,cn1)
    aik2 = Vector{Int64}(undef,n2)
    aik3 = Vector{Int64}(undef,n3)
    calcWaveindices!(aik1,cn1,cn1)
    calcWaveindices!(aik2,n2,cn2)
    calcWaveindices!(aik3,n3,cn3)

    # real-space spacing
    xH1 = L1/n1; xH2 = L2/n2; xH3 = L3/n3;
    # 1D real-space arrays
    aix1 = collect(1:n1)
    aix2 = collect(1:n2)
    aix3 = collect(1:n3)
    ax1 = aix1.*xH1
    ax2 = aix2.*xH2
    ax3 = aix3.*xH3

    # For PencilFFTs (MPI), we compute the local array
    if spec.backend == :pencil_mpi
        @assert plan !== nothing "PencilFFT backend requires a valid plan"

        # --- Local dimensions (complex space)
        local_dims_k = size_out(plan)
        pen_k = pencil_out(plan)
        i1_range, i2_range, i3_range = range_local(pen_k, (1, 1, 1))

        # Local Fourier 1D arrays
        ak1_loc = ak1[i1_range]  # Local k1 wavenumbers
        ak2_loc = ak2[i2_range]  # Local k2 wavenumbers  
        ak3_loc = ak3[i3_range]  # Local k3 wavenumbers
        aik1_loc = aik1[i1_range]  # Local k1 indices
        aik2_loc = aik2[i2_range]  # Local k2 indices
        aik3_loc = aik3[i3_range]  # Local k3 indices

        # --- Local real-space dimensions
        pen_x = pencil_in(plan)
        local_dims_x = size_in(plan)
        ix1_range, ix2_range, ix3_range = range_local(pen_x, (1, 1, 1))

        aix1_loc = aix1[ix1_range]
        aix2_loc = aix2[ix2_range]
        aix3_loc = aix3[ix3_range]
        ax1_loc = ax1[ix1_range]
        ax2_loc = ax2[ix2_range]
        ax3_loc = ax3[ix3_range]

        # --- Construct local k-grids and k-magnitude
        local_n1, local_n2, local_n3 = local_dims_k
        vk1 = ones(Float64, local_n1, 1, 1)
        vk2 = ones(Float64, 1, local_n2, 1)
        vk3 = ones(Float64, 1, 1, local_n3)
        vk1 .*= ak1_loc
        @view(vk2[1, :, 1]) .= ak2_loc
        @view(vk3[1, 1, :]) .= ak3_loc
        
        # local k-magnitude array
        akmag = Array{Float64}(undef, local_n1, local_n2, local_n3)
        @. akmag = hypot(vk1, vk2, vk3)

        return FourierArrayInfo(
            n1, n2, n3, Ntotal, cn1, cn2, cn3,
            L1, L2, L3, Volume,
            kF1, kF2, kF3,
            ak1_loc, ak2_loc, ak3_loc,
            aik1_loc, aik2_loc, aik3_loc,
            xH1, xH2, xH3,
            ax1_loc, ax2_loc, ax3_loc,
            aix1_loc, aix2_loc, aix3_loc,
            vk1, vk2, vk3,
            akmag
        )
    end
    
    # Default: FFTW (serial / threads)         
    vk1   = ones(Float64, cn1, 1,1)
    vk2   = ones(Float64, 1, n2 ,1)
    vk3   = ones(Float64, 1, 1, n3)
    akmag = Array{Float64,3}(undef,(cn1,n2,n3))
    
    vk1 .*= ak1
    @view(vk2[1,:,1]) .= ak2
    @view(vk3[1,1,:]) .= ak3
    @. akmag = hypot(vk1, vk2, vk3)

    return FourierArrayInfo(n1,n2,n3,Ntotal,cn1,cn2,cn3,
        L1,L2,L3,Volume,
        kF1,kF2,kF3,
        ak1,ak2,ak3,
        aik1,aik2,aik3,
        xH1,xH2,xH3,
        ax1,ax2,ax3,
        aix1,aix2,aix3,
        vk1,vk2,vk3,
        akmag)
end
# ------------------------------------------------------------------------------
# Internal helper: allocate arrays
# ------------------------------------------------------------------------------
"""
    allocate_fields(cfg::LNConfig, fftplan)

Allocate real-space and Fourier-space arrays for the mock field.
For PencilFFTs (MPI), this uses `allocate_input` and `allocate_output`.
For FFTW backends, it uses standard Array allocation.
"""
function allocate_fields(spec::FFTPlanSpec;fftplan=nothing)
    n1, n2, n3 = spec.dims

    if spec.backend == :pencil_mpi
       @assert fftplan !== nothing "PencilFFT backend requires a valid plan"
        # PencilFFTs handles local array shapes internally
        # Extract the inner PencilFFTPlan from the wrapped Plan struct
        inner_plan = fftplan.plan
        field_real    = allocate_input(inner_plan)
        field_fourier = allocate_output(inner_plan)
    else
        field_real    = Array{Float64}(undef, n1, n2, n3)
        field_fourier = Array{ComplexF64}(undef, div(n1,2)+1, n2, n3)
    end

    return field_real, field_fourier
end

# ==============================================================================
end
# ==============================================================================
