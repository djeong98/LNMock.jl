# ============================================================================
# Module LNMock
#
# Generate the lognormal mock galaxy distribution with velocity field from
# a given matter and galaxy power spectra.
#
# 16 September 2025
# Donghui Jeong
# ============================================================================
module LNMock
# ============================================================================

using Logging, Base.Threads
using CosmoFFTs: FFTPlanSpec, FourierArrayInfo, allocate_fields,
                     make_plan, forwardFT!, inverseFT!
using CosmoFFTs

# Expose FFT configuration handles from the standalone package
const DEFAULT_BACKEND = CosmoFFTs.DEFAULT_BACKEND
const DEFAULT_THREADS = CosmoFFTs.DEFAULT_THREADS
const WISDOM_FILE     = CosmoFFTs.WISDOM_FILE
const BackendDispatch = CosmoFFTs.BackendDispatch

include(joinpath(@__DIR__, "Utils", "Utils.jl"))
include(joinpath(@__DIR__, "Utils", "Cosmology.jl"))
using .Cosmology: set_cosmology!, get_cosmology, compute_aHf, speedoflight
include(joinpath(@__DIR__, "Spectra.jl"))
include(joinpath(@__DIR__, "Core.jl"))
using .Core: LNConfig, run_lognormal

export LNConfig, compute_aHf, run_lognormal
export FFTPlanSpec, FourierArrayInfo, make_plan, forwardFT!, inverseFT!, allocate_fields
export init_cosmology_mpi!

"""
    init_cosmology_mpi!(; H0=0.6766, Om=0.3111, Ol=0.6889, Ob=0.048975)

Initialize cosmology for MPI backend. Must be called explicitly in MPI scripts
after MPI is initialized. Only rank 0 computes cosmology; result is broadcast to all ranks.
"""
function init_cosmology_mpi!(; H0=nothing, Om=nothing, Ol=nothing, Ob=nothing)
    using MPI
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    # Parse environment or use defaults
    H0_val = H0 !== nothing ? H0 : parse(Float64, get(ENV, "LN_H0", "0.6766"))
    Om_val = Om !== nothing ? Om : parse(Float64, get(ENV, "LN_OMEGA_M", "0.3111"))
    Ol_val = Ol !== nothing ? Ol : parse(Float64, get(ENV, "LN_OMEGA_L", "0.6889"))
    Ob_val = Ob !== nothing ? Ob : parse(Float64, get(ENV, "LN_OMEGA_B", "0.048975"))

    local cosmo_obj
    if rank == 0
        cosmo_obj = set_cosmology!(H0=H0_val, Om=Om_val, Ol=Ol_val, Ob=Ob_val,
                                   calc_growth=true, Tcmb=0.0, unit="Mpc")
        @info "Cosmology initialized on rank 0" H0=H0_val Om=Om_val Ol=Ol_val Ob=Ob_val
    else
        cosmo_obj = nothing
    end

    # Broadcast cosmology object from rank 0 to all ranks
    cosmo_obj = MPI.bcast(cosmo_obj, 0, comm)

    # Set broadcasted cosmology on non-root ranks
    if rank != 0
        Cosmology._COSMO[] = cosmo_obj
    end

    return nothing
end

# Include initialization code
include(joinpath(@__DIR__, "__init__.jl"))

# ============================================================================
end
# ============================================================================
