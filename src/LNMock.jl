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

# Include initialization code
include(joinpath(@__DIR__, "__init__.jl"))

# ============================================================================
end
# ============================================================================
