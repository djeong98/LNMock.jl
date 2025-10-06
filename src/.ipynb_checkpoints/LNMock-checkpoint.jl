# ============================================================================
# Module LNmock
#
# Generate the lognormal mock galaxy distribution with velocity field from
# a given matter and galaxy power spectra.
#
# 16 September 2025
# Donghui Jeong
# ============================================================================
module LNMock
# ============================================================================

using FFTW, Logging, Base.Threads
include("Utils/Utils.jl")
include("Utils/Wisdom.jl")
include("Utils/Cosmology.jl")
using .Wisdom: wisdom_filename, load_wisdom, save_wisdom, reset_wisdom
using .Cosmology: set_cosmology!, get_cosmology, compute_aHf, speedoflight

# global refs
const DEFAULT_BACKEND = Ref(:fftw_threads)
const DEFAULT_THREADS = Ref(Threads.nthreads())

include("BackendDispatch.jl")
include("Spectra.jl")
include("Core.jl")

include("__init__.jl")

# ============================================================================
end
# ============================================================================
