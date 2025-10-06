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

using Logging, Base.Threads
using FFTW, PencilFFTs
include(joinpath(@__DIR__,"Utils","Utils.jl"))
include(joinpath(@__DIR__,"Utils","Wisdom.jl"))
include(joinpath(@__DIR__,"Utils","Cosmology.jl"))
using .Wisdom: wisdom_filename, load_wisdom, save_wisdom, reset_wisdom
using .Cosmology: set_cosmology!, get_cosmology, compute_aHf, speedoflight

# global refs
const DEFAULT_BACKEND = Ref(:fftw_single)
const DEFAULT_THREADS = Ref(1) #Threads.nthreads())
const WISDOM_FILE     = Ref{String}("")

include(joinpath(@__DIR__,"BackendDispatch.jl"))
using .BackendDispatch: FFTPlanSpec, FourierArrayInfo, allocate_fields
include(joinpath(@__DIR__,"Spectra.jl"))
include(joinpath(@__DIR__,"Core.jl"))
using .Core: LNConfig, run_lognormal

export LNConfig, compute_aHf, run_lognormal
export FFTPlanSpec, make_plan, forwardFT!, inverseFT!, allocate_fields


to_iterable(x) = isa(x, Symbol) ? (x,) : x

"""
    load_backend(modsyms, filepath)

Load one or more external modules (e.g. FFTW, MPI, PencilFFTs) via `@eval       using`,
then include the backend file and `using` its module.

- `modsyms` can be a `Symbol` or a vector of Symbols.
- `filepath` is the path to the backend `.jl` file (e.g. "Backends/FFTW_Single. jl").
"""
function load_backend(modsyms, filepath::String)

#    for sym in to_iterable(modsyms)
#       @eval @__MODULE__ using $sym
#   end

    include(joinpath(@__DIR__,filepath))

    # Extract the module name from filename, e.g. "FFTW_Single.jl" →            "FFTW_Single"
    modname = Symbol(basename(filepath)[1:end-3])
    @eval using .$(modname)
end

# Include initialization code
include(joinpath(@__DIR__,"__init__.jl"))

"""
    make_plan(spec)

Create a backend-specific FFT plan using the default backend.
"""
function make_plan(spec::BackendDispatch.FFTPlanSpec)
    return BackendDispatch.make_plan(spec; backend=DEFAULT_BACKEND[])
end


"""
    forwardFT!(out, plan, inp)

Wrapper for backend-specific forward Fourier transform using the default backend.
"""
function forwardFT!(out, plan, inp)
    return BackendDispatch.forwardFT!(out, plan, inp; backend=DEFAULT_BACKEND[])
end

"""
    inverseFT!(out, plan, inp)

Wrapper for backend-specific inverse Fourier transform using the default backend.
"""
function inverseFT!(out, plan, inp)
    return BackendDispatch.inverseFT!(out, plan, inp; backend=DEFAULT_BACKEND[])
end

# ============================================================================
end
# ============================================================================
