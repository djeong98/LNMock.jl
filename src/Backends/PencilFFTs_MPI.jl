module PencilFFTs_MPI

using FFTW, PencilFFTs
using LinearAlgebra: mul!, ldiv!
using ..BackendDispatch: FFTPlanSpec

export Plan, make_plan #, forwardFT!, inverseFT!, raw_rfft!, raw_irfft!, allocate_input, allocate_output, size_in, size_out, pencil_in, pencil_out, range_local

"""
    Plan

Wrapper for a PencilFFTs real R2C plan that supports both directions:
- mul!(F, plan, A)   → forward (unnormalized)
- ldiv!(A, plan, F)  → inverse (**includes 1/N**)
We post-scale by (N/V) to achieve 1/V real-space normalization.
"""
Base.@kwdef struct Plan
    plan::PencilFFTPlan     # RFFT plan
    N::Int
    V::Float64
end

"""
    make_plan((n1,n2,n3), V; comm=MPI.COMM_WORLD) -> Plan

Create a pencil decomposition and RFFT plan. Arrays should be allocated with
`allocate_input(plan)` / `allocate_output(plan)` for correct local shapes.
"""
function make_plan(spec::FFTPlanSpec)
    n1, n2, n3 = spec.dims
    comm = spec.comm

    pen  = Pencil((n1, n2, n3), comm)
    tform = Transforms.RFFT()
    p    = PencilFFTPlan(pen, tform)   # single plan handles fwd/inv

    V   = spec.volume
    N    = n1*n2*n3
    return Plan(; plan=p, N, V=Float64(V))
end

# -------------------------------------------------------
# Raw-level wrappers
# Note: There's no brfft! in PencilFFTs
# -------------------------------------------------------

"""
    raw_rfft!(out, plan, inp)

Low-level forward transform (real-to-complex) with PencilFFTs.

Mathematical form (unnormalized DFT):
    out(k) = ∑_x inp(x) · exp(-i k·x)

[[Caution]] No normalization is applied.
"""
raw_rfft!(out, plan, inp) = mul!(out, plan, inp)


"""
    raw_irfft!(out, plan, inp)

Inverse transform (complex-to-real) using `ldiv!` in PencilFFTs.

Mathematical form:
    out(x) = (1/Ntotal) ∑_k inp(k) · exp(+i k·x)

[[Caution]] Note: PencilFFTs automatically includes 1/Ntotal normalization here.
There is no separate unnormalized complex→real (brfft) transform in PencilFFTs.
"""
raw_irfft!(out, plan, inp) = ldiv!(out, plan, inp)


# -------------------------------------------------------
# Physics-level (normalized) wrappers
# -------------------------------------------------------

function forwardFT!(out, plan::Plan, inp)
    return forwardFT!(out, plan.plan, inp, plan.V, plan.N)
end

function inverseFT!(out, plan::Plan, inp)
    return inverseFT!(out, plan.plan, inp, plan.V, plan.N)
end

# -------------------------------------------------------
# Physics-level (normalized) core
# -------------------------------------------------------

"""
    forwardFT!(out, plan, inp, V, N)

Physics-level forward Fourier transform following cosmology convention:

    X(k) ≈ (V/N) ∑_x X(x) · exp(-i k·x)

Wraps `raw_rfft!` and applies the (V/N) normalization factor.
"""
function forwardFT!(out, plan, inp, V::Float64, N::Int)
    raw_rfft!(out, plan, inp)
    out .*= V / N
    return out
end


"""
    inverseFT!(out, plan, inp, V, N)

Physics-level inverse Fourier transform following cosmology convention:

    X(x) ≈ (1/V) ∑_k X(k) · exp(+i k·x)

PencilFFTs only provides a normalized irfft (`ldiv!`), which already applies 1/Ntotal.
We multiply by Ntotal/V to recover the cosmological convention.
"""
function inverseFT!(out, plan, inp, V::Float64, N::Int)
    raw_irfft!(out, plan, inp)    # includes 1/N automatically
    out .*= N / V                # apply N/V to match 1/V convention
    return out
end


end # module
