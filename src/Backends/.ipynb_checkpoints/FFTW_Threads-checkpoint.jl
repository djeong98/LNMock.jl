module FFTW_Threads

using ..BackendDispatch: FFTPlanSpec
using ..LNMock: DEFAULT_THREADS

export Plan, make_plan #, forwardFT!, inverseFT!, raw_rfft!, raw_brfft!, raw_irfft!

"""
    Plan

Thin wrapper for FFTW real<->complex plans (R2C/C2R) with multi-threading.
"""
Base.@kwdef struct Plan
    rplan::FFTW.rFFTWPlan{Float64, -1, true, 3}
    iplan::FFTW.brFFTWPlan{Float64, 1, true, 3}
    N::Int
    V::Float64
    nthreads::Int
end

"""
    make_plan((n1,n2,n3), V; nthreads=FFTW.get_num_threads()) -> Plan

Creates threaded FFTW plans. Set threads BEFORE plan creation.
(Your __init__.jl already imports wisdom & sets threads; pass nthreads if you want to override.)
"""
function make_plan(spec::FFTPlanSpec)
    n1, n2, n3 = spec.dims
    N = n1*n2*n3
    V = spec.volume
    cn1 = div(n1,2)+1

    FFTW.set_num_threads(DEFAULT_THREADS[])

    A = Array{Float64}(undef, n1, n2, n3)
    F = Array{ComplexF64}(undef, cn1, n2, n3)

    # Try wisdom-only first? You can add flags=WISDOM_ONLY if you prefer.
    rplan = plan_rfft(A, (1,2,3))
    iplan = plan_brfft(F, n1, (1,2,3))

    return Plan(; rplan, iplan, N, V=Float64(V), nthreads)
end

# -------------------------------------------------------
# Raw-level wrappers
# -------------------------------------------------------

"""
    raw_rfft!(out, plan, inp)

Threaded forward transform (real-to-complex) with FFTW.

Mathematical form (unnormalized DFT):
    out(k) = ∑_x inp(x) · exp(-i k·x)

[[Caution]] No normalization is applied.
"""
raw_rfft!(out, plan, inp) = mul!(out, plan, inp)


"""
    raw_brfft!(out, plan, inp)

Threaded backward transform (complex-to-real) with FFTW.

Mathematical form (unnormalized inverse DFT):
    out(x) = ∑_k inp(k) · exp(+i k·x)

[[Caution]] No normalization is applied.
"""
raw_brfft!(out, bplan, inp) = mul!(out, bplan, inp)


"""
    raw_irfft!(out, plan, inp)

Threaded inverse transform using `ldiv!` in FFTW.

Mathematical form:
    out(x) = (1/Ntotal) ∑_k inp(k) · exp(+i k·x)

This version automatically includes a factor of 1/Ntotal.
"""
raw_irfft!(out, plan, inp) = ldiv!(out, plan, inp)


# -------------------------------------------------------
# Physics-level (normalized) wrappers
# -------------------------------------------------------

function forwardFT!(out, plan::Plan, inp)
    return forwardFT!(out, plan.rplan, inp, plan.V, plan.N)
end

function inverseFT!(out, plan::Plan, inp)
    return inverseFT!(out, plan.iplan, inp, plan.V, plan.N)
end

# -------------------------------------------------------
# Physics-level (normalized) core
# -------------------------------------------------------

"""
    forwardFT!(out, plan, inp, V, N)

Physics-level forward transform (multi-threaded FFTW) with cosmology normalization:

    X(k) ≈ (V/N) ∑_x X(x) · exp(-i k·x)
"""
function forwardFT!(out, plan, inp, V::Float64, N::Int)
    raw_rfft!(out, plan, inp)
    out .*= V/N
    return out
end

"""
    inverseFT!(out, plan, inp, V)

Physics-level inverse transform (multi-threaded FFTW) with cosmology normalization:

    X(x) ≈ (1/V) ∑_k X(k) · exp(+i k·x)
"""
function inverseFT!(out, plan, inp, V::Float64, N::Int)
    raw_irfft!(out, plan, inp)
    out .*= N/V
    return out
end

end # module
