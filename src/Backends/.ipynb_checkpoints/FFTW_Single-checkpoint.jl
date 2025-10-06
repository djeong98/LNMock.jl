# ============================================================================
module FFTW_Single
# ============================================================================

using ..BackendDispatch: FFTPlanSpec


export Plan, make_plan #, forwardFT!, inverseFT!, raw_rfft!, raw_brfft!, raw_irfft!

# -------------------------------------------------------
# FFT Plan, with information needed for normalization
# -------------------------------------------------------

"""
    Plan

Thin wrapper for FFTW real<->complex plans (R2C/C2R).
- rplan: plan_rfft (forward, unnormalized)
- iplan: plan_brfft (inverse, unnormalized)
- N: total grid cells (n1*n2*n3)
- V: box volume
"""
Base.@kwdef struct Plan
    rplan::FFTW.rFFTWPlan{Float64, -1, true, 3}   # plan_rfft
    iplan::FFTW.brFFTWPlan{Float64, 1, true, 3}   # plan_brfft
    N::Int
    V::Float64
end
 
"""
    make_plan((n1,n2,n3), V) -> Plan

Creates single-threaded FFTW plans (R2C forward + C2R inverse).
Uses current array layout; call with arrays of shape (n1,n2,n3) and (div(n1,2)+1,n2,n3).
"""
function make_plan(spec::FFTPlanSpec)
    n1, n2, n3 = spec.dims
    N   = n1*n2*n3
    V   = spec.volume
    cn1 = div(n1,2)+1

    # Single-thread backend: force FFTW to 1 thread for planning & execution.
    # (This sets a global FFTW knob — consistent with the "single" backend.)
    FFTW.set_num_threads(1)

    # Dummy arrays only for planning (aliases are fine; user will pass real ones to fwd!/inv!)
    A = Array{Float64}(undef, n1, n2, n3)
    F = Array{ComplexF64}(undef, cn1, n2, n3)

    rplan = plan_rfft(A, (1,2,3))                 # forward (unnormalized)
    iplan = plan_brfft(F, n1, (1,2,3))            # inverse (unnormalized)

    return Plan(; rplan, iplan, N, V=Float64(V))
end


# -------------------------------------------------------
# Raw-level wrappers
# -------------------------------------------------------

"""
    raw_rfft!(out, plan, inp)

Low-level forward transform (real-to-complex) with FFTW.

Mathematical form (unnormalized DFT):
    out(k) = ∑_x inp(x) · exp(-i k·x)

[[Caution]] No normalization is applied.
"""
raw_rfft!(out, plan, inp) = mul!(out, plan, inp)


"""
    raw_brfft!(out, plan, inp)

Low-level backward transform (complex-to-real) with FFTW.

Mathematical form (unnormalized inverse DFT):
    out(x) = ∑_k inp(k) · exp(+i k·x)

[[Caution]] No normalization is applied.
"""
raw_brfft!(out, bplan, inp) = mul!(out, bplan, inp)


"""
    raw_irfft!(out, plan, inp)

Inverse transform using `ldiv!` in FFTW.

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

Physics-level forward transform following cosmology convention:

    X(k) ≈ (V/N) ∑_x X(x) · exp(-i k·x)

This wraps `raw_rfft!` and then multiplies the result by V/N.
"""
function forwardFT!(out, plan, inp, V::Float64, N::Int)
    raw_rfft!(out, plan, inp)
    out .*= V/N
    return out
end

"""
    inverseFT!(out, plan, inp, V)

Physics-level inverse transform following cosmology convention:

    X(x) ≈ (1/V) ∑_k X(k) · exp(+i k·x)

This wraps `raw_brfft!` and then multiplies the result by 1/V.
"""
function inverseFT!(out, plan, inp, V::Float64, N::Int)
    raw_irfft!(out, plan, inp)
    out .*= N/V
    return out
end

# ============================================================================
end # module
# ============================================================================
