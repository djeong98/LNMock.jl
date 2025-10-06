# ============================================================================
# src/Spectra.jl
# ============================================================================

module Spectra

using Interpolations, DelimitedFiles

export get_fpk, make_extrap_fn

""" 
Function to read the pre-computed pk result
"""
function get_fpk(pkfname;skipstart=1)
    pkdata = readdlm(pkfname,skipstart=skipstart)
    return make_extrap_fn(pkdata[:,1],pkdata[:,2])
end

"""
Function to interpolate the power spectrum
"""
function make_extrap_fn(k::AbstractVector, P::AbstractVector)
    core = interpolate((k,), P, Gridded(Linear()))
    # set the value outside of the range all zero
    return extrapolate(core, 0.0)
end

# ============================================================================
end # of module
# ============================================================================