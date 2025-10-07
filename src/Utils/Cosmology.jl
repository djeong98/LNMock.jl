# ===========================================================================
module Cosmology
# ===========================================================================

using cosmology

export set_cosmology!, get_cosmology, compute_aHf, speedoflight

# Physical constant used in your code (km/s)
const speedoflight = 2.99792458e5

# Store the current cosmology in a const Ref (fast & mutable contents)
const _COSMO = Ref{Any}(nothing)

"""
    set_cosmology!(; H0=0.6766, Om=0.3111, Ol=0.6889, Ob=0.048975,
                    calc_growth=true, Tcmb=0.0, unit="Mpc")

Initialize and cache a cosmology (default: Planck18-like). Call once at startup
or whenever you want to switch models.
"""
function set_cosmology!(; H0=0.6766, Om=0.3111, Ol=0.6889, Ob=0.048975,
                         calc_growth=true, Tcmb=0.0, unit="Mpc")
    _COSMO[] = cosParams(H0=H0, Om=Om, Ol=Ol, Ob=Ob, calc_growth=calc_growth, Tcmb=Tcmb, unit=unit)
    return _COSMO[]
end

"""
    get_cosmology()

Return the cached cosmology; error if not initialized.
"""
function get_cosmology()
    c = _COSMO[]
    c === nothing && error("Cosmology not initialized. Call set_cosmology!() first.")
    return c
end

"""
    compute_aHf(z; cosmo=get_cosmology())

Compute aHf(z) in (km/s)/Mpc for redshift z:
    aHf = H(z)/(1+z) * dlnD/dlna(z) * c
where H(z) returned by `cosmology.jl` is in 1/(Mpc/c) so we multiply by c.
"""
function compute_aHf(z; cosmo=get_cosmology())
    return (Hz(cosmo, z) * speedoflight) / (1 + z) * dlnDdlna(cosmo, z)
end

# ============================================================================
end # module
# ============================================================================
