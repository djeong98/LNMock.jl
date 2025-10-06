# ============================================================================
# FFTW wisdom
# ============================================================================
module Wisdom
# ============================================================================

using FFTW
using Printf

export wisdom_filename, load_wisdom, save_wisdom, reset_wisdom

"""
    wisdom_filename(; nthreads, precision=:f64)

Return the local wisdom file path for given nthreads/precision.
"""
function wisdom_filename(; nthreads::Int, precision::Symbol=:f64)
    localdir = joinpath(pwd(), "FFTWwisdom")
    isdir(localdir) || mkpath(localdir)
    return joinpath(localdir, "fftw_$(precision)_nt$(nthreads).wisdom")
end

"""
    load_wisdom(file)

Try to import FFTW wisdom from `file`. No error if missing.
"""
function load_wisdom(file::AbstractString)
    if isfile(file)
        try
            FFTW.import_wisdom(file)
            @info "Imported FFTW wisdom from $file"
        catch e
            @warn "Failed to import FFTW wisdom: $e"
        end
    else
        @debug "No FFTW wisdom file found at $file"
    end
end

"""
    save_wisdom(file)

Export all known FFTW wisdom to `file` (atomic replace).
"""
function save_wisdom(file::AbstractString)
    try
        isdir(dirname(file)) || mkpath(dirname(file))
        tmp = file * ".tmp"
        FFTW.export_wisdom(tmp)
        mv(tmp, file; force=true)
        @info "Exported FFTW wisdom to $file"
    catch e
        @warn "Failed to export FFTW wisdom: $e"
    end
end

"""
    reset_wisdom(file)

Remove both in-memory wisdom and the stored file.
Triggered if ENV["LN_WISDOM_RESET"] == "1".
"""
function reset_wisdom(file::AbstractString)
    try
        FFTW.forget_wisdom!()
        isfile(file) && rm(file; force=true)
        @info "Wisdom reset: cleared memory and removed $file"
    catch e
        @warn "Failed to reset wisdom: $e"
    end
end

# ============================================================================
end # module
# ============================================================================
