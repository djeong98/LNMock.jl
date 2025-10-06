
to_iterable(x) = isa(x, Symbol) ? (x,) : x

"""
    load_backend(modsyms, filepath)

Load one or more external modules (e.g. FFTW, MPI, PencilFFTs) via `@eval using`,
then include the backend file and `using` its module.

- `modsyms` can be a `Symbol` or a vector of Symbols.
- `filepath` is the path to the backend `.jl` file (e.g. "Backends/FFTW_Single.jl").
"""
function load_backend(modsyms, filepath::String)

    for sym in to_iterable(modsyms)
        @eval using $sym
    end

    include(filepath)

    # Extract the module name from filename, e.g. "FFTW_Single.jl" → "FFTW_Single"
    modname = Symbol(basename(filepath)[1:end-3])
    @eval using .$(modname)
end

"""
    broadcast_scalar(val, comm)

Broadcast a single scalar value from rank 0 to all ranks.
If `comm == nothing`, just returns `val` unchanged.
"""
function broadcast_scalar(val, comm)
    if comm === nothing
        return val
    end
    buf = [val]
    MPI.Bcast!(buf, root=0, comm)
    return buf[1]
end

"""
    broadcast_array(A, comm)

Broadcast a single array from rank 0 to all ranks.
If `comm == nothing`, just returns `A` unchanged.
"""
function broadcast_array!(A, comm)
    comm === nothing && return A
    MPI.Bcast!(A, root=0, comm)
    return A
end

"""
    broadcast_object(A, comm)

Broadcast object from rank 0 to all ranks.
If `comm == nothing`, just returns `obj` unchanged.
"""
function broadcast_object(obj, comm)
    comm === nothing && return obj
    return MPI.bcast(obj, 0, comm)
end
