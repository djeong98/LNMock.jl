
"""
    broadcast_scalar(val, comm)

Broadcast a single scalar value from rank 0 to all ranks.
If `comm == nothing`, just returns `val` unchanged.
"""
function broadcast_scalar(val, comm)
    if comm === nothing
        return val
    end
    MPI = _get_MPI()
    return MPI.bcast(val, 0, comm)
end

"""
    broadcast_array(A, comm)

Broadcast a single array from rank 0 to all ranks.
If `comm == nothing`, just returns `A` unchanged.
"""
function broadcast_array!(A, comm)
    comm === nothing && return A
    MPI = _get_MPI()
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
    MPI = _get_MPI()
    return MPI.bcast(obj, 0, comm)
end
