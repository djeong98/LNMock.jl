function __init__()
    global_logger(ConsoleLogger(stderr, Logging.Info))

    # Mirror backend configuration from CosmoFFTs
    DEFAULT_BACKEND[] = CosmoFFTs.default_backend()
    DEFAULT_THREADS[] = CosmoFFTs.default_threads()

    # ----------------------------------------------------
    # Cosmology: default to Planck18-like unless overridden by ENV
    # Initialize only on rank 0 for MPI, then broadcast the cosmology object
    # ----------------------------------------------------
    try
        backend = DEFAULT_BACKEND[]

        if backend == :pencil_mpi
            # MPI backend: compute cosmology only on rank 0, broadcast to others
            # MPI is already loaded by CosmoFFTs when backend is :pencil_mpi
            comm = Main.MPI.COMM_WORLD
            rank = Main.MPI.Comm_rank(comm)

            local cosmo_obj
            if rank == 0
                H0 = parse(Float64, get(ENV, "LN_H0", "0.6766"))
                Om = parse(Float64, get(ENV, "LN_OMEGA_M", "0.3111"))
                Ol = parse(Float64, get(ENV, "LN_OMEGA_L", "0.6889"))
                Ob = parse(Float64, get(ENV, "LN_OMEGA_B", "0.048975"))
                cosmo_obj = set_cosmology!(H0=H0, Om=Om, Ol=Ol, Ob=Ob, calc_growth=true, Tcmb=0.0, unit="Mpc")
                @info "Cosmology initialized on rank 0" H0 Om Ol Ob
            else
                cosmo_obj = nothing
            end

            # Broadcast the entire cosmology object from rank 0 to all ranks
            cosmo_obj = Main.MPI.bcast(cosmo_obj, 0, comm)

            # Set the broadcasted cosmology on non-root ranks
            if rank != 0
                Cosmology._COSMO[] = cosmo_obj
            end
        else
            # Non-MPI backend: normal initialization
            H0 = parse(Float64, get(ENV, "LN_H0", "0.6766"))
            Om = parse(Float64, get(ENV, "LN_OMEGA_M", "0.3111"))
            Ol = parse(Float64, get(ENV, "LN_OMEGA_L", "0.6889"))
            Ob = parse(Float64, get(ENV, "LN_OMEGA_B", "0.048975"))
            set_cosmology!(H0=H0, Om=Om, Ol=Ol, Ob=Ob, calc_growth=true, Tcmb=0.0, unit="Mpc")
            @info "Cosmology initialized" H0 Om Ol Ob
        end
    catch e
        @error "Failed to initialize cosmology" exception=(e, catch_backtrace())
    end

    return nothing
end
