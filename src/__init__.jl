function __init__()
    global_logger(ConsoleLogger(stderr, Logging.Info))

    # Mirror backend configuration from CosmoFFTs
    DEFAULT_BACKEND[] = CosmoFFTs.default_backend()
    DEFAULT_THREADS[] = CosmoFFTs.default_threads()

    # ----------------------------------------------------
    # Cosmology: default to Planck18-like unless overridden by ENV
    # For MPI backend, initialization happens at runtime in user code
    # to avoid precompilation world-age issues
    # ----------------------------------------------------
    backend = DEFAULT_BACKEND[]

    if backend != :pencil_mpi
        # Non-MPI backend: initialize cosmology normally
        try
            H0 = parse(Float64, get(ENV, "LN_H0", "0.6766"))
            Om = parse(Float64, get(ENV, "LN_OMEGA_M", "0.3111"))
            Ol = parse(Float64, get(ENV, "LN_OMEGA_L", "0.6889"))
            Ob = parse(Float64, get(ENV, "LN_OMEGA_B", "0.048975"))
            set_cosmology!(H0=H0, Om=Om, Ol=Ol, Ob=Ob, calc_growth=true, Tcmb=0.0, unit="Mpc")
            @info "Cosmology initialized" H0 Om Ol Ob
        catch e
            @error "Failed to initialize cosmology" exception=(e, catch_backtrace())
        end
    else
        # MPI backend: defer cosmology initialization to avoid world-age issues
        # User must call init_cosmology_mpi!() explicitly in their script
        @info "MPI backend detected - cosmology initialization deferred. Call LNMock.init_cosmology_mpi!() in your script."
    end

    return nothing
end
