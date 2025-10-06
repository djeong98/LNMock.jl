
function __init__()
    # ----------------------------------------------------
    # 1. Logging setup
    # ----------------------------------------------------
    global_logger(ConsoleLogger(stderr, Logging.Info))
    
    # ----------------------------------------------------
    # 2. Override backend and threads from ENV if provided
    # ----------------------------------------------------
    if haskey(ENV, "LN_BACKEND")
        DEFAULT_BACKEND[] = Symbol(ENV["LN_BACKEND"])
    end
    if haskey(ENV, "LN_THREADS")
        DEFAULT_THREADS[] = parse(Int, ENV["LN_THREADS"])
    end

    # ----------------------------------------------------
    # 3. Wisdom file: always local to the working directory
    # ----------------------------------------------------
    WISDOM_FILE[] = wisdom_filename(nthreads=DEFAULT_THREADS[])
    # Reset wisdom if requested
    if get(ENV, "LN_WISDOM_RESET", "0") == "1"
        reset_wisdom(WISDOM_FILE[])
    end
    
    # ----------------------------------------------------
    # 4. Initialize FFTW threads and wisdom (backend specific)
    # ----------------------------------------------------
    if DEFAULT_BACKEND[] == :fftw_single
        @info "Initializing FFTW single-thread backend"
        load_backend(:FFTW, "Backends/FFTW_Single.jl")
    
        load_wisdom(WISDOM_FILE[])
        # Export at exit so that new plans are stored
        atexit(() -> save_wisdom(WISDOM_FILE[]))
    elseif DEFAULT_BACKEND[] == :fftw_threads
        @info "Initializing FFTW threaded backend"
        load_backend(:FFTW, "Backends/FFTW_Threads.jl")

        FFTW.set_num_threads(DEFAULT_THREADS[])
        @info "FFTW using $(DEFAULT_THREADS[]) threads"
        load_wisdom(WISDOM_FILE[])
        atexit(() -> save_wisdom(WISDOM_FILE[]))
    elseif DEFAULT_BACKEND[] == :pencil_mpi
        @info "Initializing PencilFFTs (MPI) backend"
        try
            # Load backend code first
            load_backend((:MPI,:PencilFFTs), "Backends/PencilFFTs_MPI.jl")
            
            # Set per-rank FFTW threads for local transforms
            FFTW.set_num_threads(DEFAULT_THREADS[])
            @info "PencilFFTs (MPI) with FFTW $(DEFAULT_THREADS[]) threads per rank"
            # Import wisdom BEFORE any PencilFFTPlan is created
            load_wisdom(WISDOM_FILE[])
            
            if !MPI.Initialized()
                MPI.Init()
                atexit(() -> (MPI.Initialized() && !MPI.Finalized()) && MPI.Finalize())
            end
            
            # Export wisdom on exit (captures any new local FFTW plans created by PencilFFTs)
            atexit(() -> save_wisdom(WISDOM_FILE[]))
            
            @info "MPI backend initialized with $(MPI.Comm_size(MPI.COMM_WORLD)) ranks"
        catch e
            @error "MPI backend requested but MPI not available: $e"
        end
    end
    
    # ----------------------------------------------------
    # Cosmology: default to Planck18-like unless overridden by ENV
    # ----------------------------------------------------
    try
        H0 = parse(Float64, get(ENV, "LN_H0", "0.6766"))
        Om = parse(Float64, get(ENV, "LN_OMEGA_M", "0.3111"))
        Ol = parse(Float64, get(ENV, "LN_OMEGA_L", "0.6889"))
        Ob = parse(Float64, get(ENV, "LN_OMEGA_B", "0.048975"))
        set_cosmology!(H0=H0, Om=Om, Ol=Ol, Ob=Ob, calc_growth=true, Tcmb=0.0, unit="Mpc")
        @info "Cosmology initialized (H0=$(H0), Om=$(Om), Ol=$(Ol), Ob=$(Ob))"
    catch e
        @error "Failed to initialize cosmology: $e"
    end

    return nothing
end

