# LNMock.jl

A Julia package for generating lognormal mock galaxy distributions with velocity fields from matter and galaxy power spectra.

## Overview

LNMock.jl generates 3D lognormal density fields and converts them to discrete galaxy mock catalogs with realistic velocity fields, suitable for large-scale structure and cosmological simulations.

## Features

- **Multiple FFT backends**:
  - Single-threaded FFTW
  - Multi-threaded FFTW
  - MPI-parallelized PencilFFTs for distributed computing
- **Lognormal galaxy mocks** with Poisson sampling
- **Velocity field computation** from matter density field
- **Sub-box output** for large simulations
- **Configurable cosmology** (default: Planck 2018)
- **FFTW wisdom** support for optimized FFT performance

## Installation

```julia
using Pkg

# Install dependencies
Pkg.add(url="https://github.com/djeong98/cosmology.jl")
Pkg.add(url="https://github.com/djeong98/CosmoFFTs.jl")

# Install LNMock
Pkg.add(url="https://github.com/djeong98/LNMock.jl")
```

For MPI support, also install:
```julia
Pkg.add("MPI")
Pkg.add("PencilFFTs")
```

## Quick Start

```julia
using LNMock

# Configure the mock generation
config = LNConfig(
    dims = (512, 512, 512),          # Grid dimensions
    boxsize = (1000.0, 1000.0, 1000.0),  # Box size in Mpc/h
    Ngals = 1_000_000,                # Total number of galaxies
    z = 2.7,                          # Redshift
    pkA_file = "path/to/matter_pk.dat",    # Matter power spectrum
    pkgA_file = "path/to/galaxy_pk.dat",   # Galaxy power spectrum
    outdir = "output",                # Output directory
    outhead = "mock",                 # Output file prefix
    nreal = 10,                       # Number of realizations
    computeGRF = true,                # Compute Gaussian random field
    writeGRF = false                  # Save GRF to disk
)

# Run the mock generation
run_lognormal(config)
```

## Backend Selection

Choose the FFT backend via environment variables:

```bash
# Single-threaded (default)
export LN_BACKEND=fftw_single
julia script.jl

# Multi-threaded
export LN_BACKEND=fftw_threads
export LN_THREADS=8
julia script.jl

# MPI-parallelized
export LN_BACKEND=pencil_mpi
export LN_THREADS=4  # threads per MPI rank
mpiexec -n 16 julia script.jl
```

## Power Spectrum Format

Power spectrum files should be plain text with two columns:
```
# k [h/Mpc]    P(k) [(Mpc/h)^3]
0.001          10000.0
0.002          9500.0
...
```

## Output Format

Galaxy catalogs are saved as HDF5 files containing:
- `Mock/xyzvxvyvz`: 6×N array with positions (Mpc) and velocities (km/s)
- `Mock/Ngals`: Total number of galaxies

## Configuration Options

### LNConfig Parameters

- `dims`: Grid dimensions (n₁, n₂, n₃)
- `boxsize`: Physical box size in Mpc/h
- `Ngals`: Target number of galaxies
- `z`: Redshift
- `pkA_file`: Matter power spectrum file path
- `pkgA_file`: Galaxy power spectrum file path (log-transformed)
- `outdir`: Output directory
- `outhead`: Output file prefix
- `nsub`: Sub-box divisions (default: (1,1,1))
- `nreal`: Number of realizations (default: 1)
- `seed`: Random seed (default: 3141592653)
- `computeGRF`: Generate new Gaussian random field (default: false)
- `writeGRF`: Save GRF to disk (default: false)
- `GRFfilehead`: GRF file prefix (required if computeGRF=true)
- `comm`: MPI communicator (for MPI backend only)

## Cosmology

Default cosmology is Planck 2018. Override via environment variables:

```bash
export LN_H0=0.6766
export LN_OMEGA_M=0.3111
export LN_OMEGA_L=0.6889
export LN_OMEGA_B=0.048975
```

Or in Julia:
```julia
using LNMock.Cosmology
set_cosmology!(H0=0.7, Om=0.3, Ol=0.7, Ob=0.05)
```

## FFTW Wisdom

LNMock automatically saves and loads FFTW wisdom for faster subsequent runs. Wisdom files are stored in `FFTWwisdom/` in your working directory.

To reset wisdom:
```bash
export LN_WISDOM_RESET=1
julia script.jl
```

## Example: MPI Parallel Run

```julia
using MPI
using LNMock

MPI.Init()
comm = MPI.COMM_WORLD

config = LNConfig(
    dims = (1024, 1024, 1024),
    boxsize = (2000.0, 2000.0, 2000.0),
    Ngals = 10_000_000,
    z = 2.7,
    pkA_file = "matter_pk.dat",
    pkgA_file = "galaxy_pk.dat",
    outdir = "mpi_output",
    outhead = "mock_mpi",
    comm = comm
)

run_lognormal(config)
```

Run with:
```bash
export LN_BACKEND=pencil_mpi
mpiexec -n 32 julia --project script.jl
```

## Requirements

- Julia ≥ 1.9
- CosmoFFTs.jl (https://github.com/djeong98/CosmoFFTs.jl)
- cosmology.jl (https://github.com/djeong98/cosmology.jl)
- FFTW.jl
- HDF5.jl
- Distributions.jl
- Interpolations.jl
- FLoops.jl
- (Optional) MPI.jl for parallel execution
- (Optional) PencilFFTs.jl for MPI backend

## Citation

If you use this package in your research, please cite:

```
@software{lnmock,
  author = {Donghui Jeong},
  title = {LNMock.jl: Lognormal Mock Galaxy Generator},
  year = {2025},
  url = {https://github.com/djeong98/LNMock.jl}
}
```

## License

See LICENSE file for details.

## Author

Donghui Jeong (djeong@psu.edu)
