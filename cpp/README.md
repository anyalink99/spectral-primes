# C++ tool: `spectral_u` (MPFR)

[`spectral_u`](src/spectral_u.cpp) evaluates the regularized operator

$$U(x,\Lambda) = \sum_{\gamma_n \le \Lambda} \exp\!\left(-\frac{\gamma_n^2}{2\Lambda^2}\right) \frac{\cos(\gamma_n \ln x)}{\gamma_n}$$

using **GNU MPFR** (multiple-precision floats, correct rounding). **GMP** is a dependency of MPFR.

Use this when `x` is large and `ln x` amplifies phase sensitivity, or when you want reproducible high precision beyond IEEE-754 `double`. Zeros `γ_n` are still read from the same CSV as the Python code (limited by the precision stored in the file unless you add more digits).

## Build

From the **repository root** (so default `--csv data/zeros.csv` resolves):

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build
```

The binary is `cpp/build/spectral_u` (or `spectral_u.exe` on Windows).

### Debian / Ubuntu

```bash
sudo apt install libmpfr-dev cmake build-essential
```

### Fedora

```bash
sudo dnf install mpfr-devel cmake gcc-c++
```

### macOS (Homebrew)

```bash
brew install mpfr cmake
```

### Windows

- **MSYS2 (MinGW):** `pacman -S mingw-w64-x86_64-mpfr mingw-w64-x86_64-cmake mingw-w64-x86_64-gcc`, then configure from MSYS2 shell with paths as usual.
- **vcpkg:** `vcpkg install mpfr` then:

  ```bash
  cmake -S cpp -B cpp/build -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake
  cmake --build cpp/build
  ```

  (Adjust target triplet / path as needed.)

## Usage

Run from repo root, or pass an explicit CSV path:

```bash
./cpp/build/spectral_u --prec 512 --lambda 80 --x 1e14 --x 1e15
./cpp/build/spectral_u --csv data/zeros.csv --lambda 100 --prec 256 --x 1000000
```

- `--prec` — MPFR precision in bits (default `256`).
- `--lambda` — cutoff `Λ` (string, so you can use integers or decimals).
- `--x` — repeat for each sample point (string; MPFR parses it, e.g. `1e14`).

Output: `x` and `U(x)` in decimal, tab-separated.

## Relation to Python

The Python `spectral_primes.operator.U_batch` uses `numpy` `float64`. This binary is for **cross-checks** and **high-precision** sweeps, not a replacement for the full experiment pipeline (primes, `U_sparse`, permutation tests remain in Python).
