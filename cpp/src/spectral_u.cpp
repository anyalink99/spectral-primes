/**
 * High-precision evaluation of U(x, Lambda) from the spectral-primes preprint:
 *
 *   U(x,L) = sum_{gamma <= L} exp(-gamma^2/(2L^2)) * cos(gamma * ln x) / gamma
 *
 * Uses GNU MPFR (and GMP). Gamma values are read from CSV (same format as data/zeros.csv).
 */
#include <mpfr.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static void print_usage(const char* prog) {
  std::cerr
      << "Usage: " << prog
      << " [--csv PATH] [--lambda L] [--prec BITS] --x X [--x X2 ...]\n"
         "\n"
         "  --csv     zeros CSV with header n,gamma (default: data/zeros.csv)\n"
         "  --lambda  spectral cutoff Lambda (default: 80)\n"
         "  --prec    MPFR precision in bits (default: 256)\n"
         "  --x       argument x (string, e.g. 1e14 or 100000000000000); repeat for "
         "several points\n"
         "\n"
         "Example:\n"
         "  " << prog << " --prec 512 --lambda 80 --x 1e14 --x 1e14+1e5\n";
}

static bool read_zeros_csv(const std::string& path, std::vector<double>& gammas) {
  std::ifstream in(path);
  if (!in) {
    std::cerr << "Cannot open CSV: " << path << '\n';
    return false;
  }
  std::string line;
  if (!std::getline(in, line))
    return false;
  while (std::getline(in, line)) {
    if (line.empty())
      continue;
    auto comma = line.find(',');
    if (comma == std::string::npos)
      continue;
    std::string gstr = line.substr(comma + 1);
    try {
      double g = std::stod(gstr);
      if (g > 0.0)
        gammas.push_back(g);
    } catch (...) {
      continue;
    }
  }
  return !gammas.empty();
}

/** U(x) at arbitrary precision; x_str and lambda passed to MPFR. */
static void eval_U(
    mpfr_t result,
    const char* x_str,
    const char* lambda_str,
    const std::vector<double>& gammas,
    mpfr_prec_t prec) {
  mpfr_t x, ln_x, lambda, half_l2, gamma_m, w, phase, c, term;
  mpfr_inits2(prec, x, ln_x, lambda, half_l2, gamma_m, w, phase, c, term,
              static_cast<mpfr_ptr>(0));

  if (mpfr_set_str(x, x_str, 10, MPFR_RNDN) != 0) {
    std::cerr << "Invalid --x: " << x_str << '\n';
    mpfr_set_nan(result);
    goto cleanup;
  }
  if (mpfr_cmp_ui(x, 0) <= 0) {
    std::cerr << "x must be positive\n";
    mpfr_set_nan(result);
    goto cleanup;
  }
  if (mpfr_set_str(lambda, lambda_str, 10, MPFR_RNDN) != 0) {
    std::cerr << "Invalid lambda\n";
    mpfr_set_nan(result);
    goto cleanup;
  }
  if (mpfr_cmp_ui(lambda, 0) <= 0) {
    std::cerr << "lambda must be positive\n";
    mpfr_set_nan(result);
    goto cleanup;
  }

  mpfr_log(ln_x, x, MPFR_RNDN);
  /* half_l2 = 2 * Lambda^2  for exponent -gamma^2/(2*Lambda^2) = -gamma^2/half_l2 */
  mpfr_mul(half_l2, lambda, lambda, MPFR_RNDN);
  mpfr_mul_ui(half_l2, half_l2, 2, MPFR_RNDN);

  mpfr_set_zero(result, 1);

  for (double gd : gammas) {
    mpfr_set_d(gamma_m, gd, MPFR_RNDN);
    if (mpfr_cmp(gamma_m, lambda) > 0)
      continue;

    /* w = exp(-gamma^2 / (2*Lambda^2)) */
    mpfr_mul(w, gamma_m, gamma_m, MPFR_RNDN);
    mpfr_div(w, w, half_l2, MPFR_RNDN);
    mpfr_neg(w, w, MPFR_RNDN);
    mpfr_exp(w, w, MPFR_RNDN);

    /* phase = gamma * ln_x */
    mpfr_mul(phase, gamma_m, ln_x, MPFR_RNDN);
    mpfr_cos(c, phase, MPFR_RNDN);

    /* term = w * cos / gamma */
    mpfr_mul(term, w, c, MPFR_RNDN);
    mpfr_div(term, term, gamma_m, MPFR_RNDN);

    mpfr_add(result, result, term, MPFR_RNDN);
  }

cleanup:
  mpfr_clears(x, ln_x, lambda, half_l2, gamma_m, w, phase, c, term,
              static_cast<mpfr_ptr>(0));
}

int main(int argc, char** argv) {
  std::string csv = "data/zeros.csv";
  std::string lambda_str = "80";
  mpfr_prec_t prec = 256;
  std::vector<std::string> xs;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      print_usage(argv[0]);
      return 0;
    }
    if (a == "--csv" && i + 1 < argc) {
      csv = argv[++i];
      continue;
    }
    if (a == "--lambda" && i + 1 < argc) {
      lambda_str = argv[++i];
      continue;
    }
    if (a == "--prec" && i + 1 < argc) {
      prec = static_cast<mpfr_prec_t>(std::stoul(argv[++i]));
      if (prec < 64)
        prec = 64;
      continue;
    }
    if (a == "--x" && i + 1 < argc) {
      xs.push_back(argv[++i]);
      continue;
    }
    std::cerr << "Unknown argument: " << a << '\n';
    print_usage(argv[0]);
    return 1;
  }

  if (xs.empty()) {
    std::cerr << "Need at least one --x\n";
    print_usage(argv[0]);
    return 1;
  }

  std::vector<double> gammas;
  if (!read_zeros_csv(csv, gammas)) {
    return 1;
  }

  mpfr_set_default_prec(prec);
  mpfr_t u;
  mpfr_init2(u, prec);

  for (const std::string& xstr : xs) {
    eval_U(u, xstr.c_str(), lambda_str.c_str(), gammas, prec);
    std::cout << xstr << '\t';
    mpfr_out_str(stdout, 10, 0, u, MPFR_RNDN);
    std::cout << '\n';
  }

  mpfr_clear(u);
  mpfr_free_cache();
  return 0;
}
