#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>


inline double next_V(double V_current, double V_expensive, double V_cheap,
                     double dS, double dt, double sigma, double S, double r) {
  double dS2 = dS * dS;
  double S2 = S * S;
  return V_current +
         dt * (0.5 * sigma * sigma * S2 *
                   (V_expensive - 2 * V_current + V_cheap) / dS2 +
               r * S * (V_expensive - V_cheap) / (2 * dS) - r * V_current);
}

Eigen::VectorXd fdm_solve(double K, double r, double sigma, double q, double T, int t_steps,
                          double S_max, int S_steps, bool is_call) {
    double dt = T / t_steps;
    double dS = S_max / S_steps;

    Eigen::VectorXd V_old = Eigen::VectorXd::Zero(S_steps + 1);
    Eigen::VectorXd V_new = Eigen::VectorXd::Zero(S_steps + 1);

    for (int i = 0; i <= S_steps; ++i) {
        double S = i * dS;
        V_old[i] = is_call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
    }

    Eigen::VectorXd A(S_steps + 1), B(S_steps + 1), C(S_steps + 1);
    #pragma omp parallel for
    for (int i = 1; i < S_steps; ++i) {
        double S = i * dS;
        double S2 = S * S;
        double alpha = dt / (dS * dS);
        double beta = dt / (2 * dS);
        double a_val = 0.5 * sigma * sigma * S2 * alpha;
        double b_val =  (r - q) * S * beta;
        A[i] = a_val - b_val;
        B[i] = 1.0 - 2.0 * a_val - r * dt;
        C[i] = a_val + b_val;
    }

    for (int n = 0; n < t_steps; ++n) {
        if (is_call) {
            V_new[0] = 0.0;
            V_new[S_steps] = S_max - K;
        } else {
            V_new[0] =K;
            V_new[S_steps] = 0.0;
        }

        for (int i = 1; i < S_steps; ++i) {
            double S_val = i * dS;
            V_new[i] = A[i] * V_old[i - 1] + B[i] * V_old[i] + C[i] * V_old[i + 1];
            double intrinsic = is_call ? std::max(S_val- K, 0.0) : std::max(K - S_val, 0.0);
            
            V_new[i] = std::max(intrinsic, V_new[i]);
        }

        std::swap(V_old, V_new);
    }        ;

    return V_old;
}

double compute_delta(const Eigen::VectorXd& V, int i, double S_max, int S_steps) {
    double dS = S_max / S_steps;
    return (V[i + 1] - V[i - 1]) / (2.0 * dS);
} 

double compute_gamma(const Eigen::VectorXd& V, int i, double S_max, int S_steps) {
    double dS = S_max / S_steps;
    return (V[i + 1] - 2.0 * V[i] + V[i - 1]) / (dS * dS);
}

double compute_vega(int i, double K, double r, double sigma, double q, double T, int t_steps,
                    double S_max, int S_steps, bool is_call, double d_sigma = 0.01) {
    Eigen::VectorXd V_plus = fdm_solve(K, r, sigma + d_sigma, q, T, t_steps, S_max, S_steps, is_call);
    Eigen::VectorXd V_minus = fdm_solve(K, r, sigma - d_sigma, q, T, t_steps, S_max, S_steps, is_call);
    
    double dS = S_max / S_steps;
    return (V_plus[i] - V_minus[i]) / (2.0 * d_sigma);
}

double sigma_impl_solver(double K, double r, double q, double V_market, double T,
                         int t_steps, double S_max, int S_steps, bool is_call,
                         int max_iter, double tolerance = 1e-6) {
  double a = 0.01;
  double b = 2.0;
  int index_at = S_steps / 2;

  double fa =
      fdm_solve(K, r, a, q, T, t_steps, S_max, S_steps, is_call)[index_at] - V_market;
  double fb =
      fdm_solve(K, r, b, q, T, t_steps, S_max, S_steps, is_call)[index_at] - V_market;
  while (fa * fb > 0) {
    if (std::abs(fa) < std::abs(fb)) {
      a *= 0.5;
      fa = fdm_solve(K, r, a, q, T, t_steps, S_max, S_steps, is_call)[index_at] -
           V_market;
    } else {
      b *= 2;
      fb = fdm_solve(K, r, b, q, T, t_steps, S_max, S_steps, is_call)[index_at] -
           V_market;
    }
    if (b > 120.0 || a < 1e-10) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }
  double c = a;
  double fc = fa;
  double d = 0.0;
  double candidate, f_candidate;
  bool bisect;
  bool mflag = true;
  double avg;

  while (max_iter > 0) {
    max_iter--;

    if (std::abs(fb) < tolerance || std::abs(a - b) < tolerance) {
      return b;
    }

    if (fa != fb && fb != fc) {
      candidate = (a * fb * fc) / ((fa - fb) * (fa - fc)) +
                  (b * fa * fc) / ((fb - fa) * (fb - fc)) +
                  (c * fa * fb) / ((fc - fa) * (fc - fb));
    } else {
      candidate = b - fb * (b - a) / (fb - fa);
    }

    bisect = false;
    avg = (3 * a + b) / 4;

    if (candidate < std::min(avg, b) || candidate > std::max(avg, b)) {
      bisect = true;
    } else if (mflag && d != 0.0 &&
               std::abs(candidate - b) >= std::abs(b - c) / 2) {
      bisect = true;
    } else if (!mflag && d != 0.0 &&
               std::abs(candidate - b) >= std::abs(c - d) / 2) {
      bisect = true;
    } else if (mflag && d != 0.0 && std::abs(b - c) < tolerance) {
      bisect = true;
    } else if (!mflag && d != 0.0 && std::abs(c - d) < tolerance) {
      bisect = true;
    }

    if (bisect) {
      candidate = (a + b) / 2;
      mflag = true;
    } else {
      mflag = false;
    }

    f_candidate =
        fdm_solve(K, r, candidate, q, T, t_steps, S_max, S_steps, is_call)[index_at] -
        V_market;

    d = c;
    c = b;
    fc = fb;

    if (fa * f_candidate < 0) {
      b = candidate;
      fb = f_candidate;
    } else {
      a = candidate;
      fa = f_candidate;
    }

    if (std::abs(fa) < std::abs(fb)) {
      std::swap(a, b);
      std::swap(fa, fb);
    }
  }
  return b;
}

Eigen::VectorXd sigma_impl_solver_batch(
    const Eigen::VectorXd& Ks,
    double r,
    double q,
    const Eigen::VectorXd& V_markets,
    const Eigen::VectorXd& Ts,
    int t_steps,
    double S_max,
    int S_steps,
    const Eigen::Array<bool, Eigen::Dynamic, 1>& is_calls,
    int max_iter,
    double tolerance = 1e-6) {

  size_t n = Ks.size();
  Eigen::VectorXd results(n);

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(n); ++i) {
    results[i] = sigma_impl_solver(
        Ks[i], r, q, V_markets[i], Ts[i], t_steps, S_max, S_steps,
        is_calls[i], max_iter, tolerance);
  }

  return results;
}

PYBIND11_MODULE(numerical_methods, m) {
  m.doc() =
      "an explicit euler solver for the Black-Scholes equation"; 

  m.def("fdm_solve", &fdm_solve,
        "the function responsible for providing the solution grid",
        pybind11::arg("K"), pybind11::arg("r"), pybind11::arg("sigma"), pybind11::arg("q"),
        pybind11::arg("T"), pybind11::arg("t_steps"), pybind11::arg("S_max"),
        pybind11::arg("S_steps"), pybind11::arg("is_call"));
  m.def("compute_delta", &compute_delta, "function that finds the delta of an option", pybind11::arg("V"),  pybind11::arg("i"), pybind11::arg("S_max"), pybind11::arg("S_steps"));
  m.def("compute_gamma", &compute_gamma, "function that finds the gamma of an option", pybind11::arg("V"),  pybind11::arg("i"), pybind11::arg("S_max"), pybind11::arg("S_steps"));
  m.def("compute_vega", &compute_vega, "function that finds the vega of an option", pybind11::arg("i"),
        pybind11::arg("K"), pybind11::arg("r"), pybind11::arg("sigma"), pybind11::arg("q"),
        pybind11::arg("T"), pybind11::arg("t_steps"), pybind11::arg("S_max"),
        pybind11::arg("S_steps"), pybind11::arg("is_call"), pybind11::arg("d_sigma"));

  m.def("sigma_impl_solver", &sigma_impl_solver, "implied volatility finder",
        pybind11::arg("K"), pybind11::arg("r"), pybind11::arg("q"), pybind11::arg("V_market"),
        pybind11::arg("T"), pybind11::arg("t_steps"), pybind11::arg("S_max"),
        pybind11::arg("S_steps"), pybind11::arg("is_call"),
        pybind11::arg("max_iter"), pybind11::arg("tolerance"));
  m.def("sigma_impl_solver_batch", &sigma_impl_solver_batch, pybind11::arg("Ks"), pybind11::arg("r"), pybind11::arg("q"),
        pybind11::arg("V_markets"), pybind11::arg("Ts"), pybind11::arg("t_steps"), pybind11::arg("S_max"), pybind11::arg("S_steps"),
        pybind11::arg("is_calls"), pybind11::arg("max_iter"), pybind11::arg("tolerance"));

  };


