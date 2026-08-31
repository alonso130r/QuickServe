#include "backend_cache.hpp"
#include "experiment.hpp"
#include "solvers.hpp"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/utsname.h>
#include <unistd.h>

namespace {
std::map<std::string, std::string> args(int argc, char **argv, int start = 2) {
  std::map<std::string, std::string> out;
  for (int i = start; i < argc; ++i) {
    std::string k = argv[i];
    if (k.rfind("--", 0) != 0 || i + 1 >= argc)
      throw std::invalid_argument("expected --key value");
    out[k.substr(2)] = argv[++i];
  }
  return out;
}
std::string need(const std::map<std::string, std::string> &a,
                 const std::string &k) {
  auto i = a.find(k);
  if (i == a.end())
    throw std::invalid_argument("missing --" + k);
  return i->second;
}
std::size_t number(const std::map<std::string, std::string> &a,
                   const std::string &k, std::size_t d) {
  auto i = a.find(k);
  return i == a.end() ? d : std::stoull(i->second);
}
double real(const std::map<std::string, std::string> &a, const std::string &k,
            double fallback) {
  auto i = a.find(k);
  return i == a.end() ? fallback : std::stod(i->second);
}
std::uint64_t wall_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}
std::uint64_t monotonic_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
std::string file_fingerprint(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot hash model");
  std::uint64_t h = 1469598103934665603ULL;
  char buffer[1 << 20];
  while (in) {
    in.read(buffer, sizeof buffer);
    for (std::streamsize k = 0; k < in.gcount(); ++k) {
      h ^= static_cast<unsigned char>(buffer[k]);
      h *= 1099511628211ULL;
    }
  }
  std::ostringstream out;
  out << std::hex << h;
  return out.str();
}
std::string platform_identity() {
  struct utsname u{};
  uname(&u);
  return std::string(u.machine) + "|" + u.release + "|" + __VERSION__;
}
std::vector<unsigned> number_list(const std::map<std::string, std::string> &a,
                                  const std::string &list_key,
                                  const std::string &single_key,
                                  unsigned fallback) {
  std::string value = a.count(list_key)     ? a.at(list_key)
                      : a.count(single_key) ? a.at(single_key)
                                            : std::to_string(fallback);
  std::vector<unsigned> out;
  std::stringstream ss(value);
  std::string part;
  while (std::getline(ss, part, ','))
    out.push_back(std::stoul(part));
  return out;
}
struct Reservation {
  std::filesystem::path path;
  Reservation(const std::filesystem::path &p) : path(p) {
    if (std::filesystem::create_directory(path)) {
      std::ofstream(path / "owner") << getpid() << '\n';
      return;
    }
    std::ifstream in(path / "owner");
    int owner = 0;
    in >> owner;
    if (owner > 0 && kill(owner, 0) == -1 && errno == ESRCH) {
      std::filesystem::remove_all(path);
      if (std::filesystem::create_directory(path)) {
        std::ofstream(path / "owner") << getpid() << '\n';
        return;
      }
    }
    throw std::runtime_error("another live collector owns this cache key");
  }
  ~Reservation() {
    std::error_code e;
    std::filesystem::remove_all(path, e);
  }
};

void synthetic(const std::map<std::string, std::string> &a) {
  std::ofstream out(need(a, "output"));
  if (!out)
    throw std::runtime_error("cannot open output");
  out << "instance,comparison,solver,phase,repetition,success,p,d,objective,"
         "setup_or_update_ns,solve_ns,total_ns,iterations,primal_residual,dual_"
         "residual,status\n";
  const auto instances = sloexp::generate_instances(
      number(a, "seed", 1), number(a, "instances", 729));
  const auto reps = number(a, "repetitions", 1000);
  for (const auto &i : instances) {
    auto emit = [&](const char *c, const char *s, const sloexp::SolveResult &r,
                    std::size_t k) {
      out << i.id << ',' << c << ',' << s << ",cold," << k << ',' << r.success
          << ',' << r.p << ',' << r.d << ',' << r.objective << ','
          << r.cold_setup_samples.front() << ',' << r.cold_solve_samples.front()
          << ',' << r.cold_setup_samples.front() + r.cold_solve_samples.front()
          << ',' << r.iterations << ',' << r.primal_residual << ','
          << r.dual_residual << ',' << r.status << '\n';
      out << i.id << ',' << c << ',' << s << ",warm," << k << ',' << r.success
          << ',' << r.p << ',' << r.d << ',' << r.objective << ','
          << r.warm_update_samples.front() << ','
          << r.warm_solve_samples.front() << ','
          << r.warm_update_samples.front() + r.warm_solve_samples.front() << ','
          << r.iterations << ',' << r.primal_residual << ',' << r.dual_residual
          << ',' << r.status << '\n';
    };
    for (std::size_t k = 0; k < reps; ++k) {
      sloexp::SolveResult pq, cq, pn, cs;
      const bool prox_first = (i.id + k) % 2 == 0;
      if (prox_first) {
        pq = sloexp::solve_proxqp(i, 1);
        cq = sloexp::solve_clarabel_qp(i, 1);
        pn = sloexp::solve_proxqp(i, 1);
        cs = sloexp::solve_clarabel_socp(i, 1);
      } else {
        cq = sloexp::solve_clarabel_qp(i, 1);
        pq = sloexp::solve_proxqp(i, 1);
        cs = sloexp::solve_clarabel_socp(i, 1);
        pn = sloexp::solve_proxqp(i, 1);
      }
      emit("shared_qp", "proxqp", pq, k);
      emit("shared_qp", "clarabel", cq, k);
      emit("native", "proxqp_qp", pn, k);
      emit("native", "clarabel_socp", cs, k);
    }
  }
}

void collect(const std::map<std::string, std::string> &a) {
  const auto ps = number_list(a, "prefill-list", "prefill", 0),
             ds = number_list(a, "decode-list", "decode", 0),
             contexts = number_list(a, "context-list", "context", 512);
  const auto reps = number(a, "repetitions", 30),
             warm = number(a, "warmups", 3), threads = number(a, "threads", 1),
             seed = number(a, "seed", 1);
  const std::string run =
      a.count("run-id") ? a.at("run-id") : std::to_string(wall_ns());
  const std::string environment = need(a, "environment-id"),
                    input_seed =
                        a.count("input-seed") ? a.at("input-seed") : "0",
                    model_hash = file_fingerprint(need(a, "model"));
  const auto cache_path = std::filesystem::path(need(a, "cache"));
  if (!cache_path.parent_path().empty())
    std::filesystem::create_directories(cache_path.parent_path());
  struct Spec {
    unsigned p, d, ctx;
  };
  std::vector<Spec> specs;
  for (auto ctx : contexts)
    for (auto p : ps)
      for (auto d : ds)
        if (p + d)
          specs.push_back({p, d, ctx});
  if (specs.empty())
    throw std::invalid_argument("collection grid is empty");
  std::mt19937 rng(seed);
  std::size_t added = 0;
  for (std::size_t round = 0; round < reps; ++round) {
    std::shuffle(specs.begin(), specs.end(), rng);
    for (std::size_t order = 0; order < specs.size(); ++order) {
      const auto [p, d, ctx] = specs[order];
      const std::string key =
          "schema=3|input_schema=bos-repeat-v1|qs=" SLO_QS_REV
          "|llama=" SLO_LLAMA_REV "|build=" SLO_BUILD_TYPE "|model_fnv64=" +
          model_hash + "|platform=" + platform_identity() +
          "|environment=" + environment +
          "|backend=llama-direct|token_seed=" + input_seed +
          "|kv=bos-repeat-ctx-" + std::to_string(ctx) +
          "|threads=" + std::to_string(threads) + "|p=" + std::to_string(p) +
          "|d=" + std::to_string(d) + "|ctx=" + std::to_string(ctx) +
          "|seq=" + std::to_string(d + (p ? 1 : 0));
      Reservation reservation(
          std::filesystem::path(need(a, "cache") + ".collect." +
                                std::to_string(std::hash<std::string>{}(key))));
      sloexp::MeasurementCache cache(need(a, "cache"));
      std::size_t existing = 0;
      for (const auto &m : cache.load())
        if (m.key == key)
          ++existing;
      if (existing > round)
        continue;
      const std::size_t current = round * specs.size() + order + 1;
      const std::size_t total = reps * specs.size();
      std::cout << "[collect " << current << '/' << total << "] p=" << p
                << " d=" << d << " context=" << ctx << std::endl;
      const auto values = sloexp::measure_backend(need(a, "model"), p, d, ctx,
                                                  threads, warm, 1);
      const auto id =
          run + "-" + std::to_string(round) + "-" + std::to_string(order);
      added += cache.append({key, id, run, wall_ns(), p, d, ctx, values.front(),
                             monotonic_ns(), static_cast<std::uint32_t>(order),
                             d + (p ? 1 : 0)});
    }
  }
  std::cout << "cached " << added << " new observations across " << specs.size()
            << " compositions\n";
}

void evaluate(const std::map<std::string, std::string> &a) {
  const auto rows = sloexp::MeasurementCache(need(a, "cache")).load();
  if (rows.size() < 30)
    throw std::runtime_error(
        "evaluate requires at least 30 cached observations");
  std::map<std::string, std::vector<const sloexp::Measurement *>> by_key;
  for (const auto &r : rows)
    by_key[r.key].push_back(&r);
  std::vector<const sloexp::Measurement *> train, cal, held;
  std::map<std::string, std::vector<const sloexp::Measurement *>> held_by_key;
  std::map<std::string, std::vector<const sloexp::Measurement *>> cal_by_key;
  for (auto &[key, observations] : by_key) {
    std::sort(observations.begin(), observations.end(), [](auto *a, auto *b) {
      return std::tie(a->run_id, a->observation_id) <
             std::tie(b->run_id, b->observation_id);
    });
    if (observations.size() < 10)
      throw std::runtime_error(
          "each allocation cell needs at least 10 repetitions");
    for (std::size_t k = 0; k < observations.size(); ++k) {
      const auto bucket = k % 10;
      if (bucket < 6)
        train.push_back(observations[k]);
      else if (bucket < 8) {
        cal.push_back(observations[k]);
        cal_by_key[key].push_back(observations[k]);
      } else {
        held.push_back(observations[k]);
        held_by_key[key].push_back(observations[k]);
      }
    }
  }
  if (train.size() < 3 || cal.empty() || held.empty())
    throw std::runtime_error(
        "cache does not cover a 60/20/20 repetition split");
  Eigen::MatrixXd X(train.size(), 4);
  Eigen::VectorXd y(train.size());
  for (std::size_t k = 0; k < train.size(); ++k) {
    X.row(k) << 1.0, train[k]->prefill_tokens, train[k]->decode_items,
        train[k]->context_tokens;
    y(k) = train[k]->duration_ns;
  }
  const Eigen::Vector4d beta = X.colPivHouseholderQr().solve(y);
  const Eigen::VectorXd training_error = y - X * beta;
  const double residual_variance =
      training_error.squaredNorm() /
      std::max<Eigen::Index>(1, X.rows() - X.cols());
  const Eigen::Matrix4d xtx = X.transpose() * X;
  Eigen::Matrix4d inverse_xtx =
      xtx.colPivHouseholderQr().solve(Eigen::Matrix4d::Identity());
  Eigen::Matrix4d coefficient_covariance = residual_variance * inverse_xtx;
  coefficient_covariance =
      0.5 * (coefficient_covariance + coefficient_covariance.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> covariance_eigen(
      coefficient_covariance);
  if (covariance_eigen.info() != Eigen::Success)
    throw std::runtime_error("coefficient covariance decomposition failed");
  const Eigen::Vector4d covariance_eigenvalues =
      covariance_eigen.eigenvalues().cwiseMax(0.0).cwiseSqrt();
  const Eigen::Matrix4d covariance_factor =
      covariance_eigen.eigenvectors() * covariance_eigenvalues.asDiagonal();
  Eigen::MatrixXd RX(train.size(), 4);
  Eigen::VectorXd ry(train.size());
  for (std::size_t k = 0; k < train.size(); ++k) {
    RX.row(k) << 1.0, train[k]->prefill_tokens, train[k]->decode_items,
        train[k]->context_tokens;
    ry(k) = std::abs(train[k]->duration_ns -
                     (beta(0) + beta(1) * train[k]->prefill_tokens +
                      beta(2) * train[k]->decode_items +
                      beta(3) * train[k]->context_tokens));
  }
  Eigen::Vector4d shape = RX.colPivHouseholderQr().solve(ry).cwiseMax(1.0);
  const double runtime_scale =
      std::max(1.0, std::sqrt(ry.squaredNorm() / ry.size()));
  auto predicted_mean = [&](const sloexp::Measurement &r) {
    return beta(0) + beta(1) * r.prefill_tokens + beta(2) * r.decode_items +
           beta(3) * r.context_tokens;
  };
  auto predicted_shape = [&](const sloexp::Measurement &r) {
    return std::sqrt(std::pow(shape(0), 2) +
                     std::pow(shape(1) * r.prefill_tokens, 2) +
                     std::pow(shape(2) * r.decode_items, 2) +
                     std::pow(shape(3) * r.context_tokens, 2));
  };
  auto predicted_covariance_shape = [&](const sloexp::Measurement &r) {
    const Eigen::Vector4d phi(1.0, r.prefill_tokens, r.decode_items,
                              r.context_tokens);
    return std::sqrt((covariance_factor.transpose() * phi).squaredNorm() +
                     residual_variance);
  };
  std::vector<double> residuals;
  for (auto *r : cal)
    residuals.push_back(std::max(0.0, r->duration_ns - predicted_mean(*r)));
  const double upper_bound_quantile = real(a, "upper-bound-quantile", 0.98);
  if (!(upper_bound_quantile > 0.5 && upper_bound_quantile < 1.0))
    throw std::invalid_argument(
        "--upper-bound-quantile must be between 0.5 and 1");
  const double delta = sloexp::nearest_rank(residuals, upper_bound_quantile);
  std::vector<double> ratios;
  for (auto *r : cal) {
    const double scale = predicted_shape(*r);
    ratios.push_back(std::max(0.0, r->duration_ns - predicted_mean(*r)) /
                     scale);
  }
  const double kappa = sloexp::nearest_rank(ratios, upper_bound_quantile);
  std::vector<double> covariance_ratios;
  for (auto *r : cal) {
    const double scale = predicted_covariance_shape(*r);
    covariance_ratios.push_back(
        std::max(0.0, r->duration_ns - predicted_mean(*r)) / scale);
  }
  const double covariance_kappa =
      sloexp::nearest_rank(covariance_ratios, upper_bound_quantile);
  auto margin = [&](const sloexp::Measurement &r, bool conic) {
    return conic ? kappa * predicted_shape(r) : delta;
  };
  auto coverage = [&](const std::vector<const sloexp::Measurement *> &set,
                      bool conic) {
    std::size_t ok = 0;
    for (auto *r : set) {
      const double mean = predicted_mean(*r);
      if (r->duration_ns <= mean + margin(*r, conic))
        ++ok;
    }
    return double(ok) / set.size();
  };
  auto covariance_coverage =
      [&](const std::vector<const sloexp::Measurement *> &set) {
        std::size_t ok = 0;
        for (auto *r : set)
          if (r->duration_ns <= predicted_mean(*r) +
                                    covariance_kappa *
                                        predicted_covariance_shape(*r))
            ++ok;
        return double(ok) / set.size();
      };
  auto leave_cell_out_coverage = [&](bool conic) {
    std::size_t correct = 0, total = 0;
    for (const auto &[held_key, observations] : held_by_key) {
      std::vector<const sloexp::Measurement *> fold_train;
      for (auto *r : train)
        if (r->key != held_key)
          fold_train.push_back(r);
      Eigen::MatrixXd fold_x(fold_train.size(), 4);
      Eigen::VectorXd fold_y(fold_train.size());
      for (std::size_t k = 0; k < fold_train.size(); ++k) {
        fold_x.row(k) << 1.0, fold_train[k]->prefill_tokens,
            fold_train[k]->decode_items, fold_train[k]->context_tokens;
        fold_y(k) = fold_train[k]->duration_ns;
      }
      const Eigen::Vector4d fold_beta =
          fold_x.colPivHouseholderQr().solve(fold_y);
      for (auto *r : observations) {
        const double fold_mean =
            fold_beta(0) + fold_beta(1) * r->prefill_tokens +
            fold_beta(2) * r->decode_items + fold_beta(3) * r->context_tokens;
        if (r->duration_ns <= fold_mean + margin(*r, conic))
          ++correct;
        ++total;
      }
    }
    return total ? double(correct) / total : 0.0;
  };
  auto covariance_leave_cell_out_coverage = [&] {
    std::size_t correct = 0, total = 0;
    for (const auto &[held_key, observations] : held_by_key) {
      std::vector<const sloexp::Measurement *> fold_train;
      for (auto *r : train)
        if (r->key != held_key)
          fold_train.push_back(r);
      Eigen::MatrixXd fold_x(fold_train.size(), 4);
      Eigen::VectorXd fold_y(fold_train.size());
      for (std::size_t k = 0; k < fold_train.size(); ++k) {
        fold_x.row(k) << 1.0, fold_train[k]->prefill_tokens,
            fold_train[k]->decode_items, fold_train[k]->context_tokens;
        fold_y(k) = fold_train[k]->duration_ns;
      }
      const Eigen::Vector4d fold_beta =
          fold_x.colPivHouseholderQr().solve(fold_y);
      const Eigen::VectorXd fold_error = fold_y - fold_x * fold_beta;
      const double fold_residual_variance =
          fold_error.squaredNorm() /
          std::max<Eigen::Index>(1, fold_x.rows() - fold_x.cols());
      const Eigen::Matrix4d fold_xtx = fold_x.transpose() * fold_x;
      Eigen::Matrix4d fold_inverse = fold_xtx.colPivHouseholderQr().solve(
          Eigen::Matrix4d::Identity());
      Eigen::Matrix4d fold_covariance = fold_residual_variance * fold_inverse;
      fold_covariance = 0.5 * (fold_covariance + fold_covariance.transpose());
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> fold_eigen(
          fold_covariance);
      if (fold_eigen.info() != Eigen::Success)
        throw std::runtime_error(
            "leave-cell-out covariance decomposition failed");
      const Eigen::Matrix4d fold_factor =
          fold_eigen.eigenvectors() *
          fold_eigen.eigenvalues()
              .cwiseMax(0.0)
              .cwiseSqrt()
              .asDiagonal();
      for (auto *r : observations) {
        const Eigen::Vector4d phi(1.0, r->prefill_tokens, r->decode_items,
                                  r->context_tokens);
        const double fold_mean = fold_beta.dot(phi);
        const double fold_scale =
            std::sqrt((fold_factor.transpose() * phi).squaredNorm() +
                      fold_residual_variance);
        if (r->duration_ns <= fold_mean + covariance_kappa * fold_scale)
          ++correct;
        ++total;
      }
    }
    return total ? double(correct) / total : 0.0;
  };
  struct Cell {
    unsigned p{}, d{}, ctx{};
    double calibration_upper{};
    std::vector<double> oracle, eval;
  };
  std::vector<Cell> cells;
  for (const auto &[key, v] : held_by_key) {
    std::vector<double> calibration_durations;
    for (auto *r : cal_by_key.at(key))
      calibration_durations.push_back(r->duration_ns);
    Cell c{v.front()->prefill_tokens, v.front()->decode_items,
           v.front()->context_tokens,
           sloexp::nearest_rank(calibration_durations, upper_bound_quantile)};
    for (std::size_t k = 0; k < v.size(); ++k)
      (k % 2 ? c.eval : c.oracle).push_back(v[k]->duration_ns);
    if (c.oracle.empty() || c.eval.empty())
      continue;
    cells.push_back(std::move(c));
  }
  if (cells.empty())
    throw std::runtime_error(
        "heldout cells need at least two repetitions each");
  std::map<unsigned, std::vector<double>> training_duration_by_context;
  std::map<std::tuple<unsigned, unsigned, unsigned>, std::vector<double>>
      training_duration_by_cell;
  for (auto *r : train) {
    training_duration_by_context[r->context_tokens].push_back(r->duration_ns);
    training_duration_by_cell[{r->context_tokens, r->prefill_tokens,
                               r->decode_items}]
        .push_back(r->duration_ns);
  }
  std::map<unsigned, double> context_p99;
  for (auto &[context, durations] : training_duration_by_context)
    context_p99[context] = sloexp::nearest_rank(durations, 0.99);
  std::map<std::tuple<unsigned, unsigned, unsigned>, double> cell_p99;
  for (auto &[cell, durations] : training_duration_by_cell)
    cell_p99[cell] = sloexp::nearest_rank(durations, 0.99);
  const double ttft_target_multiplier = real(a, "ttft-target-multiplier", 3.0);
  const double tpot_target_multiplier = real(a, "tpot-target-multiplier", 2.0);
  const double risk_buffer_z = real(a, "risk-buffer-z", 1.0);
  const double risk_weight = real(a, "risk-weight", 0.25);
  std::ofstream out(need(a, "output"));
  out << "#metric,value\ntrain_observations," << train.size()
      << "\ncalibration_observations," << cal.size()
      << "\nheldout_observations," << held.size() << "\nqp_delta_ns," << delta
      << "\nupper_bound_quantile," << upper_bound_quantile
      << "\ndecision_candidate_cells," << cells.size()
      << "\nruntime_standardization_scale_ns," << runtime_scale
      << "\nqp_margin_z," << delta / runtime_scale
      << "\nttft_target_multiplier," << ttft_target_multiplier
      << "\ntpot_target_multiplier," << tpot_target_multiplier
      << "\nrisk_buffer_z," << risk_buffer_z << "\nrisk_weight," << risk_weight
      << "\nsocp_kappa," << kappa << "\ncovariance_socp_kappa,"
      << covariance_kappa << "\ncoefficient_residual_variance_ns2,"
      << residual_variance << "\nqp_calibration_coverage,"
      << coverage(cal, false) << "\nsocp_calibration_coverage,"
      << coverage(cal, true) << "\ncovariance_socp_calibration_coverage,"
      << covariance_coverage(cal) << "\nqp_heldout_coverage,"
      << coverage(held, false) << "\nsocp_heldout_coverage,"
      << coverage(held, true) << "\ncovariance_socp_heldout_coverage,"
      << covariance_coverage(held) << "\nqp_leave_cell_out_coverage,"
      << leave_cell_out_coverage(false) << "\nsocp_leave_cell_out_coverage,"
      << leave_cell_out_coverage(true)
      << "\ncovariance_socp_leave_cell_out_coverage,"
      << covariance_leave_cell_out_coverage() << "\n";
  out << "#decision,snapshot,solver,solver_success,solver_status,context,"
         "reference_z,"
         "recovery_mode,oracle_available,ttft_target_ns,tpot_target_ns,ttft_"
         "headroom_ns,tpot_headroom_ns,p_"
         "continuous,d_continuous,p_projected,d_projected,solve_ns,projection_"
         "distance,objective_mean,oracle_objective_mean,regret,ttft_violation_"
         "rate,tpot_violation_rate,joint_slo_rate,useful_work,ttft_z_slack,"
         "tpot_z_slack,boundary_penalty,projection_bound_source,empty_"
         "neighborhood\n";
  auto instances = sloexp::generate_instances(number(a, "seed", 1),
                                              number(a, "snapshots", 100));
  for (auto &i : instances) {
    const unsigned context = cells[i.id % cells.size()].ctx;
    std::vector<const Cell *> allowed;
    for (const auto &c : cells)
      if (c.ctx == context)
        allowed.push_back(&c);
    if (allowed.empty())
      continue;
    static constexpr double pressure_levels[] = {-2.0, -1.0, -0.5, 0.0, 0.5};
    const double reference_z = pressure_levels[i.id % 5];
    const Cell *reference = allowed[(i.id / 5) % allowed.size()];
    i.runtime_base = beta(0) + beta(3) * context;
    i.runtime_prefill = std::max(0.0, beta(1));
    i.runtime_decode = std::max(0.0, beta(2));
    i.batch_duration_scale = 50'000'000;
    i.fixed_margin_z = delta / runtime_scale;
    i.risk_buffer_z = risk_buffer_z;
    i.risk_weight = risk_weight;
    i.uncertainty_kappa = kappa;
    i.uncertainty_base = shape(0);
    i.uncertainty_prefill = shape(1);
    i.uncertainty_decode = shape(2);
    i.uncertainty_base = std::hypot(shape(0), shape(3) * context);
    i.covariance_kappa = covariance_kappa;
    for (int k = 0; k < 4; ++k) {
      i.covariance_factor_constant[k] =
          covariance_factor(0, k) + covariance_factor(3, k) * context;
      i.covariance_factor_prefill[k] = covariance_factor(1, k);
      i.covariance_factor_decode[k] = covariance_factor(2, k);
    }
    i.covariance_residual = std::sqrt(residual_variance);
    i.ttft_scale = runtime_scale;
    i.tpot_scale = runtime_scale;
    i.memory_available = 1;
    i.memory_base = 0;
    i.memory_prefill = 1.0 / i.token_budget;
    i.memory_decode = 1.0 / i.sequence_capacity;
    sloexp::Snapshot snap;
    snap.now_ns = 10'000'000'000;
    snap.window_ns = 60'000'000'000;
    const double measured_scale = context_p99.at(context);
    snap.ttft_target_ns = ttft_target_multiplier * measured_scale;
    snap.tpot_target_ns = tpot_target_multiplier * measured_scale;
    const double reference_runtime =
        cell_p99.at({context, reference->p, reference->d});
    const double desired_headroom = std::max(
        runtime_scale, reference_runtime - reference_z * runtime_scale);
    snap.previous_prefill = i.previous_prefill;
    snap.previous_decode = i.previous_decode;
    for (unsigned k = 0; k < 100; ++k) {
      snap.historical_ttft.push_back(
          {snap.now_ns - k * 1'000'000, 0.05 * snap.ttft_target_ns});
      snap.historical_tpot.push_back(
          {snap.now_ns - k * 1'000'000, 0.05 * snap.tpot_target_ns});
    }
    for (unsigned k = 0; k < 4; ++k)
      snap.prefills.push_back(
          {k,
           snap.now_ns - static_cast<std::uint64_t>(snap.ttft_target_ns -
                                                    desired_headroom),
           128, 128});
    for (unsigned k = 0; k < 64; ++k)
      snap.decodes.push_back(
          {k, snap.now_ns - static_cast<std::uint64_t>(snap.tpot_target_ns -
                                                       desired_headroom)});
    i.ttft_headroom = sloexp::sliding_window_headroom(snap, true);
    i.tpot_headroom = sloexp::sliding_window_headroom(snap, false);
    auto q = sloexp::solve_proxqp(i, 1),
         clarabel_q = sloexp::solve_clarabel_qp(i, 1),
         s = sloexp::solve_clarabel_socp(i, 1),
         covariance_s = sloexp::solve_clarabel_covariance_socp(i, 1);
    std::vector<double> historical_ttft, historical_tpot;
    for (const auto &v : snap.historical_ttft)
      historical_ttft.push_back(v.value_ns);
    for (const auto &v : snap.historical_tpot)
      historical_tpot.push_back(v.value_ns);
    const bool recovery =
        sloexp::nearest_rank(historical_ttft, 0.99) > snap.ttft_target_ns ||
        sloexp::nearest_rank(historical_tpot, 0.99) > snap.tpot_target_ns;
    auto realized_risk_penalty = [&](double runtime) {
      const double zt = sloexp::standardized_runtime_slack(
          runtime, i.ttft_headroom, i.ttft_scale);
      const double zd = sloexp::standardized_runtime_slack(
          runtime, i.tpot_headroom, i.tpot_scale);
      return 0.5 * i.risk_weight *
             (std::pow(std::max(0.0, zt + i.risk_buffer_z), 2) +
              std::pow(std::max(0.0, zd + i.risk_buffer_z), 2));
    };
    const Cell *oracle_cell = nullptr;
    bool oracle_feasible = false;
    double oracle_obj = std::numeric_limits<double>::infinity(),
           oracle_violation = std::numeric_limits<double>::infinity();
    for (auto *c : allowed) {
      std::size_t feasible = 0;
      bool resources_ok = true;
      double obj = 0, violation = 0;
      for (double t : c->oracle) {
        auto score = sloexp::score_one_step(
            snap, {c->p, c->d, t, 0}, t, 512, 64, i.reward_prefill,
            i.reward_decode, i.runtime_weight, i.rho_prefill, i.rho_decode,
            i.batch_duration_scale);
        feasible += score.ttft_satisfied && score.tpot_satisfied;
        resources_ok =
            resources_ok && sloexp::resource_feasible(i, c->p, c->d, t);
        obj += score.objective + realized_risk_penalty(t);
        violation +=
            std::max(0.0, score.ttft_p99_ns / snap.ttft_target_ns - 1) +
            std::max(0.0, score.tpot_p99_ns / snap.tpot_target_ns - 1);
      }
      obj /= c->oracle.size();
      violation /= c->oracle.size();
      if (!resources_ok)
        continue;
      const bool slo_ok = double(feasible) / c->oracle.size() >= 0.99;
      if ((slo_ok && (!oracle_feasible || obj < oracle_obj)) ||
          (recovery && !oracle_feasible && !slo_ok &&
           (violation < oracle_violation ||
            (violation == oracle_violation && obj < oracle_obj)))) {
        oracle_feasible = slo_ok;
        oracle_obj = obj;
        oracle_violation = violation;
        oracle_cell = c;
      }
    }
    double oracle_eval = std::numeric_limits<double>::quiet_NaN();
    if (oracle_cell) {
      oracle_eval = 0;
      for (double t : oracle_cell->eval)
        oracle_eval +=
            sloexp::score_one_step(
                snap, {oracle_cell->p, oracle_cell->d, t, 0}, t, 512, 64,
                i.reward_prefill, i.reward_decode, i.runtime_weight,
                i.rho_prefill, i.rho_decode, i.batch_duration_scale)
                .objective +
            realized_risk_penalty(t);
      oracle_eval /= oracle_cell->eval.size();
    }
    for (auto pair : {std::pair{"qp", q},
                      std::pair{"clarabel_qp", clarabel_q},
                      std::pair{"socp", s},
                      std::pair{"covariance_socp", covariance_s}}) {
      std::vector<std::pair<double, const Cell *>> near;
      for (auto *c : allowed) {
        double dp = (c->p - pair.second.p) / 512.0,
               dd = (c->d - pair.second.d) / 64.0;
        near.push_back({dp * dp + dd * dd, c});
      }
      std::sort(near.begin(), near.end(), [](auto &a, auto &b) {
        if (a.first != b.first)
          return a.first < b.first;
        return std::tie(a.second->p, a.second->d) <
               std::tie(b.second->p, b.second->d);
      });
      const Cell *chosen = nullptr;
      std::string chosen_bound_source = "none";
      double chosen_dist = std::numeric_limits<double>::quiet_NaN(),
             chosen_bound = std::numeric_limits<double>::quiet_NaN(),
             best_violation = std::numeric_limits<double>::infinity(),
             best_objective = std::numeric_limits<double>::infinity();
      for (auto [dist, c] : near) {
        const double mean =
            i.runtime_base + i.runtime_prefill * c->p + i.runtime_decode * c->d;
        double model_margin = delta;
        if (pair.first == std::string("socp"))
          model_margin =
              kappa * std::sqrt(std::pow(i.uncertainty_base, 2) +
                                std::pow(i.uncertainty_prefill * c->p, 2) +
                                std::pow(i.uncertainty_decode * c->d, 2));
        else if (pair.first == std::string("covariance_socp")) {
          double variance = std::pow(i.covariance_residual, 2);
          for (int k = 0; k < 4; ++k) {
            const double component = i.covariance_factor_constant[k] +
                                     i.covariance_factor_prefill[k] * c->p +
                                     i.covariance_factor_decode[k] * c->d;
            variance += component * component;
          }
          model_margin = covariance_kappa * std::sqrt(variance);
        }
        const double model_bound = mean + model_margin;
        const bool use_cell_bound = c->calibration_upper < model_bound;
        const double bound =
            use_cell_bound ? c->calibration_upper : model_bound;
        const char *bound_source = use_cell_bound ? "cell_p98" : "global_model";
        auto sc = sloexp::score_one_step(
            snap, {c->p, c->d, bound, 0}, bound, 512, 64, i.reward_prefill,
            i.reward_decode, i.runtime_weight, i.rho_prefill, i.rho_decode,
            i.batch_duration_scale);
        const bool feasible = sloexp::resource_feasible(i, c->p, c->d, bound) &&
                              sc.ttft_satisfied && sc.tpot_satisfied;
        if (feasible) {
          chosen = c;
          chosen_dist = std::sqrt(dist);
          chosen_bound = bound;
          chosen_bound_source = bound_source;
          break;
        }
        if (recovery && sloexp::resource_feasible(i, c->p, c->d, bound)) {
          const double violation =
              std::max(0.0, sc.ttft_p99_ns / snap.ttft_target_ns - 1) +
              std::max(0.0, sc.tpot_p99_ns / snap.tpot_target_ns - 1);
          if (violation < best_violation ||
              (violation == best_violation && sc.objective < best_objective)) {
            best_violation = violation;
            best_objective = sc.objective;
            chosen = c;
            chosen_dist = std::sqrt(dist);
            chosen_bound = bound;
            chosen_bound_source = bound_source;
          }
        }
      }
      const bool empty = !chosen;
      if (!chosen) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        out << i.id << ',' << pair.first << ',' << pair.second.success << ','
            << pair.second.status << ',' << context << ',' << reference_z << ','
            << recovery << ',' << (oracle_cell != nullptr) << ','
            << snap.ttft_target_ns << ',' << snap.tpot_target_ns << ','
            << i.ttft_headroom << ',' << i.tpot_headroom << ',' << pair.second.p
            << ',' << pair.second.d << ',' << nan << ',' << nan << ','
            << pair.second.solve_ns << ',' << nan << ',' << nan << ','
            << oracle_eval << ',' << nan << ',' << nan << ',' << nan << ','
            << nan << ',' << nan << ',' << nan << ',' << nan << ',' << nan
            << ",none," << empty << '\n';
        continue;
      }
      double obj = 0;
      std::size_t tv = 0, pv = 0, joint = 0;
      for (double t : chosen->eval) {
        auto sc = sloexp::score_one_step(
            snap, {chosen->p, chosen->d, t, 0}, t, 512, 64, i.reward_prefill,
            i.reward_decode, i.runtime_weight, i.rho_prefill, i.rho_decode,
            i.batch_duration_scale);
        obj += sc.objective + realized_risk_penalty(t);
        tv += !sc.ttft_satisfied;
        pv += !sc.tpot_satisfied;
        joint += sc.ttft_satisfied && sc.tpot_satisfied;
      }
      obj /= chosen->eval.size();
      const double ttft_z = sloexp::standardized_runtime_slack(
          chosen_bound, i.ttft_headroom, i.ttft_scale);
      const double tpot_z = sloexp::standardized_runtime_slack(
          chosen_bound, i.tpot_headroom, i.tpot_scale);
      const double boundary_penalty =
          0.5 * i.risk_weight *
          (std::pow(std::max(0.0, ttft_z + i.risk_buffer_z), 2) +
           std::pow(std::max(0.0, tpot_z + i.risk_buffer_z), 2));
      out << i.id << ',' << pair.first << ',' << pair.second.success << ','
          << pair.second.status << ',' << context << ',' << reference_z << ','
          << recovery << ',' << (oracle_cell != nullptr) << ','
          << snap.ttft_target_ns << ',' << snap.tpot_target_ns << ','
          << i.ttft_headroom << ',' << i.tpot_headroom << ',' << pair.second.p
          << ',' << pair.second.d << ',' << chosen->p << ',' << chosen->d << ','
          << pair.second.solve_ns << ',' << chosen_dist << ',' << obj << ','
          << oracle_eval << ',' << obj - oracle_eval << ','
          << double(tv) / chosen->eval.size() << ','
          << double(pv) / chosen->eval.size() << ','
          << double(joint) / chosen->eval.size() << ','
          << double(chosen->p) / 512 + double(chosen->d) / 64 << ',' << ttft_z
          << ',' << tpot_z << ',' << boundary_penalty << ','
          << chosen_bound_source << ',' << empty << '\n';
    }
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2)
      throw std::invalid_argument("usage: slo_optimization_experiment "
                                  "<synthetic|collect|evaluate> [options]");
    auto a = args(argc, argv);
    std::string mode = argv[1];
    if (mode == "synthetic")
      synthetic(a);
    else if (mode == "collect")
      collect(a);
    else if (mode == "evaluate")
      evaluate(a);
    else
      throw std::invalid_argument("unknown mode");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "slo_optimization_experiment: " << e.what() << '\n';
    return 1;
  }
}
