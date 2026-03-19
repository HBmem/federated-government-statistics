#include "report_writer.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

/*
  Minimal aggregation for this iteration:
  - population_count: sum across nodes
  - employed_count/not_employed_count: sum
  - any histogram bucket fields: sum if present
  - income_sum/count/min/max: merge safely where present

  Coordinator can be extended later to compute derived values (mean, median approx).
*/
static bool is_number_string(const std::string& s) {
  if (s.empty()) return false;
  // Accept simple integers/decimals with optional leading '-'
  size_t i = 0;
  if (s[0] == '-') i = 1;
  bool has_digit = false;
  bool has_dot = false;
  for (; i < s.size(); ++i) {
    if (std::isdigit(static_cast<unsigned char>(s[i]))) { has_digit = true; continue; }
    if (s[i] == '.' && !has_dot) { has_dot = true; continue; }
    return false;
  }
  return has_digit;
}

static long long to_ll(const Json& j, const std::string& k) {
  if (!j.contains(k)) return 0;
  const std::string s = j[k].get<std::string>();
  if (!is_number_string(s)) return 0;
  return std::stoll(s);
}

static double to_double(const Json& j, const std::string& k, bool& ok) {
  ok = false;
  if (!j.contains(k)) return 0.0;
  const std::string s = j[k].get<std::string>();
  if (!is_number_string(s)) return 0.0;
  ok = true;
  return std::stod(s);
}

static std::string utc_timestamp_compact() {
  std::time_t t = std::time(nullptr);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y%m%dT%H%M%SZ", &tm);
  return std::string(buf);
}

namespace report_writer {

Json build(const AppConfig& cfg,
           const std::unordered_map<int, Json>& discovery_by_rank,
           const std::unordered_map<int, Json>& results_by_rank,
           const std::unordered_map<int, std::string>& failure_by_rank) {
  Json report;

  // Run metadata
  report["run"] = {
      {"time_window", {{"start_date", cfg.run.start_date}, {"end_date", cfg.run.end_date}}},
      {"category", cfg.run.category},
      {"enabled_metrics", cfg.enabled_metric_list()},
      {"include_node_payloads", cfg.run.include_node_payloads}
  };

  // Per-node section
  report["nodes"] = Json::array();

  // Global aggregates
  Json global;
  global["population_count"] = 0LL;
  global["employed_count"] = 0LL;
  global["not_employed_count"] = 0LL;

  // Merge some well-known histogram keys if present (income + age).
  std::vector<std::string> sum_keys = {
      "inc_lt_25k","inc_25_50k","inc_50_75k","inc_75_100k","inc_100_150k","inc_150_200k","inc_200k_plus",
      "age_0_4","age_5_9","age_10_14","age_15_17","age_18_24","age_25_34","age_35_44","age_45_54",
      "age_55_64","age_65_74","age_75_84","age_85_plus"
  };
  for (const auto& k : sum_keys) global[k] = 0LL;

  // Merge income numeric partials when present.
  double income_sum = 0.0;
  long long income_count = 0;
  bool has_income_min = false, has_income_max = false;
  double income_min = 0.0, income_max = 0.0;

  for (int r = 1; r < static_cast<int>(cfg.nodes_by_rank.size()); ++r) {
    Json node_obj;
    node_obj["rank"] = r;
    node_obj["county_fips"] = cfg.node_for_rank(r).county_fips;

    if (failure_by_rank.count(r)) {
      node_obj["status"] = "failed";
      node_obj["error"] = failure_by_rank.at(r);
      report["nodes"].push_back(node_obj);
      continue;
    }

    node_obj["status"] = "ok";
    if (discovery_by_rank.count(r)) node_obj["discovery"] = discovery_by_rank.at(r);

    if (results_by_rank.count(r)) {
      const Json& res = results_by_rank.at(r);
      node_obj["results"] = cfg.run.include_node_payloads ? res : Json::object();

      if (res.contains("worker_metrics")) {
        node_obj["worker_metrics"] = res["worker_metrics"];
      }

      if (res.contains("query_metrics")) {
        node_obj["query_metrics"] = res["query_metrics"];
      }

      // Aggregate across metrics returned.
      if (res.contains("metrics")) {
        const Json& metrics = res["metrics"];

        // population
        if (metrics.contains("population") && metrics["population"].contains("population_count")) {
          global["population_count"] = global["population_count"].get<long long>() +
                                       to_ll(metrics["population"], "population_count");
        }

        // unemployment
        if (metrics.contains("unemployment")) {
          global["employed_count"] = global["employed_count"].get<long long>() +
                                     to_ll(metrics["unemployment"], "employed_count");
          global["not_employed_count"] = global["not_employed_count"].get<long long>() +
                                         to_ll(metrics["unemployment"], "not_employed_count");
        }

        // income numeric partials (Schema A/C only)
        if (metrics.contains("income")) {
          bool okSum=false, okMin=false, okMax=false;
          double s = to_double(metrics["income"], "income_sum", okSum);
          double mn = to_double(metrics["income"], "income_min", okMin);
          double mx = to_double(metrics["income"], "income_max", okMax);
          long long cnt = to_ll(metrics["income"], "income_count");

          if (okSum) income_sum += s;
          income_count += cnt;

          if (okMin) {
            if (!has_income_min) { income_min = mn; has_income_min = true; }
            else income_min = std::min(income_min, mn);
          }
          if (okMax) {
            if (!has_income_max) { income_max = mx; has_income_max = true; }
            else income_max = std::max(income_max, mx);
          }

          // Histogram buckets
          for (const auto& k : sum_keys) {
            if (metrics["income"].contains(k)) {
              global[k] = global[k].get<long long>() + to_ll(metrics["income"], k);
            }
          }
        }

        // age histogram
        if (metrics.contains("age")) {
          for (const auto& k : sum_keys) {
            if (metrics["age"].contains(k)) {
              global[k] = global[k].get<long long>() + to_ll(metrics["age"], k);
            }
          }
        }
      }
    }

    report["nodes"].push_back(node_obj);
  }

  // Attach income numeric merges
  global["income_sum"] = income_sum;
  global["income_count"] = income_count;
  if (has_income_min) global["income_min"] = income_min;
  if (has_income_max) global["income_max"] = income_max;

  report["global_aggregates"] = global;

  // Failures summary
  Json failures = Json::array();
  for (const auto& kv : failure_by_rank) {
    failures.push_back({{"rank", kv.first}, {"error", kv.second}});
  }
  report["failures"] = failures;

  return report;
}

void write_to_disk(const AppConfig& cfg, const Json& report) {
  std::filesystem::create_directories(cfg.run.output_dir);

  std::string fname = cfg.run.output_dir + "/run_" + utc_timestamp_compact() + ".json";
  std::ofstream out(fname);
  if (!out) throw std::runtime_error("Failed to write report: " + fname);

  out << report.dump(2) << "\n";
}

} // namespace report_writer