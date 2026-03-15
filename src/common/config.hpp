#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "json.hpp"

/*
  Centralized configuration structures.

  - Reads Config/coordinator_config.json
  - Reads queries/registry.json
  - Reads Config/nodes.txt
  - Applies CLI overrides for time window
*/

struct NodeInfo {
  bool is_coordinator = false;

  // Connection info for worker nodes.
  std::string host;
  int port = 5432;
  std::string dbname;
  std::string user;
  std::string password;

  // Always provided via nodes.txt (used to label results, including Schema B).
  std::string county_fips;

  // Optional label (not required).
  std::string label;
};

struct LoggingConfig {
  std::string level = "info";
  std::string log_dir = "output/logs";
};

struct RunConfig {
  std::string start_date; // YYYY-MM-DD
  std::string end_date;   // YYYY-MM-DD
  std::string category = "demographic";
  std::string output_dir = "output/runs";
  bool include_node_payloads = true;
};

struct FaultConfig {
  int worker_response_timeout_sec = 20;
  int worker_query_timeout_sec = 15;
  int max_retries_per_phase = 1;
};

struct MetricsConfig {
  bool population = true;
  bool demographics = true;
  bool income = true;
  bool age = true;
  bool unemployment = true;
  bool quality = true;

  std::string bucket_config_path = "config/buckets.json";
};

struct ResidencyRules {
  bool require_county_match = true;
  bool strict_missing_fields = true;

  bool require_moved_in_on_or_before_end = true;
  bool exclude_if_moved_out_before_start = true;
  bool exclude_if_dead_on_or_before_end = true;
};

struct QueryTemplatesConfig {
  std::string registry_path = "queries/registry.json";
  bool allow_probe_fallback = true;
};

struct DbTuningConfig {
  int statement_timeout_ms = 12000;
  int lock_timeout_ms = 3000;
};

struct AppConfig {
  // Paths
  std::string nodes_path = "config/nodes.txt";
  std::string coordinator_config_path = "config/coordinator_config.json";

  // Loaded from files
  RunConfig run;
  FaultConfig fault;
  DbTuningConfig db;
  MetricsConfig metrics;
  ResidencyRules residency;
  QueryTemplatesConfig templates;
  LoggingConfig logging;

  // Loaded from nodes.txt
  std::vector<NodeInfo> nodes_by_rank; // index == rank

  // Loaded from registry.json:
  // schema_type -> metric -> template path
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>> registry;

  static AppConfig load_from_cli(int argc, char** argv);

  // Helpers
  const NodeInfo& node_for_rank(int rank) const;
  NodeInfo& node_for_rank(int rank);

  std::vector<std::string> enabled_metric_list() const;
};