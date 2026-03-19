#include "config.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "strings.hpp"

static Json read_json_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Failed to open JSON file: " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  return Json::parse(ss.str());
}

static std::vector<std::string> split_ws(const std::string& s) {
  std::istringstream iss(s);
  std::vector<std::string> out;
  std::string token;
  while (iss >> token) out.push_back(token);
  return out;
}

static std::unordered_map<std::string, std::string> parse_kv_tokens(const std::vector<std::string>& tokens) {
  std::unordered_map<std::string, std::string> kv;
  for (const auto& t : tokens) {
    auto pos = t.find('=');
    if (pos == std::string::npos) continue;
    kv[t.substr(0, pos)] = t.substr(pos + 1);
  }
  return kv;
}

static std::vector<NodeInfo> load_nodes_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Failed to open nodes file: " + path);

  std::vector<NodeInfo> nodes;
  std::string line;

  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty()) continue;
    if (line.rfind("#", 0) == 0) continue;

    if (line == "COORDINATOR") {
      NodeInfo n;
      n.is_coordinator = true;
      nodes.push_back(n);
      continue;
    }

    auto tokens = split_ws(line);
    auto kv = parse_kv_tokens(tokens);

    NodeInfo n;
    n.is_coordinator = false;
    n.host = kv["host"];
    n.port = kv.count("port") ? std::stoi(kv["port"]) : 5432;
    n.dbname = kv["dbname"];
    n.user = kv["user"];
    n.password = kv["password"];
    n.county_fips = kv["county_fips"];
    if (kv.count("label")) n.label = kv["label"];

    // Basic validation: ensure required keys exist for workers.
    if (n.host.empty() || n.dbname.empty() || n.user.empty() || n.password.empty() || n.county_fips.empty())
      throw std::runtime_error("Invalid worker line in nodes.txt (missing required keys): " + line);

    nodes.push_back(n);
  }

  if (nodes.empty() || !nodes[0].is_coordinator)
    throw std::runtime_error("nodes.txt must begin with COORDINATOR as rank 0.");

  return nodes;
}

static void apply_coordinator_config(AppConfig& cfg, const Json& j) {
  // run
  cfg.run.start_date = j["run"]["time_window"]["start_date"].get<std::string>();
  cfg.run.end_date   = j["run"]["time_window"]["end_date"].get<std::string>();
  cfg.run.category   = j["run"]["category"].get<std::string>();
  cfg.run.output_dir = j["run"]["output_dir"].get<std::string>();
  cfg.run.include_node_payloads = j["run"]["include_node_payloads"].get<bool>();

    // performance
  cfg.performance.capture_process_metrics = j["performance"]["capture_process_metrics"].get<bool>();
  cfg.performance.capture_query_explain_metrics = j["performance"]["capture_query_explain_metrics"].get<bool>();
  cfg.performance.explain_analyze = j["performance"]["explain_analyze"].get<bool>();
  cfg.performance.explain_buffers = j["performance"]["explain_buffers"].get<bool>();

  // fault_tolerance
  cfg.fault.worker_response_timeout_sec = j["fault_tolerance"]["worker_response_timeout_sec"].get<int>();
  cfg.fault.worker_query_timeout_sec    = j["fault_tolerance"]["worker_query_timeout_sec"].get<int>();
  cfg.fault.max_retries_per_phase       = j["fault_tolerance"]["max_retries_per_phase"].get<int>();

  // db
  cfg.db.statement_timeout_ms = j["db"]["statement_timeout_ms"].get<int>();
  cfg.db.lock_timeout_ms      = j["db"]["lock_timeout_ms"].get<int>();

  // metrics enabled
  cfg.metrics.population   = j["metrics"]["enabled"]["population"].get<bool>();
  cfg.metrics.demographics = j["metrics"]["enabled"]["demographics"].get<bool>();
  cfg.metrics.income       = j["metrics"]["enabled"]["income"].get<bool>();
  cfg.metrics.age          = j["metrics"]["enabled"]["age"].get<bool>();
  cfg.metrics.unemployment = j["metrics"]["enabled"]["unemployment"].get<bool>();
  cfg.metrics.quality      = j["metrics"]["enabled"]["quality"].get<bool>();
  cfg.metrics.bucket_config_path = j["metrics"]["bucket_config_path"].get<std::string>();

  // residency_rules
  cfg.residency.require_county_match = j["residency_rules"]["require_county_match"].get<bool>();
  cfg.residency.strict_missing_fields = j["residency_rules"]["strict_missing_fields"].get<bool>();

  cfg.residency.require_moved_in_on_or_before_end =
      j["residency_rules"]["date_logic"]["require_moved_in_on_or_before_end"].get<bool>();
  cfg.residency.exclude_if_moved_out_before_start =
      j["residency_rules"]["date_logic"]["exclude_if_moved_out_before_start"].get<bool>();
  cfg.residency.exclude_if_dead_on_or_before_end =
      j["residency_rules"]["date_logic"]["exclude_if_dead_on_or_before_end"].get<bool>();

  // templates
  cfg.templates.registry_path = j["query_templates"]["registry_path"].get<std::string>();
  cfg.templates.allow_probe_fallback = j["query_templates"]["allow_probe_fallback"].get<bool>();

  // logging
  cfg.logging.log_dir = j["logging"]["log_dir"].get<std::string>();
  cfg.logging.level   = j["logging"]["level"].get<std::string>();
}

static void apply_registry(AppConfig& cfg, const Json& j) {
  // registry.json structure:
  // { "schemas": { "A": { "population": "path", ... }, "B": {...}, "C": {...} } }
  const auto& schemas = j["schemas"];
  for (auto it = schemas.begin(); it != schemas.end(); ++it) {
    std::string schema_key = it.key();
    std::unordered_map<std::string, std::string> metric_map;

    for (auto it2 = it.value().begin(); it2 != it.value().end(); ++it2) {
      metric_map[it2.key()] = it2.value().get<std::string>();
    }
    cfg.registry[schema_key] = std::move(metric_map);
  }
}

static std::string arg_value(int argc, char** argv, const std::string& key, const std::string& def = "") {
  // Supports: --key=value or --key value
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind(key + "=", 0) == 0) return a.substr(key.size() + 1);
    if (a == key && i + 1 < argc) return argv[i + 1];
  }
  return def;
}

AppConfig AppConfig::load_from_cli(int argc, char** argv) {
  AppConfig cfg;

  // Allow override of config paths
  cfg.coordinator_config_path = arg_value(argc, argv, "--config", cfg.coordinator_config_path);
  cfg.nodes_path              = arg_value(argc, argv, "--nodes", cfg.nodes_path);

  // Load nodes first (used by workers for county_fips)
  cfg.nodes_by_rank = load_nodes_file(cfg.nodes_path);

  // Load coordinator config JSON
  Json jcfg = read_json_file(cfg.coordinator_config_path);
  apply_coordinator_config(cfg, jcfg);

  // CLI override for time window (defaults to config file)
  const std::string cli_start = arg_value(argc, argv, "--start", "");
  const std::string cli_end   = arg_value(argc, argv, "--end", "");
  if (!cli_start.empty()) cfg.run.start_date = cli_start;
  if (!cli_end.empty())   cfg.run.end_date = cli_end;

  // Load registry for templates
  Json jreg = read_json_file(cfg.templates.registry_path);
  apply_registry(cfg, jreg);

  return cfg;
}

const NodeInfo& AppConfig::node_for_rank(int rank) const {
  if (rank < 0 || rank >= static_cast<int>(nodes_by_rank.size()))
    throw std::runtime_error("Rank out of range for nodes_by_rank.");
  return nodes_by_rank[rank];
}

NodeInfo& AppConfig::node_for_rank(int rank) {
  if (rank < 0 || rank >= static_cast<int>(nodes_by_rank.size()))
    throw std::runtime_error("Rank out of range for nodes_by_rank.");
  return nodes_by_rank[rank];
}

std::vector<std::string> AppConfig::enabled_metric_list() const {
  std::vector<std::string> out;
  if (metrics.population) out.push_back("population");
  if (metrics.demographics) out.push_back("demographics");
  if (metrics.income) out.push_back("income");
  if (metrics.age) out.push_back("age");
  if (metrics.unemployment) out.push_back("unemployment");
  // quality is embedded in each query result via counters, so not dispatched separately.
  return out;
}