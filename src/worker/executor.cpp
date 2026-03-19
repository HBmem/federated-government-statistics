#include "./executor.hpp"

#include "../db/sql_template.hpp"
#include "../common/metrics.hpp"

#include <unordered_map>
#include <stdexcept>
#include <string>

namespace {

/*
  Convert the first row of a PGresult into a JSON object using strings.
*/
Json pg_single_row_to_json(PGresult* r) {
  Json row = Json::object();
  int nrows = PQntuples(r);
  int ncols = PQnfields(r);

  if (nrows <= 0) return row;

  for (int c = 0; c < ncols; ++c) {
    std::string name = PQfname(r, c);
    row[name] = PQgetvalue(r, 0, c);
  }
  return row;
}

/*
  Get the number of rows returned in a PGresult.
*/
int pg_rows_returned(PGresult* r) {
  return PQntuples(r);
}

/*
  Parse EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) output.

  PostgreSQL returns a single row containing a JSON array.
  We extract:
  - Execution Time
  - Shared Hit Blocks
  - Shared Read Blocks
  - Temp Read Blocks
  - Temp Written Blocks

  These may appear at the top level and/or under Plan.
*/
Json parse_explain_json_metrics(PGresult* r) {
  Json out = Json::object();

  if (PQntuples(r) <= 0 || PQnfields(r) <= 0) {
    return out;
  }

  const char* raw = PQgetvalue(r, 0, 0);
  if (!raw) return out;

  Json explain_root = Json::parse(raw);

  if (!explain_root.is_array() || explain_root.empty()) return out;

  Json obj = explain_root.at(0);

  if (obj.contains("Execution Time")) {
    out["db_cpu_time_ms"] = obj["Execution Time"];
  }

  if (obj.contains("Plan")) {
    Json plan = obj["Plan"];

    out["db_shared_hits"] = plan.value("Shared Hit Blocks", 0);
    out["db_reads"] = plan.value("Shared Read Blocks", 0);
    out["db_temp_reads"] = plan.value("Temp Read Blocks", 0);
    out["db_temp_writes"] = plan.value("Temp Written Blocks", 0);
  }

  return out;
}

/*
  Execute EXPLAIN ANALYZE BUFFERS FORMAT JSON for a query and return parsed metrics.
*/
Json collect_explain_metrics(PgClient& db, const AppConfig& cfg, const std::string& sql) {
  Json out = Json::object();

  std::string explain_sql = "EXPLAIN (FORMAT JSON";

  if (cfg.performance.explain_analyze) {
    explain_sql += ", ANALYZE";
  }
  if (cfg.performance.explain_buffers) {
    explain_sql += ", BUFFERS";
  }

  explain_sql += ") " + sql;

  PGresult* r = db.exec(explain_sql);
  out = parse_explain_json_metrics(r);
  PQclear(r);

  return out;
}

}  // namespace

namespace executor {

Json run(Logger& log, const AppConfig& cfg, const NodeInfo& node, PgClient& db,
         const std::string& schema_type) {
  Json out;
  out["ok"] = true;
  out["schema_type"] = schema_type;
  out["county_fips"] = node.county_fips;
  out["metrics"] = Json::object();
  out["query_metrics"] = Json::array();

  try {
    db.exec_ok("SET statement_timeout = " + std::to_string(cfg.db.statement_timeout_ms) + ";");
    db.exec_ok("SET lock_timeout = " + std::to_string(cfg.db.lock_timeout_ms) + ";");
  } catch (const std::exception& ex) {
    out["db_tuning_warning"] = ex.what();
  }

  std::unordered_map<std::string, std::string> params{
      {"start_date", cfg.run.start_date},
      {"end_date", cfg.run.end_date},
  };

  for (const auto& metric : cfg.enabled_metric_list()) {
    Json qmeta;
    qmeta["metric_name"] = metric;
    qmeta["success"] = false;
    qmeta["rows_returned"] = 0;
    qmeta["query_ms"] = 0;
    qmeta["db_cpu_time_ms"] = nullptr;
    qmeta["db_shared_hits"] = nullptr;
    qmeta["db_reads"] = nullptr;
    qmeta["db_temp_reads"] = nullptr;
    qmeta["db_temp_writes"] = nullptr;
    qmeta["error_message"] = nullptr;

    if (!cfg.registry.count(schema_type) || !cfg.registry.at(schema_type).count(metric)) {
      std::string err = "No template registered for schema=" + schema_type + " metric=" + metric;

      Json fail;
      fail["ok"] = false;
      fail["error"] = err;

      qmeta["template_path"] = nullptr;
      qmeta["error_message"] = err;

      out["metrics"][metric] = fail;
      out["query_metrics"].push_back(qmeta);
      continue;
    }

    const std::string tpl_path = cfg.registry.at(schema_type).at(metric);
    qmeta["template_path"] = tpl_path;

    try {
      std::string tpl = sql_template::load_file(tpl_path);
      std::string sql = sql_template::substitute(db.raw(), tpl, params);

      perf::Stopwatch query_timer;
      PGresult* r = db.exec(sql);
      int rows_returned = pg_rows_returned(r);
      Json row = pg_single_row_to_json(r);
      PQclear(r);

      qmeta["query_ms"] = query_timer.elapsed_ms();
      qmeta["rows_returned"] = rows_returned;
      qmeta["success"] = true;

      row["county_fips"] = node.county_fips;
      row["metric"] = metric;
      row["template_path"] = tpl_path;

      if (cfg.performance.capture_query_explain_metrics) {
        try {
          Json explain_metrics = collect_explain_metrics(db, cfg, sql);
          for (auto it = explain_metrics.begin(); it != explain_metrics.end(); ++it) {
            qmeta[it.key()] = it.value();
          }
        } catch (const std::exception& ex) {
          qmeta["explain_error"] = ex.what();
        }
      }

      out["metrics"][metric] = row;
      out["query_metrics"].push_back(qmeta);
    } catch (const std::exception& ex) {
      Json fail;
      fail["ok"] = false;
      fail["error"] = ex.what();
      fail["template_path"] = tpl_path;

      qmeta["error_message"] = ex.what();

      out["metrics"][metric] = fail;
      out["query_metrics"].push_back(qmeta);
    }
  }

  log.debug("executor finished for county_fips=" + node.county_fips);
  return out;
}

}  // namespace executor