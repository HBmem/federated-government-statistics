#include "executor.hpp"

#include "../db/sql_template.hpp"

#include <unordered_map>

static Json pg_single_row_to_json(PGresult* r) {
  Json row;
  int nrows = PQntuples(r);
  int ncols = PQnfields(r);

  if (nrows <= 0) return row;

  for (int c = 0; c < ncols; ++c) {
    std::string name = PQfname(r, c);
    row[name] = PQgetvalue(r, 0, c);
  }
  return row;
}

namespace executor {

Json run(Logger& log, const AppConfig& cfg, const NodeInfo& node, PgClient& db,
         const std::string& schema_type) {
  Json out;
  out["ok"] = true;
  out["schema_type"] = schema_type;
  out["county_fips"] = node.county_fips;
  out["metrics"] = Json::object();

  // Best-effort DB tuning.
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
    if (!cfg.registry.count(schema_type) || !cfg.registry.at(schema_type).count(metric)) {
      Json fail;
      fail["ok"] = false;
      fail["error"] = "No template registered for schema=" + schema_type + " metric=" + metric;
      out["metrics"][metric] = fail;
      continue;
    }

    const std::string tpl_path = cfg.registry.at(schema_type).at(metric);

    try {
      std::string tpl = sql_template::load_file(tpl_path);
      std::string sql = sql_template::substitute(db.raw(), tpl, params);

      PGresult* r = db.exec(sql);
      Json row = pg_single_row_to_json(r);
      PQclear(r);

      // Attach county_fips for consistent grouping at coordinator.
      row["county_fips"] = node.county_fips;
      row["metric"] = metric;
      row["template_path"] = tpl_path;

      out["metrics"][metric] = row;
    } catch (const std::exception& ex) {
      Json fail;
      fail["ok"] = false;
      fail["error"] = ex.what();
      fail["template_path"] = tpl_path;
      out["metrics"][metric] = fail;
    }
  }

  log.debug("executor finished for county_fips=" + node.county_fips);
  return out;
}

} // namespace executor