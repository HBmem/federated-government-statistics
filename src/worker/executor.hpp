#pragma once

#include "../common/config.hpp"
#include "../common/json.hpp"
#include "../db/pg_client.hpp"
#include "../common/logging.hpp"

/*
  Executes query templates for each enabled metric module.

  - Uses schema_type to select template path from registry.json
  - Substitutes :start_date / :end_date safely
  - Returns a JSON object with per-metric results and quality counters
*/
namespace executor {

Json run(Logger& log, const AppConfig& cfg, const NodeInfo& node, PgClient& db,
         const std::string& schema_type);

}