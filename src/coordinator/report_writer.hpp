#pragma once

#include "../common/config.hpp"
#include "../common/json.hpp"

#include <unordered_map>

/*
  Builds and writes the final coordinator JSON report.

  Includes:
  - run metadata (time window, enabled metrics)
  - per-node schema + results
  - failures list
  - global merged aggregates (basic merging)
*/
namespace report_writer {

Json build(const AppConfig& cfg,
           const std::unordered_map<int, Json>& discovery_by_rank,
           const std::unordered_map<int, Json>& results_by_rank,
           const std::unordered_map<int, std::string>& failure_by_rank);

void write_to_disk(const AppConfig& cfg, const Json& report);

}