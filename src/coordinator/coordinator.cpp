#include "coordinator.hpp"

#include <mpi.h>
#include <unordered_map>

#include "../common/mpi_codec.hpp"
#include "../common/time.hpp"
#include "report_writer.hpp"

/*
  Coordinator flow:
  1) Send "discover" to all workers
  2) Collect discovery responses with timeout
  3) Send "execute" plan to responsive workers, including their schema_type
  4) Collect results with timeout
  5) Write final JSON report
  6) Send "shutdown" to all workers
*/
void Coordinator::run() {
  log_.info("coordinator starting with world_size=" + std::to_string(world_size_));

  // ---- Phase 1: Discovery ----
  Json discover_msg;
  discover_msg["type"] = "discover";

  for (int r = 1; r < world_size_; ++r) {
    mpi_codec::send_string(r, mpi_codec::TAG_CONTROL, discover_msg.dump(), MPI_COMM_WORLD);
  }

  const int64_t t0 = now_ms();
  std::unordered_map<int, Json> discovery_by_rank;
  std::unordered_map<int, std::string> failure_by_rank;

  while (static_cast<int>(discovery_by_rank.size()) < (world_size_ - 1)) {
    if ((now_ms() - t0) > cfg_.fault.worker_response_timeout_sec * 1000LL) {
      break;
    }

    int src = -1;
    std::string payload;
    if (!mpi_codec::try_recv_string_any(src, mpi_codec::TAG_DATA, payload, MPI_COMM_WORLD)) {
      continue;
    }

    Json msg = Json::parse(payload);
    bool ok = msg.value("ok", false);

    if (ok) discovery_by_rank[src] = msg;
    else failure_by_rank[src] = msg.value("error", "unknown discovery error");
  }

  // Mark any missing ranks as timeouts.
  for (int r = 1; r < world_size_; ++r) {
    if (!discovery_by_rank.count(r) && !failure_by_rank.count(r)) {
      failure_by_rank[r] = "discovery_timeout";
    }
  }

  // ---- Phase 2: Execute ----
  std::unordered_map<int, std::string> schema_by_rank;
  for (const auto& kv : discovery_by_rank) {
    int rank = kv.first;
    schema_by_rank[rank] = kv.second.value("schema_type", "Unknown");
  }

  for (int r = 1; r < world_size_; ++r) {
    if (failure_by_rank.count(r)) continue; // skip failed discovery nodes

    Json exec_msg;
    exec_msg["type"] = "execute";
    exec_msg["schema_type"] = schema_by_rank[r];
    mpi_codec::send_string(r, mpi_codec::TAG_CONTROL, exec_msg.dump(), MPI_COMM_WORLD);
  }

  const int64_t t1 = now_ms();
  std::unordered_map<int, Json> results_by_rank;

  while (static_cast<int>(results_by_rank.size()) < static_cast<int>(discovery_by_rank.size())) {
    if ((now_ms() - t1) > cfg_.fault.worker_query_timeout_sec * 1000LL) {
      break;
    }

    int src = -1;
    std::string payload;
    if (!mpi_codec::try_recv_string_any(src, mpi_codec::TAG_DATA, payload, MPI_COMM_WORLD)) {
      continue;
    }

    Json msg = Json::parse(payload);
    bool ok = msg.value("ok", false);
    if (ok) results_by_rank[src] = msg;
    else failure_by_rank[src] = msg.value("error", "unknown execution error");
  }

  // Mark missing execution results as timeouts for those that passed discovery.
  for (const auto& kv : discovery_by_rank) {
    int r = kv.first;
    if (!results_by_rank.count(r) && !failure_by_rank.count(r)) {
      failure_by_rank[r] = "execution_timeout";
    }
  }

  // ---- Phase 3: Report ----
  Json report = report_writer::build(cfg_, discovery_by_rank, results_by_rank, failure_by_rank);
  report_writer::write_to_disk(cfg_, report);

  // ---- Shutdown workers ----
  Json shutdown;
  shutdown["type"] = "shutdown";
  for (int r = 1; r < world_size_; ++r) {
    mpi_codec::send_string(r, mpi_codec::TAG_CONTROL, shutdown.dump(), MPI_COMM_WORLD);
  }

  log_.info("coordinator finished");
}