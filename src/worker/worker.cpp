#include "./worker.hpp"

#include <mpi.h>
#include <future>
#include <thread>
#include <chrono>
#include <cstdlib>

#include "../common/metrics.hpp"
#include "../common/mpi_codec.hpp"
#include "../db/pg_client.hpp"
#include "./discovery.hpp"
#include "./executor.hpp"

/*
  Worker loop:
  - Wait for coordinator control message
  - Capture timing and process metrics for each request
  - Return run-level metrics with discovery/execution output
*/
void Worker::run() {
  const NodeInfo& node = cfg_.node_for_rank(rank_);
  if (node.is_coordinator) return;

  PgClient db;

  perf::Stopwatch connect_timer;
  db.connect(PgClient::conninfo(node.host, node.port, node.dbname, node.user, node.password));
  int64_t db_connect_ms = connect_timer.elapsed_ms();

  log_.info("worker rank=" + std::to_string(rank_) +
            " connected to db on port=" + std::to_string(node.port));

  while (true) {
    std::string payload = mpi_codec::recv_string(0, mpi_codec::TAG_CONTROL, MPI_COMM_WORLD);
    Json msg = Json::parse(payload);

    const std::string type = msg.value("type", "");
    if (type == "shutdown") {
      log_.info("worker rank=" + std::to_string(rank_) + " shutting down");
      break;
    }

    perf::Stopwatch worker_total_timer;
    perf::ProcessSnapshot process_start;
    if (cfg_.performance.capture_process_metrics) {
      process_start = perf::capture_process_snapshot();
    }

    if (type == "discover") {
      perf::Stopwatch discovery_timer;
      Json desc;

      try {
        desc = discovery::run(db);
        desc["rank"] = rank_;
        desc["county_fips"] = node.county_fips;
      } catch (const std::exception& ex) {
        desc = Json::object();
        desc["ok"] = false;
        desc["rank"] = rank_;
        desc["county_fips"] = node.county_fips;
        desc["error"] = ex.what();
      }

      int64_t discovery_ms = discovery_timer.elapsed_ms();

      Json worker_metrics;
      worker_metrics["worker_total_ms"] = worker_total_timer.elapsed_ms();
      worker_metrics["db_connect_ms"] = db_connect_ms;
      worker_metrics["discovery_ms"] = discovery_ms;
      worker_metrics["execution_ms"] = 0;
      worker_metrics["result_serialize_ms"] = 0;
      worker_metrics["result_send_ms"] = 0;

      if (cfg_.performance.capture_process_metrics) {
        perf::ProcessSnapshot process_end = perf::capture_process_snapshot();
        Json proc = perf::build_process_metrics_json(process_start, process_end);
        for (auto it = proc.begin(); it != proc.end(); ++it) {
          worker_metrics[it.key()] = it.value();
        }
      }

      desc["worker_metrics"] = worker_metrics;

      perf::Stopwatch serialize_timer;
      std::string out_payload = desc.dump();
      int64_t serialize_ms = serialize_timer.elapsed_ms();
      desc["worker_metrics"]["result_serialize_ms"] = serialize_ms;

      perf::Stopwatch send_timer;
      mpi_codec::send_string(0, mpi_codec::TAG_DATA, desc.dump(), MPI_COMM_WORLD);
      int64_t send_ms = send_timer.elapsed_ms();

      // The send timing cannot be inserted into the already-sent payload after send.
      // To preserve accurate numbers, include send timing before serialization for future phases.
      // Discovery keeps send timing as 0 in the returned payload.
      (void)send_ms;
      continue;
    }

    if (type == "execute") {
      const std::string schema_type = msg.value("schema_type", "Unknown");

      perf::Stopwatch execution_timer;
      Json result;

      std::future<Json> fut = std::async(std::launch::async, [&]() -> Json {
        Json res;
        try {
          res = executor::run(log_, cfg_, node, db, schema_type);
          res["rank"] = rank_;
        } catch (const std::exception& ex) {
          res = Json::object();
          res["ok"] = false;
          res["rank"] = rank_;
          res["county_fips"] = node.county_fips;
          res["schema_type"] = schema_type;
          res["error"] = ex.what();
        }
        return res;
      });

      auto start_time = std::chrono::steady_clock::now();
      while (fut.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout) {
        int src_rank;
        std::string dummy;
        if (mpi_codec::try_recv_string_any(src_rank, mpi_codec::TAG_CONTROL, dummy, MPI_COMM_WORLD) && src_rank == 0) {
          Json msg = Json::parse(dummy);
          if (msg.value("type", "") == "shutdown") {
            db.cancel();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            result = Json::object();
            result["ok"] = false;
            result["rank"] = rank_;
            result["county_fips"] = node.county_fips;
            result["schema_type"] = schema_type;
            result["error"] = "Execution cancelled due to shutdown";
            goto after_execute;
          }
        }
        auto now = std::chrono::steady_clock::now();
        int timeout_sec = cfg_.fault.worker_query_timeout_sec;
        if (std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count() > timeout_sec) {
          log_.error("TIMEOUT: Query execution exceeded " + std::to_string(timeout_sec) + " seconds on rank=" + std::to_string(rank_) +
                     " county_fips=" + node.county_fips + ". Cancelling query and exiting program.");
          db.cancel();
          std::this_thread::sleep_for(std::chrono::milliseconds(500));
          result = Json::object();
          result["ok"] = false;
          result["rank"] = rank_;
          result["county_fips"] = node.county_fips;
          result["schema_type"] = schema_type;
          result["error"] = "Worker execution timeout - query cancelled";
          
          // Log the timeout status and exit the program
          log_.error("Worker rank=" + std::to_string(rank_) + " experienced execution timeout. Exiting.");
          mpi_codec::send_string(0, mpi_codec::TAG_DATA, result.dump(), MPI_COMM_WORLD);
          std::exit(EXIT_FAILURE);
        }
      }
      result = fut.get();

      after_execute:

      int64_t execution_ms = execution_timer.elapsed_ms();

      Json worker_metrics;
      worker_metrics["worker_total_ms"] = 0;
      worker_metrics["db_connect_ms"] = db_connect_ms;
      worker_metrics["discovery_ms"] = 0;
      worker_metrics["execution_ms"] = execution_ms;
      worker_metrics["result_serialize_ms"] = 0;
      worker_metrics["result_send_ms"] = 0;

      if (cfg_.performance.capture_process_metrics) {
        perf::ProcessSnapshot process_end = perf::capture_process_snapshot();
        Json proc = perf::build_process_metrics_json(process_start, process_end);
        for (auto it = proc.begin(); it != proc.end(); ++it) {
          worker_metrics[it.key()] = it.value();
        }
      }

      result["worker_metrics"] = worker_metrics;

      perf::Stopwatch serialize_timer;
      std::string out_payload = result.dump();
      int64_t serialize_ms = serialize_timer.elapsed_ms();

      result["worker_metrics"]["result_serialize_ms"] = serialize_ms;
      result["worker_metrics"]["worker_total_ms"] = worker_total_timer.elapsed_ms();

      std::string final_payload = result.dump();

      perf::Stopwatch send_timer;
      mpi_codec::send_string(0, mpi_codec::TAG_DATA, result.dump(), MPI_COMM_WORLD);
      int64_t send_ms = send_timer.elapsed_ms();

      (void)send_ms;
      continue;
    }

    Json unknown;
    unknown["ok"] = false;
    unknown["rank"] = rank_;
    unknown["county_fips"] = node.county_fips;
    unknown["error"] = "Unknown message type: " + type;

    mpi_codec::send_string(0, mpi_codec::TAG_DATA, unknown.dump(), MPI_COMM_WORLD);
  }

  db.disconnect();
}