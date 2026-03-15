#include "worker.hpp"

#include <mpi.h>

#include "../common/mpi_codec.hpp"
#include "../db/pg_client.hpp"
#include "discovery.hpp"
#include "executor.hpp"

/*
  Worker loop:
  - Wait for coordinator control message (JSON)
  - Handle "discover" -> send schema descriptor
  - Handle "execute"  -> run metric templates and send results
*/
void Worker::run() {
  const NodeInfo& node = cfg_.node_for_rank(rank_);
  if (node.is_coordinator) return;

  PgClient db;
  db.connect(PgClient::conninfo(node.host, node.port, node.dbname, node.user, node.password));

  log_.info("worker rank=" + std::to_string(rank_) + " connected to db on port=" + std::to_string(node.port));

  while (true) {
    // Coordinator is rank 0.
    std::string payload = mpi_codec::recv_string(0, mpi_codec::TAG_CONTROL, MPI_COMM_WORLD);
    Json msg = Json::parse(payload);

    const std::string type = msg.value("type", "");
    if (type == "shutdown") {
      log_.info("worker rank=" + std::to_string(rank_) + " shutting down");
      break;
    }

    if (type == "discover") {
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

      mpi_codec::send_string(0, mpi_codec::TAG_DATA, desc.dump(), MPI_COMM_WORLD);
      continue;
    }

    if (type == "execute") {
      const std::string schema_type = msg.value("schema_type", "Unknown");

      Json result;
      try {
        result = executor::run(log_, cfg_, node, db, schema_type);
        result["rank"] = rank_;
      } catch (const std::exception& ex) {
        result = Json::object();
        result["ok"] = false;
        result["rank"] = rank_;
        result["county_fips"] = node.county_fips;
        result["schema_type"] = schema_type;
        result["error"] = ex.what();
      }

      mpi_codec::send_string(0, mpi_codec::TAG_DATA, result.dump(), MPI_COMM_WORLD);
      continue;
    }

    // Unknown message type: return a failure response for auditing.
    Json unknown;
    unknown["ok"] = false;
    unknown["rank"] = rank_;
    unknown["county_fips"] = node.county_fips;
    unknown["error"] = "Unknown message type: " + type;

    mpi_codec::send_string(0, mpi_codec::TAG_DATA, unknown.dump(), MPI_COMM_WORLD);
  }

  db.disconnect();
}