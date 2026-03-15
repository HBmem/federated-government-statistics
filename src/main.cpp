#include <mpi.h>

#include "common/logging.hpp"
#include "common/time.hpp"
#include "common/config.hpp"
#include "coordinator/coordinator.hpp"
#include "worker/worker.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

/*
  Entry point:
  - rank 0 runs coordinator
  - ranks 1..N-1 run worker
*/
int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int world = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  try {
    // Parse CLI once per rank (simple and avoids MPI broadcasts for config paths).
    AppConfig cfg = AppConfig::load_from_cli(argc, argv);

    // Initialize logger (file logging is optional; this uses stderr by default).
    Logger log(cfg.logging.level);

    if (rank == 0) {
      Coordinator coord(log, cfg, world);
      coord.run();
    } else {
      Worker worker(log, cfg, rank);
      worker.run();
    }
  } catch (const std::exception& ex) {
    std::cerr << "[fatal] " << ex.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}