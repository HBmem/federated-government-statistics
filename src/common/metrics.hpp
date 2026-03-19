#pragma once

#include <cstdint>
#include <string>

#include "./json.hpp"

/*
  Shared runtime metrics helpers.

  This module provides:
  - wall clock timers
  - process CPU usage snapshots
  - resident memory usage snapshots
  - max RSS information
*/

namespace perf {

/*
  Stopwatch for wall-clock timing in milliseconds.
*/
class Stopwatch {
public:
  Stopwatch();
  void reset();
  int64_t elapsed_ms() const;

private:
  int64_t start_ms_;
};

/*
  Snapshot of worker process resource usage.
*/
struct ProcessSnapshot {
  int64_t cpu_user_ms = 0;
  int64_t cpu_system_ms = 0;
  int64_t rss_kb = 0;
  int64_t max_rss_kb = 0;
};

/*
  Read the current process snapshot.
  Designed for Linux containers and Unix-like environments.
*/
ProcessSnapshot capture_process_snapshot();

/*
  Compute end - start deltas for CPU time and keep end-state memory values.
*/
Json build_process_metrics_json(const ProcessSnapshot& start, const ProcessSnapshot& end);

}  // namespace perf