#include "./metrics.hpp"

#include <chrono>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <unistd.h>

namespace {

/*
  Return monotonic wall-clock time in milliseconds.
*/
int64_t monotonic_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

/*
  Read VmRSS from /proc/self/status.
  Returns current resident set size in KB on Linux.
*/
int64_t read_rss_kb_from_proc_status() {
  std::ifstream in("/proc/self/status");
  if (!in) return 0;

  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      std::istringstream iss(line);
      std::string key;
      int64_t kb = 0;
      std::string unit;
      iss >> key >> kb >> unit;
      return kb;
    }
  }
  return 0;
}

/*
  Convert timeval to milliseconds.
*/
int64_t timeval_to_ms(const timeval& tv) {
  return static_cast<int64_t>(tv.tv_sec) * 1000LL +
         static_cast<int64_t>(tv.tv_usec) / 1000LL;
}

}  // namespace

namespace perf {

Stopwatch::Stopwatch() : start_ms_(monotonic_ms()) {}

void Stopwatch::reset() {
  start_ms_ = monotonic_ms();
}

int64_t Stopwatch::elapsed_ms() const {
  return monotonic_ms() - start_ms_;
}

ProcessSnapshot capture_process_snapshot() {
  ProcessSnapshot s;

  rusage ru{};
  if (getrusage(RUSAGE_SELF, &ru) == 0) {
    s.cpu_user_ms = timeval_to_ms(ru.ru_utime);
    s.cpu_system_ms = timeval_to_ms(ru.ru_stime);

    /*
      On Linux, ru_maxrss is in KB.
      On some other Unix systems it may differ, but this project targets Linux containers.
    */
    s.max_rss_kb = static_cast<int64_t>(ru.ru_maxrss);
  }

  s.rss_kb = read_rss_kb_from_proc_status();
  return s;
}

Json build_process_metrics_json(const ProcessSnapshot& start, const ProcessSnapshot& end) {
  Json j;
  j["cpu_user_ms"] = end.cpu_user_ms - start.cpu_user_ms;
  j["cpu_system_ms"] = end.cpu_system_ms - start.cpu_system_ms;
  j["rss_kb_at_start"] = start.rss_kb;
  j["rss_kb_at_end"] = end.rss_kb;
  j["max_rss_kb"] = end.max_rss_kb;
  return j;
}

}  // namespace perf