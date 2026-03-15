#pragma once

#include <chrono>
#include <cstdint>

/*
  Time utilities used for coordinator timeouts.
*/
inline int64_t now_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}