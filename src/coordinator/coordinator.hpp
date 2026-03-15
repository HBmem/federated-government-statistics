#pragma once

#include "../common/config.hpp"
#include "../common/logging.hpp"

class Coordinator {
public:
  Coordinator(Logger& log, const AppConfig& cfg, int world_size)
      : log_(log), cfg_(cfg), world_size_(world_size) {}

  void run();

private:
  Logger& log_;
  const AppConfig& cfg_;
  int world_size_;
};