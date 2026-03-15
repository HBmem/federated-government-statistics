#pragma once

#include "../common/config.hpp"
#include "../common/logging.hpp"

class Worker {
public:
  Worker(Logger& log, const AppConfig& cfg, int rank)
      : log_(log), cfg_(cfg), rank_(rank) {}

  void run();

private:
  Logger& log_;
  const AppConfig& cfg_;
  int rank_;
};