#pragma once

#include <iostream>
#include <string>

/*
  Lightweight logger.

  Logs to stderr. You can later extend this to file logging or structured JSON logs.
*/
class Logger {
public:
  explicit Logger(const std::string& level) : level_(level) {}

  void info(const std::string& msg) const { log("info", msg); }
  void warn(const std::string& msg) const { log("warn", msg); }
  void error(const std::string& msg) const { log("error", msg); }
  void debug(const std::string& msg) const { if (level_ == "debug" || level_ == "trace") log("debug", msg); }
  void trace(const std::string& msg) const { if (level_ == "trace") log("trace", msg); }

private:
  std::string level_;

  void log(const std::string& lvl, const std::string& msg) const {
    std::cerr << "[" << lvl << "] " << msg << "\n";
  }
};