#pragma once

#include <libpq-fe.h>
#include <string>
#include <stdexcept>

class PgClient {
public:
  PgClient() = default;
  ~PgClient();

  PgClient(const PgClient&) = delete;
  PgClient& operator=(const PgClient&) = delete;

  void connect(const std::string& conninfo);
  void disconnect();

  bool is_connected() const { return conn_ != nullptr; }

  // Expose raw PGconn for safe literal escaping in sql_template substitution.
  PGconn* raw() const { return conn_; }

  PGresult* exec(const std::string& sql);
  void exec_ok(const std::string& sql);
  void cancel();

  static std::string conninfo(
      const std::string& host,
      int port,
      const std::string& dbname,
      const std::string& user,
      const std::string& password);

private:
  PGconn* conn_ = nullptr;
  void require_ok(PGresult* r, const std::string& sql);
};