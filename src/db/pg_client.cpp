#include "pg_client.hpp"

#include <sstream>

PgClient::~PgClient() {
  disconnect();
}

void PgClient::connect(const std::string& conninfo) {
  disconnect();
  conn_ = PQconnectdb(conninfo.c_str());
  if (!conn_ || PQstatus(conn_) != CONNECTION_OK) {
    std::string err = conn_ ? PQerrorMessage(conn_) : "PQconnectdb returned null connection";
    disconnect();
    throw std::runtime_error("Postgres connect failed: " + err);
  }
}

void PgClient::disconnect() {
  if (conn_) {
    PQfinish(conn_);
    conn_ = nullptr;
  }
}

PGresult* PgClient::exec(const std::string& sql) {
  if (!conn_) throw std::runtime_error("PgClient::exec called without connection");
  PGresult* r = PQexec(conn_, sql.c_str());
  require_ok(r, sql);
  return r;
}

void PgClient::exec_ok(const std::string& sql) {
  PGresult* r = exec(sql);
  PQclear(r);
}

void PgClient::require_ok(PGresult* r, const std::string& sql) {
  if (!r) throw std::runtime_error("Postgres returned null PGresult for SQL: " + sql);

  auto st = PQresultStatus(r);
  if (!(st == PGRES_TUPLES_OK || st == PGRES_COMMAND_OK)) {
    std::string err = PQerrorMessage(conn_);
    PQclear(r);
    throw std::runtime_error("Postgres exec failed: " + err + " SQL: " + sql);
  }
}

std::string PgClient::conninfo(
    const std::string& host,
    int port,
    const std::string& dbname,
    const std::string& user,
    const std::string& password) {
  std::ostringstream ss;
  ss << "host=" << host << " port=" << port
     << " dbname=" << dbname
     << " user=" << user
     << " password=" << password;
  return ss.str();
}