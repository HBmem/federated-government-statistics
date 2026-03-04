#include "db.hpp"
#include <stdexcept>

PgDb::PgDb(const std::string &conninfo) {
    conn_ = PQconnectdb(conninfo.c_str());
    if (!conn_ || PQstatus(conn_) != CONNECTION_OK) {
        std::string msg = conn_ ? PQerrorMessage(conn_) : "null conn";
        if (conn_) PQfinish(conn_);
        throw std::runtime_error("DB connect failed: " + msg);
    }
    // Safety: timeouts so dead/slow nodes fail gracefully
    PGresult* r = exec("SET statement_timeout = '1min';");
    PQclear(r);
}

PgDb::~PgDb() {
    if (conn_) {
        PQfinish(conn_);
    }
}

PGresult *PgDb::exec(const std::string &sql) {
    PGresult *res = PQexec(conn_, sql.c_str());
    if (!res) {
        throw std::runtime_error("DB exec failed: null result");
    }
    auto status = PQresultStatus(res);
    if (status != PGRES_COMMAND_OK && status != PGRES_TUPLES_OK) {
        std::string msg = PQerrorMessage(conn_);
        PQclear(res);
        throw std::runtime_error("DB exec failed: " + msg);
    }
    return res;
}

PGresult *PgDb::execParams(const std::string &sql, int nParams, const char * const *params) {
    PGresult *res = PQexecParams(conn_, sql.c_str(), nParams, nullptr, params, nullptr, nullptr, 0);
    if (!res) {
        throw std::runtime_error("DB execParams failed: null result");
    }
    auto status = PQresultStatus(res);
    if (status != PGRES_COMMAND_OK && status != PGRES_TUPLES_OK) {
        std::string msg = PQerrorMessage(conn_);
        PQclear(res);
        throw std::runtime_error("DB execParams failed: " + msg);
    }
    return res;
}