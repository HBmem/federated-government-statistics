#pragma once
#include <string>
#include <libpq-fe.h>

class PgDb
{
private:
    PGconn *conn_ = nullptr;
public:
    explicit PgDb(const std::string &conninfo);
    ~PgDb();

    PgDb(const PgDb &) = delete;
    PgDb &operator=(const PgDb &) = delete;

    PGresult *exec(const std::string &sql);
    PGresult *execParams(const std::string &sql, int nParams, const char * const *param);

    PGconn *raw() {
        return conn_;
    }
};
