#include <iostream>
#include <string>
#include <libpq-fe.h>

static void exit_nicely(PGconn *conn, const std::string &message) {
    std::cerr << "ERROR: " << message << "\n";
    if (conn) {
        std::cerr << "PostgreSQL error message: " << PQerrorMessage(conn) << "\n";
        PQfinish(conn);
    }
    std::exit(1);
}

int main() {
    const char* conninfo = "host=127.0.0.1 port=5541 dbname=county user=federated password=federated";

    PGconn* conn = PQconnectdb(conninfo);
    if (PQstatus(conn) != CONNECTION_OK) {
        exit_nicely(conn, "Connection to database failed");
    }

    // Timeout behavior
    PGresult* r1 = PQexec(conn, "SET statement_timeout TO 5000"); // Set timeout to 5000 ms
    if (PQresultStatus(r1) != PGRES_COMMAND_OK) {
        exit_nicely(conn, "Failed to set statement timeout");
    }
    PQclear(r1);

    PGresult* res = PQexec(conn, "select * from household where income_usd> 270000;");
    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        exit_nicely(conn, "SELECT query failed");
    }

    int rows = PQntuples(res);
    int cols = PQnfields(res);

    for (int i = 0; i < cols; i++) {
        std::cout << PQfname(res, i) << (i + 1 == cols ? "\n" : "\t");
    }

    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            const char* value = PQgetvalue(res, i, j);
            std::cout << (value ? value : "NULL") << (j + 1 == cols ? "\n" : "\t");
        }
    }

    PQclear(res);
    PQfinish(conn);
    return 0;
    
}