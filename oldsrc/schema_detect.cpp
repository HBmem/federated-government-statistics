#include "schema_detect.hpp"
#include <string>

static bool tableExists(PgDb &db, const std::string &name) {
    const char *params[1] = { name.c_str() };
    PGresult *res = db.execParams(
        "Select to_regclass($1) IS NOT NULL;",
        1, params
    );

    bool ok = false;
    if (res && PQresultStatus(res) == PGRES_TUPLES_OK && PQntuples(res) == 1 && PQnfields(res) == 1) {
        ok = PQgetvalue(res, 0, 0)[0] == 't';
    }
    PQclear(res);
    return ok;
};

static bool columnExists(PgDb &db, const std::string &table, const std::string &column) {
    // Query information_schema.columns for a robust column existence check.
    const char *params[2] = { table.c_str(), column.c_str() };
    PGresult *res = nullptr;
    try {
        // Ensure the column exists on the table resolved in the current search_path
        res = db.execParams(
            "SELECT EXISTS("
            "SELECT 1 FROM pg_catalog.pg_attribute a"
            " JOIN pg_catalog.pg_class c ON a.attrelid = c.oid"
            " JOIN pg_catalog.pg_namespace n ON c.relnamespace = n.oid"
            " WHERE c.relname = $1 AND a.attname = $2 AND a.attnum > 0 AND NOT a.attisdropped"
            " AND n.nspname = current_schema() );",
            2, params
        );
    } catch (...) {
        if (res) PQclear(res);
        return false;
    }

    bool ok = false;
    if (res && PQresultStatus(res) == PGRES_TUPLES_OK && PQntuples(res) == 1) {
        ok = PQgetvalue(res, 0, 0)[0] == 't';
    }
    PQclear(res);
    return ok;
}

SchemaType detectSchema(PgDb &db) {
    if (tableExists(db, "household") && tableExists(db, "address") && tableExists(db, "resident")) {
        return SchemaType::A;
    } else if (tableExists(db, "person_record")) {
        return SchemaType::B;
    } else if (tableExists(db, "citizen")) {
        // Some DBs may have a citizen table but not the expected JSON column; verify it
        if (columnExists(db, "citizen", "profile_json")) {
            return SchemaType::C;
        }
        return SchemaType::Unknown;
    } else {
        return SchemaType::Unknown;
    }
}