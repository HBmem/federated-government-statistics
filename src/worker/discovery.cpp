#include "discovery.hpp"

#include <unordered_set>

static bool table_exists(PgClient& db, const std::string& table) {
  const std::string sql =
      "SELECT 1 FROM information_schema.tables "
      "WHERE table_schema='public' AND table_name='" + table + "' LIMIT 1;";
  PGresult* r = db.exec(sql);
  bool exists = (PQntuples(r) > 0);
  PQclear(r);
  return exists;
}

namespace discovery {

Json run(PgClient& db) {
  Json out;
  out["ok"] = true;

  bool has_resident = table_exists(db, "resident");
  bool has_household = table_exists(db, "household");
  bool has_address = table_exists(db, "address");
  bool has_person = table_exists(db, "person_record");
  bool has_citizen = table_exists(db, "citizen");

  std::string schema = "Unknown";
  if (has_resident && has_household && has_address) schema = "A";
  else if (has_person) schema = "B";
  else if (has_citizen) schema = "C";

  out["schema_type"] = schema;

  // Pull column details for tables that exist (helps debugging and audit).
  Json tables = Json::array();
  const std::vector<std::string> check_tables = {"resident", "household", "address", "person_record", "citizen"};

  for (const auto& t : check_tables) {
    if (!table_exists(db, t)) continue;

    Json tinfo;
    tinfo["name"] = t;
    tinfo["columns"] = Json::array();

    std::string sql =
        "SELECT column_name, data_type "
        "FROM information_schema.columns "
        "WHERE table_schema='public' AND table_name='" + t + "' "
        "ORDER BY ordinal_position;";

    PGresult* r = db.exec(sql);
    const int n = PQntuples(r);
    for (int i = 0; i < n; ++i) {
      Json c;
      c["name"] = PQgetvalue(r, i, 0);
      c["type"] = PQgetvalue(r, i, 1);
      tinfo["columns"].push_back(c);
    }
    PQclear(r);

    tables.push_back(tinfo);
  }

  out["tables"] = tables;
  return out;
}

} // namespace discovery