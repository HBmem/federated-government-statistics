#pragma once

#include "../common/json.hpp"
#include "../db/pg_client.hpp"

/*
  Worker discovery inspects the local database to determine:
  - schema_type: A / B / C / Unknown
  - list of tables
  - list of columns by table (name, data_type)

  The coordinator uses schema_type to select query templates.
*/
namespace discovery {

Json run(PgClient& db);

}