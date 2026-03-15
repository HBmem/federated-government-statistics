#pragma once

#include <libpq-fe.h>
#include <string>
#include <unordered_map>

/*
  SQL template loader and parameter substitution.

  Templates use placeholders like:
    :start_date
    :end_date

  We replace placeholders with SQL literals safely using PQescapeLiteral.
*/
namespace sql_template {

std::string load_file(const std::string& path);

// Substitute :param occurrences with escaped SQL literals (quoted strings).
// Example: params["start_date"] = "2026-01-01" -> '2026-01-01'
std::string substitute(PGconn* conn, const std::string& sql,
                       const std::unordered_map<std::string, std::string>& params);

}