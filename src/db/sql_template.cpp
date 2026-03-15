#include "sql_template.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

static void replace_all(std::string& s, const std::string& from, const std::string& to) {
  if (from.empty()) return;
  size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.size(), to);
    pos += to.size();
  }
}

namespace sql_template {

std::string load_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Failed to open SQL template: " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string substitute(PGconn* conn, const std::string& sql,
                       const std::unordered_map<std::string, std::string>& params) {
  std::string out = sql;

  for (const auto& kv : params) {
    const std::string& key = kv.first;
    const std::string& val = kv.second;

    // Escape as SQL literal including surrounding quotes.
    char* lit = PQescapeLiteral(conn, val.c_str(), val.size());
    if (!lit) throw std::runtime_error("PQescapeLiteral failed");

    std::string placeholder = ":" + key;
    replace_all(out, placeholder, std::string(lit));

    PQfreemem(lit);
  }

  return out;
}

} // namespace sql_template