#pragma once

#include <string>
#include <cctype>

inline std::string ltrim(std::string s) {
  size_t i = 0;
  while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
  return s.substr(i);
}

inline std::string rtrim(std::string s) {
  if (s.empty()) return s;
  size_t i = s.size();
  while (i > 0 && std::isspace(static_cast<unsigned char>(s[i - 1]))) --i;
  return s.substr(0, i);
}

inline std::string trim(std::string s) {
  return rtrim(ltrim(std::move(s)));
}