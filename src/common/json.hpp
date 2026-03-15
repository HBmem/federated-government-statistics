#pragma once

/*
  JSON library wrapper.

  This code expects nlohmann::json.
  Add it as a dependency (single-header) in your project, e.g.:
    - vendor "nlohmann/json.hpp" in ThirdParty/
    - or install via package manager and include it in CMake

  If you already have a JSON library preference, swap this wrapper later.
*/

#include <nlohmann/json.hpp>

using Json = nlohmann::json;