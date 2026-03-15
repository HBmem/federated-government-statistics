cmake_minimum_required(VERSION 3.16)
project(federated_mpi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Dependencies ---
find_package(MPI REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBPQ REQUIRED libpq)

# Prefer CMake package for nlohmann_json if present.
find_package(nlohmann_json 3.2.0 QUIET)

# --- Target ---
add_executable(federated
  src/main.cpp

  src/common/config.cpp
  src/common/mpi_codec.cpp

  src/db/pg_client.cpp
  src/db/sql_template.cpp

  src/worker/discovery.cpp
  src/worker/executor.cpp
  src/worker/worker.cpp

  src/coordinator/coordinator.cpp
  src/coordinator/report_writer.cpp
)

target_include_directories(federated PRIVATE
  ${MPI_CXX_INCLUDE_DIRS}
  ${LIBPQ_INCLUDE_DIRS}

  # Project includes (lets you do: #include "common/config.hpp")
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# If nlohmann_json is not found as a CMake package, fall back to a vendored header.
# Place header at: ThirdParty/nlohmann/json.hpp
if(NOT nlohmann_json_FOUND)
  target_include_directories(federated PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty
  )
endif()

target_link_libraries(federated PRIVATE
  MPI::MPI_CXX
  ${LIBPQ_LIBRARIES}
)

# Link nlohmann_json if found (header-only, but this provides include dirs and target)
if(nlohmann_json_FOUND)
  target_link_libraries(federated PRIVATE nlohmann_json::nlohmann_json)
endif()

# libpq compile flags (e.g. -pthread) from pkg-config
target_compile_options(federated PRIVATE ${LIBPQ_CFLAGS_OTHER})

# Helpful warnings (safe defaults)
if (MSVC)
  target_compile_options(federated PRIVATE /W4)
else()
  target_compile_options(federated PRIVATE -Wall -Wextra -Wpedantic)
endif()

# If MPI requires special link flags on some systems
if(MPI_CXX_LINK_FLAGS)
  set_target_properties(federated PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
endif()