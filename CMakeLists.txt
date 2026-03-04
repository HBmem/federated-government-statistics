cmake_minimum_required(VERSION 3.16)
project(federated_mpi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBPQ REQUIRED libpq)
find_package(nlohmann_json 3.2.0 QUIET)

add_executable(federated
  src/main.cpp
  src/db.cpp
  src/schema_detect.cpp
  src/contract.cpp
  src/aggregate.cpp
  src/serialize.cpp
  src/json_aggregator.cpp
  src/adapters/schema_a.cpp
  src/adapters/schema_b.cpp
  src/adapters/schema_c.cpp
)

target_include_directories(federated PRIVATE
  ${MPI_CXX_INCLUDE_DIRS}
  ${LIBPQ_INCLUDE_DIRS}
  src
)

target_link_libraries(federated PRIVATE
  MPI::MPI_CXX
  ${LIBPQ_LIBRARIES}
)

# Link nlohmann_json if found, otherwise rely on header-only mode
if(nlohmann_json_FOUND)
  target_link_libraries(federated PRIVATE nlohmann_json::nlohmann_json)
endif()

target_compile_options(federated PRIVATE ${LIBPQ_CFLAGS_OTHER})