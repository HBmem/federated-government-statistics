#pragma once
#include <vector>
#include <string>

#include "../db.hpp"
#include "../types.hpp"
#include "../aggregate.hpp"

struct AdapterResult {
    WorkerStatus status;
    std::vector<AggregateRow> aggregateRows;
};

class Adapter {
public:
    virtual ~Adapter() = default;
    virtual AdapterResult run(PgDb &db, const ContractRequest &req, const std::string &county_fips) = 0;
};