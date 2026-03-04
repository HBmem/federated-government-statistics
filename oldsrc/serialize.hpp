#pragma once
#include <string>
#include <vector>
#include "types.hpp"
#include "aggregate.hpp"

std::string serializeWorkerPayload(
    const WorkerStatus &status_in,
    const std::vector<AggregateRow> &aggregates_in
);

void parseWorkerPayload(
    const std::string &payload,
    WorkerStatus &status_out,
    std::vector<AggregateRow> &aggregates_out
);