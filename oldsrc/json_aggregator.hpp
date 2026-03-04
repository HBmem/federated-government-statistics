#pragma once
#include <string>
#include <vector>
#include <map>
#include "types.hpp"
#include "aggregate.hpp"

// Structure to hold aggregated data with worker source information
struct WorkerNodeData {
    int rank;
    std::string source_description;
    std::string county_fips;
    WorkerStatus status;
    std::vector<AggregateRow> aggregates;
};

struct JsonAggregationResult {
    std::string start_date;
    std::string end_date;
    std::vector<WorkerNodeData> worker_data;  // Individual worker data marked with source
    std::map<AggregateKey, AggregateRow> aggregated_data;  // Merged aggregates by key
};

// Aggregate data from all workers and produce JSON
std::string createJsonReport(
    const std::string &start_date,
    const std::string &end_date,
    const std::vector<WorkerNodeData> &worker_data_list
);

// Helper function to aggregate AggregateRows
AggregateRow aggregateRows(const std::vector<AggregateRow> &rows);

// Helper to compare AggregateKey
inline bool operator<(const AggregateKey &a, const AggregateKey &b) {
    if (a.county_fips != b.county_fips) return a.county_fips < b.county_fips;
    if (a.window_start != b.window_start) return a.window_start < b.window_start;
    return a.window_end < b.window_end;
}
