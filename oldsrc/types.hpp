#pragma once
#include <string>
#include <cstdint>

struct ContractRequest {
    std::string start_date;
    std::string end_date;
};

struct WorkerStatus {
    bool ok = true;
    std::string message;
    int64_t rows_scanned = 0;
    int64_t rows_used = 0;
    int64_t rows_dropped = 0;
    int64_t elapsed_ms = 0;
};