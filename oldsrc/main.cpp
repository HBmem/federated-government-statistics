#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "mpi_util.hpp"
#include "contract.hpp"
#include "adapters/adapter.hpp"
#include "serialize.hpp"
#include "json_aggregator.hpp"

AdapterResult runWorkerContract(const std::string &conninfo, const ContractRequest &req, const std::string &county_fips);

static std::string trim(const std::string &s) {
    const auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static std::string extractCountyFips(const std::string &conninfo) {
    // Extract county_fips from conninfo string
    // Format: "host=... port=... ... county_fips=53033"
    const std::string prefix = "county_fips=";
    size_t pos = conninfo.find(prefix);
    if (pos == std::string::npos) {
        throw std::runtime_error("county_fips not found in conninfo: " + conninfo);
    }
    
    size_t start = pos + prefix.length();
    size_t end = conninfo.find_first_of(" \t\n", start);
    
    if (end == std::string::npos) {
        // county_fips is at the end of the string
        return conninfo.substr(start);
    }
    
    return conninfo.substr(start, end - start);
}

static std::string stripCountyFips(const std::string &conninfo) {
    // Remove county_fips parameter from conninfo for PostgreSQL connection
    // This keeps only the valid PostgreSQL connection options
    const std::string prefix = "county_fips=";
    size_t pos = conninfo.find(prefix);
    if (pos == std::string::npos) {
        return conninfo;
    }
    
    // Find the end of the county_fips value (space or end of string)
    size_t end = conninfo.find_first_of(" \t\n", pos);
    
    if (end == std::string::npos) {
        // county_fips is at the end, just remove it and any trailing space
        if (pos > 0 && conninfo[pos - 1] == ' ') {
            return conninfo.substr(0, pos - 1);
        }
        return conninfo.substr(0, pos);
    }
    
    // Remove county_fips=value and the space before it (if present)
    std::string result = conninfo.substr(0, pos);
    if (result.length() > 0 && result.back() == ' ') {
        result.pop_back();
    }
    result += conninfo.substr(end);
    return trim(result);
}

static std::vector<std::string> readNodes(const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open nodes file: " + path);
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty()) {     // skip empty lines
            continue;
        }
        if (line[0] == '#') {   // skip comment lines
            continue;
        }
        lines.push_back(line);  // trim and store valid lines
    }
    return lines;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string nodes_path = "../config/nodes.txt";
    if (argc >= 2) {
        nodes_path = argv[1];
    }

    auto nodes = readNodes(nodes_path);
    if ((int)nodes.size() != size) {
        if (rank == 0) {
            std::cerr << "Number of nodes in " << nodes_path << " (" << nodes.size() << ") does not match MPI size (" << size << ")\n";
        }
        MPI_Finalize();
        return 1;
    }

    // Coordinator creates contract request and broadcasts to workers
    ContractRequest contract_req;
    if (rank == 0) {
        contract_req.start_date = "2026-01-01";
        contract_req.end_date = "2027-01-01";
    }

    std::string req_wire;
    if (rank == 0) {
        req_wire = contract_req.start_date + "|" + contract_req.end_date;
    }
    req_wire = bcastString(0, MPI_COMM_WORLD, req_wire);

    // Workers parse contract request
    if (rank != 0) {
        auto bar = req_wire.find('|');
        contract_req.start_date = req_wire.substr(0, bar);
        contract_req.end_date = req_wire.substr(bar + 1);
    }

    // Each worker runs contract and gets result; Coordinator sends empty payload
    std::string payload;
    if (rank == 0) {
        WorkerStatus status;
        status.ok = true;
        status.message = "coordinator";
        payload = serializeWorkerPayload(status, {});
    } else {
        try {
            std::string county_fips = extractCountyFips(nodes[rank]);
            std::string clean_conninfo = stripCountyFips(nodes[rank]);
            auto result = runWorkerContract(clean_conninfo, contract_req, county_fips);
            payload = serializeWorkerPayload(result.status, result.aggregateRows);
        } catch (const std::exception &e) {
            WorkerStatus status;
            status.ok = false;
            status.message = e.what();
            payload = serializeWorkerPayload(status, {});
        }
    }

    auto payloads = gatherStringsToRoot(0, MPI_COMM_WORLD, payload);

    if (rank != 0) {
        std::cerr << "Rank " << rank << " conninfo: '" << nodes[rank] << "' payload_len=" << payload.size() << "\n";
    }

    if (rank == 0) {
        std::vector<WorkerNodeData> worker_data_list;

        // Parse payloads from all workers
        for (int r = 1; r < size; r++) {
            WorkerNodeData worker_data;
            worker_data.rank = r;
            worker_data.county_fips = extractCountyFips(nodes[r]);
            worker_data.source_description = "Worker rank " + std::to_string(r) + " processing county " + worker_data.county_fips;

            try {
                parseWorkerPayload(payloads[r], worker_data.status, worker_data.aggregates);
            } catch (const std::exception &e) {
                std::cerr << "Failed to parse payload from worker " << r << ": " << e.what() << "\n";
                worker_data.status.ok = false;
                worker_data.status.message = std::string("Parse error: ") + e.what();
                worker_data.aggregates.clear();
            }

            worker_data_list.push_back(worker_data);

            // Print worker summary
            std::cout << "Worker rank " << r
                      << " status: " << (worker_data.status.ok ? "OK" : "ERROR")
                      << ", message: " << worker_data.status.message
                      << ", rows_scanned: " << worker_data.status.rows_scanned
                      << ", rows_used: " << worker_data.status.rows_used
                      << ", rows_dropped: " << worker_data.status.rows_dropped
                      << ", elapsed_ms: " << worker_data.status.elapsed_ms
                      << "\n";
        }

        // Generate JSON report
        std::string json_report = createJsonReport(contract_req.start_date, contract_req.end_date, worker_data_list);
        
        // Output JSON to stdout
        // std::cout << "\n=== JSON AGGREGATION REPORT ===\n";
        // std::cout << json_report << "\n";
        
        // Also save to file
        std::ofstream json_file("aggregation_report.json");
        if (json_file) {
            json_file << json_report;
            json_file.close();
            std::cerr << "JSON report saved to aggregation_report.json\n";
        }
    }

    MPI_Finalize();
    return 0;
}