#include "serialize.hpp"
#include <sstream>
#include <stdexcept>
#include <iostream>

static std::string esc(const std::string &s) {
    std::string t = s;
    for (auto &ch : t) {
        if (ch == '|') {
            ch = '/';
        }
        // Normalize newlines and carriage returns so payload remains single-line
        if (ch == '\n' || ch == '\r') {
            ch = ' ';
        }
    }
    return t;
}

std::string serializeWorkerPayload(
    const WorkerStatus &status_in,
    const std::vector<AggregateRow> &aggregates_in
) {
    std::ostringstream oss;
    oss << "STATUS|"
        << (status_in.ok ? "1" : "0") << "|"
        << status_in.elapsed_ms << "|"
        << status_in.rows_scanned << "|"
        << status_in.rows_used << "|"
        << status_in.rows_dropped << "|"
        << esc(status_in.message)
        << "\n";

    for (const auto &r : aggregates_in) {
        oss << "ROW|"
            << r.key.county_fips << "|"
            << r.key.window_start << "|"
            << r.key.window_end << "|"
            // Population
            << r.population_count << "|"
            << r.population_min << "|"
            << r.population_max << "|"
            << r.population_mean << "|"
            << r.population_median << "|"
            // Households
            << r.household_count << "|"
            << r.household_min << "|"
            << r.household_max << "|"
            << r.household_mean << "|"
            << r.household_median << "|"
            // Income
            << r.income_sum << "|"
            << r.income_count << "|"
            << r.income_min << "|"
            << r.income_max << "|"
            << r.income_mean << "|"
            << r.income_median << "|"
            // Income buckets (each: count, sum, min, max, mean, median)
            << r.income_bucket_under25k.count << "|"
            << r.income_bucket_under25k.sum << "|"
            << r.income_bucket_under25k.min << "|"
            << r.income_bucket_under25k.max << "|"
            << r.income_bucket_under25k.mean << "|"
            << r.income_bucket_under25k.median << "|"
            << r.income_bucket_25_50k.count << "|"
            << r.income_bucket_25_50k.sum << "|"
            << r.income_bucket_25_50k.min << "|"
            << r.income_bucket_25_50k.max << "|"
            << r.income_bucket_25_50k.mean << "|"
            << r.income_bucket_25_50k.median << "|"
            << r.income_bucket_50_75k.count << "|"
            << r.income_bucket_50_75k.sum << "|"
            << r.income_bucket_50_75k.min << "|"
            << r.income_bucket_50_75k.max << "|"
            << r.income_bucket_50_75k.mean << "|"
            << r.income_bucket_50_75k.median << "|"
            << r.income_bucket_75_100k.count << "|"
            << r.income_bucket_75_100k.sum << "|"
            << r.income_bucket_75_100k.min << "|"
            << r.income_bucket_75_100k.max << "|"
            << r.income_bucket_75_100k.mean << "|"
            << r.income_bucket_75_100k.median << "|"
            << r.income_bucket_100_150k.count << "|"
            << r.income_bucket_100_150k.sum << "|"
            << r.income_bucket_100_150k.min << "|"
            << r.income_bucket_100_150k.max << "|"
            << r.income_bucket_100_150k.mean << "|"
            << r.income_bucket_100_150k.median << "|"
            << r.income_bucket_150_200k.count << "|"
            << r.income_bucket_150_200k.sum << "|"
            << r.income_bucket_150_200k.min << "|"
            << r.income_bucket_150_200k.max << "|"
            << r.income_bucket_150_200k.mean << "|"
            << r.income_bucket_150_200k.median << "|"
            << r.income_bucket_200k_plus.count << "|"
            << r.income_bucket_200k_plus.sum << "|"
            << r.income_bucket_200k_plus.min << "|"
            << r.income_bucket_200k_plus.max << "|"
            << r.income_bucket_200k_plus.mean << "|"
            << r.income_bucket_200k_plus.median << "|"
            // Age
            << r.age_sum << "|"
            << r.age_count << "|"
            << r.age_min << "|"
            << r.age_max << "|"
            << r.age_mean << "|"
            << r.age_median << "|"
            // Age buckets (count, sum, min, max, mean, median)
            << r.age_bucket_0_18.count << "|"
            << r.age_bucket_0_18.sum << "|"
            << r.age_bucket_0_18.min << "|"
            << r.age_bucket_0_18.max << "|"
            << r.age_bucket_0_18.mean << "|"
            << r.age_bucket_0_18.median << "|"
            << r.age_bucket_19_35.count << "|"
            << r.age_bucket_19_35.sum << "|"
            << r.age_bucket_19_35.min << "|"
            << r.age_bucket_19_35.max << "|"
            << r.age_bucket_19_35.mean << "|"
            << r.age_bucket_19_35.median << "|"
            << r.age_bucket_36_50.count << "|"
            << r.age_bucket_36_50.sum << "|"
            << r.age_bucket_36_50.min << "|"
            << r.age_bucket_36_50.max << "|"
            << r.age_bucket_36_50.mean << "|"
            << r.age_bucket_36_50.median << "|"
            << r.age_bucket_51_65.count << "|"
            << r.age_bucket_51_65.sum << "|"
            << r.age_bucket_51_65.min << "|"
            << r.age_bucket_51_65.max << "|"
            << r.age_bucket_51_65.mean << "|"
            << r.age_bucket_51_65.median << "|"
            << r.age_bucket_66_plus.count << "|"
            << r.age_bucket_66_plus.sum << "|"
            << r.age_bucket_66_plus.min << "|"
            << r.age_bucket_66_plus.max << "|"
            << r.age_bucket_66_plus.mean << "|"
            << r.age_bucket_66_plus.median << "|"
            // Unemployment
            << r.employed_count << "|"
            << r.employed_min << "|"
            << r.employed_max << "|"
            << r.employed_mean << "|"
            << r.employed_median << "|"
            << r.unemployed_count << "|"
            << r.unemployed_min << "|"
            << r.unemployed_max << "|"
            << r.unemployed_mean << "|"
            << r.unemployed_median << "|"
            // Quality
            << r.rows_scanned << "|"
            << r.rows_used << "|"
            << r.rows_dropped
            << "\n";
    }
    
    return oss.str();
}

void parseWorkerPayload(
    const std::string &payload,
    WorkerStatus &status_out,
    std::vector<AggregateRow> &aggregates_out
) {
    std::istringstream iss(payload);
    std:: string line;

    aggregates_out.clear();
    bool have_status = false;

    while (std::getline(iss, line)) {
        if (line.empty()) {
            continue;
        }
        
        std::istringstream ls(line);
        std::string tok;
        std::vector<std::string> parts;
        while (std::getline(ls, tok, '|')) {
            parts.push_back(tok);
        }
        
        if (parts.size() == 0) {
            continue;
        }
        
        if (parts[0] == "STATUS") {
            if (parts.size() < 6) {
                throw std::runtime_error("BAD STATUS line");
            }
            status_out.ok = parts[1] == "1";
            status_out.elapsed_ms = std::stoll(parts[2]);
            status_out.rows_scanned = std::stoll(parts[3]);
            status_out.rows_used = std::stoll(parts[4]);
            status_out.rows_dropped = std::stoll(parts[5]);
            status_out.message = (parts.size() > 6) ? parts[6] : "";
            have_status = true;
        } else if (parts[0] == "ROW") {
            if (parts.size() < 111) {
                throw std::runtime_error("BAD ROW line (line " + std::to_string(__LINE__ - 2) + ") - expected at least 111 fields, got " + std::to_string(parts.size()));
            }
            
            AggregateRow r;
            int idx = 1;
            r.key.county_fips = parts[idx++];
            r.key.window_start = parts[idx++];
            r.key.window_end = parts[idx++];
            
            // Population (4 fields + count)
            r.population_count = std::stoll(parts[idx++]);
            r.population_min = std::stod(parts[idx++]);
            r.population_max = std::stod(parts[idx++]);
            r.population_mean = std::stod(parts[idx++]);
            r.population_median = std::stod(parts[idx++]);
            
            // Households
            r.household_count = std::stoll(parts[idx++]);
            r.household_min = std::stod(parts[idx++]);
            r.household_max = std::stod(parts[idx++]);
            r.household_mean = std::stod(parts[idx++]);
            r.household_median = std::stod(parts[idx++]);
            
            // Income
            r.income_sum = std::stod(parts[idx++]);
            r.income_count = std::stoll(parts[idx++]);
            r.income_min = std::stod(parts[idx++]);
            r.income_max = std::stod(parts[idx++]);
            r.income_mean = std::stod(parts[idx++]);
            r.income_median = std::stod(parts[idx++]);
            
            // Income buckets (7 buckets × 6 fields each)
            r.income_bucket_under25k.count = std::stoll(parts[idx++]);
            r.income_bucket_under25k.sum = std::stod(parts[idx++]);
            r.income_bucket_under25k.min = std::stod(parts[idx++]);
            r.income_bucket_under25k.max = std::stod(parts[idx++]);
            r.income_bucket_under25k.mean = std::stod(parts[idx++]);
            r.income_bucket_under25k.median = std::stod(parts[idx++]);
            
            r.income_bucket_25_50k.count = std::stoll(parts[idx++]);
            r.income_bucket_25_50k.sum = std::stod(parts[idx++]);
            r.income_bucket_25_50k.min = std::stod(parts[idx++]);
            r.income_bucket_25_50k.max = std::stod(parts[idx++]);
            r.income_bucket_25_50k.mean = std::stod(parts[idx++]);
            r.income_bucket_25_50k.median = std::stod(parts[idx++]);
            
            r.income_bucket_50_75k.count = std::stoll(parts[idx++]);
            r.income_bucket_50_75k.sum = std::stod(parts[idx++]);
            r.income_bucket_50_75k.min = std::stod(parts[idx++]);
            r.income_bucket_50_75k.max = std::stod(parts[idx++]);
            r.income_bucket_50_75k.mean = std::stod(parts[idx++]);
            r.income_bucket_50_75k.median = std::stod(parts[idx++]);
            
            r.income_bucket_75_100k.count = std::stoll(parts[idx++]);
            r.income_bucket_75_100k.sum = std::stod(parts[idx++]);
            r.income_bucket_75_100k.min = std::stod(parts[idx++]);
            r.income_bucket_75_100k.max = std::stod(parts[idx++]);
            r.income_bucket_75_100k.mean = std::stod(parts[idx++]);
            r.income_bucket_75_100k.median = std::stod(parts[idx++]);
            
            r.income_bucket_100_150k.count = std::stoll(parts[idx++]);
            r.income_bucket_100_150k.sum = std::stod(parts[idx++]);
            r.income_bucket_100_150k.min = std::stod(parts[idx++]);
            r.income_bucket_100_150k.max = std::stod(parts[idx++]);
            r.income_bucket_100_150k.mean = std::stod(parts[idx++]);
            r.income_bucket_100_150k.median = std::stod(parts[idx++]);
            
            r.income_bucket_150_200k.count = std::stoll(parts[idx++]);
            r.income_bucket_150_200k.sum = std::stod(parts[idx++]);
            r.income_bucket_150_200k.min = std::stod(parts[idx++]);
            r.income_bucket_150_200k.max = std::stod(parts[idx++]);
            r.income_bucket_150_200k.mean = std::stod(parts[idx++]);
            r.income_bucket_150_200k.median = std::stod(parts[idx++]);
            
            r.income_bucket_200k_plus.count = std::stoll(parts[idx++]);
            r.income_bucket_200k_plus.sum = std::stod(parts[idx++]);
            r.income_bucket_200k_plus.min = std::stod(parts[idx++]);
            r.income_bucket_200k_plus.max = std::stod(parts[idx++]);
            r.income_bucket_200k_plus.mean = std::stod(parts[idx++]);
            r.income_bucket_200k_plus.median = std::stod(parts[idx++]);
            
            // Age
            r.age_sum = std::stoll(parts[idx++]);
            r.age_count = std::stoll(parts[idx++]);
            r.age_min = std::stod(parts[idx++]);
            r.age_max = std::stod(parts[idx++]);
            r.age_mean = std::stod(parts[idx++]);
            r.age_median = std::stod(parts[idx++]);
            
            // Age buckets (5 buckets × 6 fields each)
            r.age_bucket_0_18.count = std::stoll(parts[idx++]);
            r.age_bucket_0_18.sum = std::stod(parts[idx++]);
            r.age_bucket_0_18.min = std::stod(parts[idx++]);
            r.age_bucket_0_18.max = std::stod(parts[idx++]);
            r.age_bucket_0_18.mean = std::stod(parts[idx++]);
            r.age_bucket_0_18.median = std::stod(parts[idx++]);
            
            r.age_bucket_19_35.count = std::stoll(parts[idx++]);
            r.age_bucket_19_35.sum = std::stod(parts[idx++]);
            r.age_bucket_19_35.min = std::stod(parts[idx++]);
            r.age_bucket_19_35.max = std::stod(parts[idx++]);
            r.age_bucket_19_35.mean = std::stod(parts[idx++]);
            r.age_bucket_19_35.median = std::stod(parts[idx++]);
            
            r.age_bucket_36_50.count = std::stoll(parts[idx++]);
            r.age_bucket_36_50.sum = std::stod(parts[idx++]);
            r.age_bucket_36_50.min = std::stod(parts[idx++]);
            r.age_bucket_36_50.max = std::stod(parts[idx++]);
            r.age_bucket_36_50.mean = std::stod(parts[idx++]);
            r.age_bucket_36_50.median = std::stod(parts[idx++]);
            
            r.age_bucket_51_65.count = std::stoll(parts[idx++]);
            r.age_bucket_51_65.sum = std::stod(parts[idx++]);
            r.age_bucket_51_65.min = std::stod(parts[idx++]);
            r.age_bucket_51_65.max = std::stod(parts[idx++]);
            r.age_bucket_51_65.mean = std::stod(parts[idx++]);
            r.age_bucket_51_65.median = std::stod(parts[idx++]);
            
            r.age_bucket_66_plus.count = std::stoll(parts[idx++]);
            r.age_bucket_66_plus.sum = std::stod(parts[idx++]);
            r.age_bucket_66_plus.min = std::stod(parts[idx++]);
            r.age_bucket_66_plus.max = std::stod(parts[idx++]);
            r.age_bucket_66_plus.mean = std::stod(parts[idx++]);
            r.age_bucket_66_plus.median = std::stod(parts[idx++]);
            
            // Unemployment
            r.employed_count = std::stoll(parts[idx++]);
            r.employed_min = std::stod(parts[idx++]);
            r.employed_max = std::stod(parts[idx++]);
            r.employed_mean = std::stod(parts[idx++]);
            r.employed_median = std::stod(parts[idx++]);
            r.unemployed_count = std::stoll(parts[idx++]);
            r.unemployed_min = std::stod(parts[idx++]);
            r.unemployed_max = std::stod(parts[idx++]);
            r.unemployed_mean = std::stod(parts[idx++]);
            r.unemployed_median = std::stod(parts[idx++]);
            
            // Quality
            r.rows_scanned = std::stoll(parts[idx++]);
            r.rows_used = std::stoll(parts[idx++]);
            r.rows_dropped = std::stoll(parts[idx++]);

            aggregates_out.push_back(std::move(r));
        } else {
            throw std::runtime_error("Unknown line type");
        }
    }

    if (!have_status) {
        throw std::runtime_error("Missing STATUS line");
    }
    
}