#include "json_aggregator.hpp"
#include <sstream>
#include <iostream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <set>
#include <limits>

using json = nlohmann::json;

// Helper function to calculate median from a vector of values
static double calculateMedian(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    if (n % 2 == 0) {
        return (values[n/2 - 1] + values[n/2]) / 2.0;
    }
    return values[n/2];
}

// Helper function to calculate mean
static double calculateMean(const std::vector<double> &values) {
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (double v : values) sum += v;
    return sum / values.size();
}

// Merge BucketStats
static BucketStats mergeBucketStats(const std::vector<BucketStats> &buckets) {
    BucketStats result;
    if (buckets.empty()) return result;
    
    result.count = 0;
    result.sum = 0.0;
    result.min = std::numeric_limits<double>::max();
    result.max = std::numeric_limits<double>::lowest();
    
    std::vector<double> all_values;
    
    for (const auto &b : buckets) {
        result.count += b.count;
        result.sum += b.sum;
        if (b.min < result.min && b.count > 0) result.min = b.min;
        if (b.max > result.max && b.count > 0) result.max = b.max;
        // Approximate median calculation by including values
        if (b.count > 0 && b.sum > 0) {
            double avg_per_item = b.sum / b.count;
            for (int64_t i = 0; i < b.count; i++) {
                all_values.push_back(avg_per_item);
            }
        }
    }
    
    if (result.count == 0) {
        result.min = 0.0;
        result.max = 0.0;
    }
    
    result.mean = (result.count > 0) ? (result.sum / result.count) : 0.0;
    result.median = calculateMedian(all_values);
    
    return result;
}

AggregateRow aggregateRows(const std::vector<AggregateRow> &rows) {
    AggregateRow result;
    if (rows.empty()) return result;
    
    result.key = rows[0].key;
    
    std::vector<double> population_vals, household_vals, income_vals, age_vals;
    std::vector<double> employed_vals, unemployed_vals;
    std::vector<BucketStats> income_b_under25k_vec;
    std::vector<BucketStats> income_b_25_50k_vec;
    std::vector<BucketStats> income_b_50_75k_vec;
    std::vector<BucketStats> income_b_75_100k_vec;
    std::vector<BucketStats> income_b_100_150k_vec;
    std::vector<BucketStats> income_b_150_200k_vec;
    std::vector<BucketStats> income_b_200k_plus_vec;

    std::vector<BucketStats> age_b_0_18_vec;
    std::vector<BucketStats> age_b_19_35_vec;
    std::vector<BucketStats> age_b_36_50_vec;
    std::vector<BucketStats> age_b_51_65_vec;
    std::vector<BucketStats> age_b_66_plus_vec;
    
    for (const auto &row : rows) {
        result.population_count += row.population_count;
        result.household_count += row.household_count;
        result.income_sum += row.income_sum;
        result.income_count += row.income_count;
        result.age_sum += row.age_sum;
        result.age_count += row.age_count;
        result.employed_count += row.employed_count;
        result.unemployed_count += row.unemployed_count;
        result.rows_scanned += row.rows_scanned;
        result.rows_used += row.rows_used;
        result.rows_dropped += row.rows_dropped;
        
        // Collect values for statistics
        if (row.population_count > 0) {
            population_vals.push_back(row.population_count);
            if (result.population_min == 0 || row.population_min < result.population_min) 
                result.population_min = row.population_min;
            if (row.population_max > result.population_max) 
                result.population_max = row.population_max;
        }
        
        if (row.household_count > 0) {
            household_vals.push_back(row.household_count);
            if (result.household_min == 0 || row.household_min < result.household_min) 
                result.household_min = row.household_min;
            if (row.household_max > result.household_max) 
                result.household_max = row.household_max;
        }
        
        if (row.income_count > 0) {
            income_vals.push_back(row.income_sum / row.income_count);
            if (result.income_min == 0 || row.income_min < result.income_min) 
                result.income_min = row.income_min;
            if (row.income_max > result.income_max) 
                result.income_max = row.income_max;
        }
        
        if (row.age_count > 0) {
            age_vals.push_back(static_cast<double>(row.age_sum) / row.age_count);
            if (result.age_min == 0 || row.age_min < result.age_min) 
                result.age_min = row.age_min;
            if (row.age_max > result.age_max) 
                result.age_max = row.age_max;
        }
        
        if (row.employed_count > 0) {
            employed_vals.push_back(row.employed_count);
        }
        
        if (row.unemployed_count > 0) {
            unemployed_vals.push_back(row.unemployed_count);
        }
        
        // Aggregate buckets per bucket-type
        income_b_under25k_vec.push_back(row.income_bucket_under25k);
        income_b_25_50k_vec.push_back(row.income_bucket_25_50k);
        income_b_50_75k_vec.push_back(row.income_bucket_50_75k);
        income_b_75_100k_vec.push_back(row.income_bucket_75_100k);
        income_b_100_150k_vec.push_back(row.income_bucket_100_150k);
        income_b_150_200k_vec.push_back(row.income_bucket_150_200k);
        income_b_200k_plus_vec.push_back(row.income_bucket_200k_plus);

        age_b_0_18_vec.push_back(row.age_bucket_0_18);
        age_b_19_35_vec.push_back(row.age_bucket_19_35);
        age_b_36_50_vec.push_back(row.age_bucket_36_50);
        age_b_51_65_vec.push_back(row.age_bucket_51_65);
        age_b_66_plus_vec.push_back(row.age_bucket_66_plus);
    }
    
    // Calculate means and medians
    result.population_mean = calculateMean(population_vals);
    result.population_median = calculateMedian(population_vals);
    
    result.household_mean = calculateMean(household_vals);
    result.household_median = calculateMedian(household_vals);
    
    result.income_mean = (result.income_count > 0) ? (result.income_sum / result.income_count) : 0.0;
    result.income_median = calculateMedian(income_vals);
    
    result.age_mean = (result.age_count > 0) ? (static_cast<double>(result.age_sum) / result.age_count) : 0.0;
    result.age_median = calculateMedian(age_vals);
    
    result.employed_mean = calculateMean(employed_vals);
    result.employed_median = calculateMedian(employed_vals);
    
    result.unemployed_mean = calculateMean(unemployed_vals);
    result.unemployed_median = calculateMedian(unemployed_vals);

    // Merge bucket stats per bucket type and assign to result
    result.income_bucket_under25k = mergeBucketStats(income_b_under25k_vec);
    result.income_bucket_25_50k = mergeBucketStats(income_b_25_50k_vec);
    result.income_bucket_50_75k = mergeBucketStats(income_b_50_75k_vec);
    result.income_bucket_75_100k = mergeBucketStats(income_b_75_100k_vec);
    result.income_bucket_100_150k = mergeBucketStats(income_b_100_150k_vec);
    result.income_bucket_150_200k = mergeBucketStats(income_b_150_200k_vec);
    result.income_bucket_200k_plus = mergeBucketStats(income_b_200k_plus_vec);

    result.age_bucket_0_18 = mergeBucketStats(age_b_0_18_vec);
    result.age_bucket_19_35 = mergeBucketStats(age_b_19_35_vec);
    result.age_bucket_36_50 = mergeBucketStats(age_b_36_50_vec);
    result.age_bucket_51_65 = mergeBucketStats(age_b_51_65_vec);
    result.age_bucket_66_plus = mergeBucketStats(age_b_66_plus_vec);
    
    return result;
}

static json bucketStatsToJson(const BucketStats &stats) {
    json obj;
    obj["count"] = stats.count;
    obj["sum"] = stats.sum;
    obj["min"] = stats.min;
    obj["max"] = stats.max;
    obj["mean"] = stats.mean;
    obj["median"] = stats.median;
    return obj;
}

static json metricStatsToJson(int64_t count, double sum, double min, double max, 
                               double mean, double median) {
    json obj;
    obj["count"] = count;
    obj["sum"] = sum;
    obj["min"] = min;
    obj["max"] = max;
    obj["mean"] = mean;
    obj["median"] = median;
    return obj;
}

static json countOnlyMetricToJson(int64_t count, double min, double max, 
                                   double mean, double median) {
    json obj;
    obj["count"] = count;
    obj["min"] = min;
    obj["max"] = max;
    obj["mean"] = mean;
    obj["median"] = median;
    return obj;
}

static json aggregateRowToJson(const AggregateRow &row, bool include_sum = true) {
    json obj;
    obj["county_fips"] = row.key.county_fips;
    obj["window"]["start"] = row.key.window_start;
    obj["window"]["end"] = row.key.window_end;
    
    json metrics;
    
    // Population
    metrics["population"] = countOnlyMetricToJson(
        row.population_count, row.population_min, row.population_max, 
        row.population_mean, row.population_median
    );
    
    // Households
    metrics["households"] = countOnlyMetricToJson(
        row.household_count, row.household_min, row.household_max,
        row.household_mean, row.household_median
    );
    
    // Income
    metrics["income"] = metricStatsToJson(
        row.income_count, row.income_sum, row.income_min, row.income_max,
        row.income_mean, row.income_median
    );
    
    // Income buckets
    json income_bucket;
    income_bucket["<25k"] = bucketStatsToJson(row.income_bucket_under25k);
    income_bucket["25-50k"] = bucketStatsToJson(row.income_bucket_25_50k);
    income_bucket["50-75k"] = bucketStatsToJson(row.income_bucket_50_75k);
    income_bucket["75-100k"] = bucketStatsToJson(row.income_bucket_75_100k);
    income_bucket["100-150k"] = bucketStatsToJson(row.income_bucket_100_150k);
    income_bucket["150-200k"] = bucketStatsToJson(row.income_bucket_150_200k);
    income_bucket["200k+"] = bucketStatsToJson(row.income_bucket_200k_plus);
    metrics["income_bucket"] = income_bucket;
    
    // Age
    metrics["age"] = countOnlyMetricToJson(
        row.age_count, row.age_min, row.age_max,
        row.age_mean, row.age_median
    );
    
    // Age buckets
    json age_bucket;
    age_bucket["0-18"] = bucketStatsToJson(row.age_bucket_0_18);
    age_bucket["19-35"] = bucketStatsToJson(row.age_bucket_19_35);
    age_bucket["36-50"] = bucketStatsToJson(row.age_bucket_36_50);
    age_bucket["51-65"] = bucketStatsToJson(row.age_bucket_51_65);
    age_bucket["66+"] = bucketStatsToJson(row.age_bucket_66_plus);
    metrics["age_bucket"] = age_bucket;
    
    // Unemployment rate
    json unemployment;
    unemployment["with_jobs"] = countOnlyMetricToJson(
        row.employed_count, row.employed_min, row.employed_max,
        row.employed_mean, row.employed_median
    );
    unemployment["without_jobs"] = countOnlyMetricToJson(
        row.unemployed_count, row.unemployed_min, row.unemployed_max,
        row.unemployed_mean, row.unemployed_median
    );
    metrics["unemployment_rate"] = unemployment;
    
    obj["metrics"] = metrics;
    
    json quality;
    quality["rows_scanned"] = row.rows_scanned;
    quality["rows_used"] = row.rows_used;
    quality["rows_dropped"] = row.rows_dropped;
    obj["quality"] = quality;
    
    return obj;
}

std::string createJsonReport(
    const std::string &start_date,
    const std::string &end_date,
    const std::vector<WorkerNodeData> &worker_data_list
) {
    json report;

    // Metadata
    json metadata;
    metadata["source"] = "demographic_data";
    
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
    metadata["timestamp"] = ss.str();
    
    metadata["time_window"]["start"] = start_date + "T00:00:00Z";
    metadata["time_window"]["end"] = end_date + "T23:59:59Z";
    
    json counties = json::array();
    for (const auto &worker : worker_data_list) {
        counties.push_back(worker.county_fips);
    }
    metadata["counties"] = counties;
    
    report["coordinator"]["Metadata"] = metadata;

    // Aggregate all data for coordinator aggregates
    std::map<AggregateKey, std::vector<AggregateRow>> grouped_rows;
    std::set<std::string> all_counties;
    
    for (const auto &worker : worker_data_list) {
        all_counties.insert(worker.county_fips);
        for (const auto &agg : worker.aggregates) {
            grouped_rows[agg.key].push_back(agg);
        }
    }
    
    // Calculate global aggregates
    std::vector<AggregateRow> global_aggs;
    for (auto &[key, rows] : grouped_rows) {
        global_aggs.push_back(aggregateRows(rows));
    }
    
    // Aggregate across all keys for coordinator-level metrics
    AggregateRow coordinator_agg = aggregateRows(global_aggs);
    
    // Create coordinator aggregates section
    json coordinator_metrics;
    
    coordinator_metrics["population"] = countOnlyMetricToJson(
        coordinator_agg.population_count, coordinator_agg.population_min, coordinator_agg.population_max,
        coordinator_agg.population_mean, coordinator_agg.population_median
    );
    
    coordinator_metrics["households"] = countOnlyMetricToJson(
        coordinator_agg.household_count, coordinator_agg.household_min, coordinator_agg.household_max,
        coordinator_agg.household_mean, coordinator_agg.household_median
    );
    
    coordinator_metrics["income"] = metricStatsToJson(
        coordinator_agg.income_count, coordinator_agg.income_sum, coordinator_agg.income_min, 
        coordinator_agg.income_max, coordinator_agg.income_mean, coordinator_agg.income_median
    );
    
    json income_bucket;
    income_bucket["<25k"] = bucketStatsToJson(coordinator_agg.income_bucket_under25k);
    income_bucket["25-50k"] = bucketStatsToJson(coordinator_agg.income_bucket_25_50k);
    income_bucket["50-75k"] = bucketStatsToJson(coordinator_agg.income_bucket_50_75k);
    income_bucket["75-100k"] = bucketStatsToJson(coordinator_agg.income_bucket_75_100k);
    income_bucket["100-150k"] = bucketStatsToJson(coordinator_agg.income_bucket_100_150k);
    income_bucket["150-200k"] = bucketStatsToJson(coordinator_agg.income_bucket_150_200k);
    income_bucket["200k+"] = bucketStatsToJson(coordinator_agg.income_bucket_200k_plus);
    coordinator_metrics["income_bucket"] = income_bucket;
    
    coordinator_metrics["age"] = countOnlyMetricToJson(
        coordinator_agg.age_count, coordinator_agg.age_min, coordinator_agg.age_max,
        coordinator_agg.age_mean, coordinator_agg.age_median
    );
    
    json age_bucket;
    age_bucket["0-18"] = bucketStatsToJson(coordinator_agg.age_bucket_0_18);
    age_bucket["19-35"] = bucketStatsToJson(coordinator_agg.age_bucket_19_35);
    age_bucket["36-50"] = bucketStatsToJson(coordinator_agg.age_bucket_36_50);
    age_bucket["51-65"] = bucketStatsToJson(coordinator_agg.age_bucket_51_65);
    age_bucket["66+"] = bucketStatsToJson(coordinator_agg.age_bucket_66_plus);
    coordinator_metrics["age_bucket"] = age_bucket;
    
    json unemployment;
    unemployment["with_jobs"] = countOnlyMetricToJson(
        coordinator_agg.employed_count, coordinator_agg.employed_min, coordinator_agg.employed_max,
        coordinator_agg.employed_mean, coordinator_agg.employed_median
    );
    unemployment["without_jobs"] = countOnlyMetricToJson(
        coordinator_agg.unemployed_count, coordinator_agg.unemployed_min, coordinator_agg.unemployed_max,
        coordinator_agg.unemployed_mean, coordinator_agg.unemployed_median
    );
    coordinator_metrics["unemployment_rate"] = unemployment;
    
    json coordinator_aggregates;
    coordinator_aggregates["metrics"] = coordinator_metrics;
    
    json coordinator_quality;
    coordinator_quality["rows_scanned"] = coordinator_agg.rows_scanned;
    coordinator_quality["rows_used"] = coordinator_agg.rows_used;
    coordinator_quality["rows_dropped"] = coordinator_agg.rows_dropped;
    coordinator_aggregates["quality"] = coordinator_quality;
    
    report["coordinator"]["aggregates"] = coordinator_aggregates;

    // Worker nodes
    json worker_nodes = json::array();
    for (const auto &worker : worker_data_list) {
        json worker_entry;
        worker_entry["rank"] = worker.rank;
        worker_entry["source_description"] = worker.source_description;
        worker_entry["county_fips"] = worker.county_fips;
        
        worker_entry["status"]["ok"] = worker.status.ok;
        worker_entry["status"]["message"] = worker.status.message;
        worker_entry["status"]["elapsed_ms"] = worker.status.elapsed_ms;
        
        // Aggregate worker's data
        AggregateRow worker_agg = aggregateRows(worker.aggregates);
        
        worker_entry["metrics"] = json::object();
        
        worker_entry["metrics"]["population"] = countOnlyMetricToJson(
            worker_agg.population_count, worker_agg.population_min, worker_agg.population_max,
            worker_agg.population_mean, worker_agg.population_median
        );
        
        worker_entry["metrics"]["households"] = countOnlyMetricToJson(
            worker_agg.household_count, worker_agg.household_min, worker_agg.household_max,
            worker_agg.household_mean, worker_agg.household_median
        );
        
        worker_entry["metrics"]["income"] = metricStatsToJson(
            worker_agg.income_count, worker_agg.income_sum, worker_agg.income_min,
            worker_agg.income_max, worker_agg.income_mean, worker_agg.income_median
        );
        
        json worker_income_bucket;
        worker_income_bucket["<25k"] = bucketStatsToJson(worker_agg.income_bucket_under25k);
        worker_income_bucket["25-50k"] = bucketStatsToJson(worker_agg.income_bucket_25_50k);
        worker_income_bucket["50-75k"] = bucketStatsToJson(worker_agg.income_bucket_50_75k);
        worker_income_bucket["75-100k"] = bucketStatsToJson(worker_agg.income_bucket_75_100k);
        worker_income_bucket["100-150k"] = bucketStatsToJson(worker_agg.income_bucket_100_150k);
        worker_income_bucket["150-200k"] = bucketStatsToJson(worker_agg.income_bucket_150_200k);
        worker_income_bucket["200k+"] = bucketStatsToJson(worker_agg.income_bucket_200k_plus);
        worker_entry["metrics"]["income_bucket"] = worker_income_bucket;
        
        worker_entry["metrics"]["age"] = countOnlyMetricToJson(
            worker_agg.age_count, worker_agg.age_min, worker_agg.age_max,
            worker_agg.age_mean, worker_agg.age_median
        );
        
        json worker_age_bucket;
        worker_age_bucket["0-18"] = bucketStatsToJson(worker_agg.age_bucket_0_18);
        worker_age_bucket["19-35"] = bucketStatsToJson(worker_agg.age_bucket_19_35);
        worker_age_bucket["36-50"] = bucketStatsToJson(worker_agg.age_bucket_36_50);
        worker_age_bucket["51-65"] = bucketStatsToJson(worker_agg.age_bucket_51_65);
        worker_age_bucket["66+"] = bucketStatsToJson(worker_agg.age_bucket_66_plus);
        worker_entry["metrics"]["age_bucket"] = worker_age_bucket;
        
        json worker_unemployment;
        worker_unemployment["with_jobs"] = countOnlyMetricToJson(
            worker_agg.employed_count, worker_agg.employed_min, worker_agg.employed_max,
            worker_agg.employed_mean, worker_agg.employed_median
        );
        worker_unemployment["without_jobs"] = countOnlyMetricToJson(
            worker_agg.unemployed_count, worker_agg.unemployed_min, worker_agg.unemployed_max,
            worker_agg.unemployed_mean, worker_agg.unemployed_median
        );
        worker_entry["metrics"]["unemployment_rate"] = worker_unemployment;
        
        json worker_quality;
        worker_quality["rows_scanned"] = worker.status.rows_scanned;
        worker_quality["rows_used"] = worker.status.rows_used;
        worker_quality["rows_dropped"] = worker.status.rows_dropped;
        worker_entry["quality"] = worker_quality;
        
        worker_nodes.push_back(worker_entry);
    }
    
    report["worker_nodes"] = worker_nodes;

    return report.dump(2);
}

