#pragma once
#include <string>
#include <cstdint>
#include <vector>
#include <map>

struct AggregateKey {
    std::string county_fips;
    std::string window_start;
    std::string window_end;
};

// Income bucket statistics
struct BucketStats {
    int64_t count = 0;
    double sum = 0.0;
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double median = 0.0;
};

struct AggregateRow {
    AggregateKey key;

    // Population
    int64_t population_count = 0;
    double population_min = 0.0;
    double population_max = 0.0;
    double population_mean = 0.0;
    double population_median = 0.0;

    // Households
    int64_t household_count = 0;
    double household_min = 0.0;
    double household_max = 0.0;
    double household_mean = 0.0;
    double household_median = 0.0;

    // Income
    double income_sum = 0.0;
    int64_t income_count = 0;
    double income_min = 0.0;
    double income_max = 0.0;
    double income_mean = 0.0;
    double income_median = 0.0;

    // Income buckets
    BucketStats income_bucket_under25k;
    BucketStats income_bucket_25_50k;
    BucketStats income_bucket_50_75k;
    BucketStats income_bucket_75_100k;
    BucketStats income_bucket_100_150k;
    BucketStats income_bucket_150_200k;
    BucketStats income_bucket_200k_plus;

    // Age
    int64_t age_sum = 0;
    int64_t age_count = 0;
    double age_min = 0.0;
    double age_max = 0.0;
    double age_mean = 0.0;
    double age_median = 0.0;

    // Age buckets (count only for now)
    BucketStats age_bucket_0_18;
    BucketStats age_bucket_19_35;
    BucketStats age_bucket_36_50;
    BucketStats age_bucket_51_65;
    BucketStats age_bucket_66_plus;

    // Unemployment
    int64_t employed_count = 0;
    int64_t unemployed_count = 0;
    double employed_min = 0.0;
    double employed_max = 0.0;
    double employed_mean = 0.0;
    double employed_median = 0.0;
    double unemployed_min = 0.0;
    double unemployed_max = 0.0;
    double unemployed_mean = 0.0;
    double unemployed_median = 0.0;

    // Quality
    int64_t rows_scanned = 0;
    int64_t rows_used = 0;
    int64_t rows_dropped = 0;
};