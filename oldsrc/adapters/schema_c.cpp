#include "adapter.hpp"
#include <chrono>
#include <memory>
#include <cmath>

// Helper function to safely parse a numeric string
static double parseDouble(const char *val) {
    if (!val || !*val) return 0.0;
    return std::stod(val);
}

static int64_t parseInt(const char *val) {
    if (!val || !*val) return 0;
    return std::stoll(val);
}

class SchemaCAdapter : public Adapter {
public:
    AdapterResult run(PgDb &db, const ContractRequest &req, const std::string &county_fips) override {
        auto t0 = std::chrono::steady_clock::now();

        AdapterResult out;
        out.status.ok = true;

        const char *params[2] = { req.start_date.c_str(), req.end_date.c_str() };

        PGresult *res = db.execParams(
            "WITH base AS (\n"
            "  SELECT\n"
            "    date_trunc('month', created_at_utc)::date AS month_start,\n"
            "    (date_trunc('month', created_at_utc) + interval '1 month')::date AS month_end,\n"
            "    profile_json\n"
            "  FROM citizen\n"
            "  WHERE created_at_utc >= $1::date\n"
            "    AND created_at_utc <  $2::date\n"
            "),\n"
            "extracted AS (\n"
            "  SELECT\n"
            "    month_start, month_end,\n"
            "    (profile_json #>> '{demographics,dob}') AS dob_s,\n"
            "    (profile_json #>> '{residency,moved_in_date}') AS moved_in_s,\n"
            "    (profile_json #>> '{residency,move_out_date}') AS move_out_s,\n"
            "    (profile_json #>> '{residency,death_date}') AS death_s,\n"
            "    (profile_json #>> '{residency,active_status}') AS active_s,\n"
            "    (profile_json #>> '{employment,employment_status}') AS employment_status,\n"
            "    (profile_json #>> '{employment,has_job}') AS has_job_s,\n"
            "    NULLIF(profile_json #>> '{household,income_usd}', '')::numeric AS income_usd\n"
            "  FROM base\n"
            "),\n"
            "normalized AS (\n"
            "  SELECT\n"
            "    month_start, month_end,\n"
            "    CASE WHEN dob_s ~ '^\\d{4}-\\d{2}-\\d{2}$' THEN dob_s::date ELSE NULL END AS dob_dt,\n"
            "    CASE WHEN moved_in_s ~ '^\\d{4}-\\d{2}-\\d{2}$' THEN moved_in_s::date ELSE NULL END AS moved_in_dt,\n"
            "    CASE WHEN move_out_s ~ '^\\d{4}-\\d{2}-\\d{2}$' THEN move_out_s::date ELSE NULL END AS move_out_dt,\n"
            "    CASE WHEN death_s ~ '^\\d{4}-\\d{2}-\\d{2}$' THEN death_s::date ELSE NULL END AS death_dt,\n"
            "    CASE\n"
            "      WHEN active_s IN ('true','TRUE','1','t','T') THEN TRUE\n"
            "      WHEN active_s IN ('false','FALSE','0','f','F') THEN FALSE\n"
            "      ELSE NULL\n"
            "    END AS active_bool,\n"
            "    income_usd,\n"
            "    (employment_status IN ('Employed','Unemployed') OR has_job_s IS NOT NULL) AS in_labor_force,\n"
            "    (has_job_s IN ('true','TRUE','1','t','T') OR employment_status='Employed') AS is_employed,\n"
            "    (employment_status='Unemployed') AS is_unemployed\n"
            "  FROM extracted\n"
            "),\n"
            "derived AS (\n"
            "  SELECT\n"
            "    *,\n"
            "    (\n"
            "      active_bool IS TRUE\n"
            "      AND moved_in_dt IS NOT NULL\n"
            "      AND moved_in_dt < month_end\n"
            "      AND (move_out_dt IS NULL OR move_out_dt >= month_end)\n"
            "      AND (death_dt    IS NULL OR death_dt    >= month_end)\n"
            "      AND dob_dt IS NOT NULL\n"
            "      AND dob_dt < month_end\n"
            "    ) AS is_valid,\n"
            "    CASE\n"
            "      WHEN dob_dt IS NULL OR dob_dt >= month_end THEN NULL\n"
            "      ELSE EXTRACT(YEAR FROM age(month_end, dob_dt))::int\n"
            "    END AS age_years,\n"
            "    CASE\n"
            "      WHEN income_usd IS NULL OR income_usd < 0 THEN NULL\n"
            "      WHEN income_usd < 25000  THEN '<25k'\n"
            "      WHEN income_usd < 50000  THEN '25-50k'\n"
            "      WHEN income_usd < 75000  THEN '50-75k'\n"
            "      WHEN income_usd < 100000 THEN '75-100k'\n"
            "      WHEN income_usd < 150000 THEN '100-150k'\n"
            "      WHEN income_usd < 200000 THEN '150-200k'\n"
            "      ELSE '200k+'\n"
            "    END AS income_bucket,\n"
            "    CASE\n"
            "      WHEN dob_dt IS NULL OR dob_dt >= month_end THEN NULL\n"
            "      WHEN EXTRACT(YEAR FROM age(month_end, dob_dt))::int < 18 THEN '0-18'\n"
            "      WHEN EXTRACT(YEAR FROM age(month_end, dob_dt))::int < 35 THEN '19-35'\n"
            "      WHEN EXTRACT(YEAR FROM age(month_end, dob_dt))::int < 50 THEN '36-50'\n"
            "      WHEN EXTRACT(YEAR FROM age(month_end, dob_dt))::int < 65 THEN '51-65'\n"
            "      ELSE '66+'\n"
            "    END AS age_bucket\n"
            "  FROM normalized\n"
            ")\n"
            "SELECT\n"
               "    month_start,\n"
               "    month_end,\n"
               "    COUNT(*) FILTER (WHERE is_valid) AS population_count,\n"
               "    COUNT(DISTINCT (profile_json #>> '{household,hh_id}')) FILTER (WHERE is_valid) AS household_count,\n"
               "    1 AS population_min,\n"
               "    1 AS population_max,\n"
               "    1 AS household_min,\n"
               "    1 AS household_max,\n"
            "  COUNT(income_usd) FILTER (WHERE is_valid AND income_usd IS NOT NULL AND income_usd >= 0) AS income_count,\n"
            "  COALESCE(SUM(income_usd) FILTER (WHERE is_valid AND income_usd IS NOT NULL AND income_usd >= 0), 0) AS income_sum,\n"
            "  COALESCE(MIN(income_usd) FILTER (WHERE is_valid AND income_usd IS NOT NULL AND income_usd >= 0), 0) AS income_min,\n"
            "  COALESCE(MAX(income_usd) FILTER (WHERE is_valid AND income_usd IS NOT NULL AND income_usd >= 0), 0) AS income_max,\n"
            "  COUNT(*) FILTER (WHERE is_valid AND income_bucket = '<25k')      AS income_b_lt25k_count,\n"
            "  COALESCE(SUM(income_usd) FILTER (WHERE is_valid AND income_bucket = '<25k'), 0) AS income_b_lt25k_sum,\n"
            "  COALESCE(MIN(income_usd) FILTER (WHERE is_valid AND income_bucket = '<25k'), 0) AS income_b_lt25k_min,\n"
            "  COALESCE(MAX(income_usd) FILTER (WHERE is_valid AND income_bucket = '<25k'), 0) AS income_b_lt25k_max,\n"
            "  COUNT(*) FILTER (WHERE is_valid AND income_bucket = '25-50k')    AS income_b_25_50k_count,\n"
            "  COALESCE(SUM(income_usd) FILTER (WHERE is_valid AND income_bucket = '25-50k'), 0) AS income_b_25_50k_sum,\n"
            "  COALESCE(MIN(income_usd) FILTER (WHERE is_valid AND income_bucket = '25-50k'), 0) AS income_b_25_50k_min,\n"
            "  COALESCE(MAX(income_usd) FILTER (WHERE is_valid AND income_bucket = '25-50k'), 0) AS income_b_25_50k_max,\n"
            "  COUNT(*) FILTER (WHERE is_valid AND income_bucket = '50-75k')    AS income_b_50_75k_count,\n"
            "  COALESCE(SUM(income_usd) FILTER (WHERE is_valid AND income_bucket = '50-75k'), 0) AS income_b_50_75k_sum,\n"
            "  COALESCE(MIN(income_usd) FILTER (WHERE is_valid AND income_bucket = '50-75k'), 0) AS income_b_50_75k_min,\n"
            "  COALESCE(MAX(income_usd) FILTER (WHERE is_valid AND income_bucket = '50-75k'), 0) AS income_b_50_75k_max,\n"
            "  COUNT(*) FILTER (WHERE is_valid AND income_bucket = '75-100k')   AS income_b_75_100k_count,\n"
            "  COALESCE(SUM(income_usd) FILTER (WHERE is_valid AND income_bucket = '75-100k'), 0) AS income_b_75_100k_sum,\n"
            "  COALESCE(MIN(income_usd) FILTER (WHERE is_valid AND income_bucket = '75-100k'), 0) AS income_b_75_100k_min,\n"
            "  COALESCE(MAX(income_usd) FILTER (WHERE is_valid AND income_bucket = '75-100k'), 0) AS income_b_75_100k_max,\n"
            "  COUNT(*) FILTER (WHERE is_valid AND income_bucket = '100-150k')  AS income_b_100_150k_count,\n"
            "  COALESCE(SUM(income_usd) FILTER (WHERE is_valid AND income_bucket = '100-150k'), 0) AS income_b_100_150k_sum,\n"
            "  COALESCE(MIN(income_usd) FILTER (WHERE is_valid AND income_bucket = '100-150k'), 0) AS income_b_100_150k_min,\n"
            "  COALESCE(MAX(income_usd) FILTER (WHERE is_valid AND income_bucket = '100-150k'), 0) AS income_b_100_150k_max,\n"
            "  COUNT(*) FILTER (WHERE is_valid AND income_bucket = '150-200k')  AS income_b_150_200k_count,\n"
            "  COALESCE(SUM(income_usd) FILTER (WHERE is_valid AND income_bucket = '150-200k'), 0) AS income_b_150_200k_sum,\n"
            "  COALESCE(MIN(income_usd) FILTER (WHERE is_valid AND income_bucket = '150-200k'), 0) AS income_b_150_200k_min,\n"
            "  COALESCE(MAX(income_usd) FILTER (WHERE is_valid AND income_bucket = '150-200k'), 0) AS income_b_150_200k_max,\n"
            "  COUNT(*) FILTER (WHERE is_valid AND income_bucket = '200k+')     AS income_b_200k_count,\n"
            "  COALESCE(SUM(income_usd) FILTER (WHERE is_valid AND income_bucket = '200k+'), 0) AS income_b_200k_sum,\n"
            "  COALESCE(MIN(income_usd) FILTER (WHERE is_valid AND income_bucket = '200k+'), 0) AS income_b_200k_min,\n"
            "  COALESCE(MAX(income_usd) FILTER (WHERE is_valid AND income_bucket = '200k+'), 0) AS income_b_200k_max,\n"
            "  COUNT(age_years) FILTER (WHERE is_valid AND age_years IS NOT NULL) AS age_count,\n"
            "  COALESCE(SUM(age_years) FILTER (WHERE is_valid AND age_years IS NOT NULL), 0) AS age_sum,\n"
            "  COALESCE(MIN(age_years) FILTER (WHERE is_valid AND age_years IS NOT NULL), 0) AS age_min,\n"
            "  COALESCE(MAX(age_years) FILTER (WHERE is_valid AND age_years IS NOT NULL), 0) AS age_max,\n"
            "  COUNT(*) FILTER (WHERE is_valid AND age_bucket = '0-18')   AS age_b_0_18_count,\n"
            "  COALESCE(SUM(age_years) FILTER (WHERE is_valid AND age_bucket = '0-18'), 0) AS age_b_0_18_sum,\n"
            "  COALESCE(MIN(age_years) FILTER (WHERE is_valid AND age_bucket = '0-18'), 0) AS age_b_0_18_min,\n"
            "  COALESCE(MAX(age_years) FILTER (WHERE is_valid AND age_bucket = '0-18'), 0) AS age_b_0_18_max,\n"
            "  COUNT(*) FILTER (WHERE is_valid AND age_bucket = '19-35')  AS age_b_19_35_count,\n"
            "  COALESCE(SUM(age_years) FILTER (WHERE is_valid AND age_bucket = '19-35'), 0) AS age_b_19_35_sum,\n"
            "  COALESCE(MIN(age_years) FILTER (WHERE is_valid AND age_bucket = '19-35'), 0) AS age_b_19_35_min,\n"
            "  COALESCE(MAX(age_years) FILTER (WHERE is_valid AND age_bucket = '19-35'), 0) AS age_b_19_35_max,\n"
            "  COUNT(*) FILTER (WHERE is_valid AND age_bucket = '36-50')  AS age_b_36_50_count,\n"
            "  COALESCE(SUM(age_years) FILTER (WHERE is_valid AND age_bucket = '36-50'), 0) AS age_b_36_50_sum,\n"
            "  COALESCE(MIN(age_years) FILTER (WHERE is_valid AND age_bucket = '36-50'), 0) AS age_b_36_50_min,\n"
            "  COALESCE(MAX(age_years) FILTER (WHERE is_valid AND age_bucket = '36-50'), 0) AS age_b_36_50_max,\n"
            "  COUNT(*) FILTER (WHERE is_valid AND age_bucket = '51-65')  AS age_b_51_65_count,\n"
            "  COALESCE(SUM(age_years) FILTER (WHERE is_valid AND age_bucket = '51-65'), 0) AS age_b_51_65_sum,\n"
            "  COALESCE(MIN(age_years) FILTER (WHERE is_valid AND age_bucket = '51-65'), 0) AS age_b_51_65_min,\n"
            "  COALESCE(MAX(age_years) FILTER (WHERE is_valid AND age_bucket = '51-65'), 0) AS age_b_51_65_max,\n"
            "  COUNT(*) FILTER (WHERE is_valid AND age_bucket = '66+')    AS age_b_66p_count,\n"
            "  COALESCE(SUM(age_years) FILTER (WHERE is_valid AND age_bucket = '66+'), 0) AS age_b_66p_sum,\n"
            "  COALESCE(MIN(age_years) FILTER (WHERE is_valid AND age_bucket = '66+'), 0) AS age_b_66p_min,\n"
            "  COALESCE(MAX(age_years) FILTER (WHERE is_valid AND age_bucket = '66+'), 0) AS age_b_66p_max,\n"
            "  COUNT(*) FILTER (WHERE is_valid AND in_labor_force AND is_employed)   AS employed_count,\n"
            "  COUNT(*) FILTER (WHERE is_valid AND in_labor_force AND is_unemployed) AS unemployed_count,\n"
            "  COUNT(*) AS rows_scanned,\n"
            "  COUNT(*) FILTER (WHERE is_valid) AS rows_used,\n"
            "  COUNT(*) FILTER (WHERE NOT is_valid) AS rows_dropped\n"
            "FROM derived\n"
            "GROUP BY month_start, month_end\n"
            "ORDER BY month_start;",
            2, params
        );

        if (PQresultStatus(res) != PGRES_TUPLES_OK) {
            out.status.ok = false;
            out.status.message = "SchemaC: Query failed: " + std::string(PQerrorMessage(db.raw()));
            if (res)
            {
                PQclear(res);
            }
            return out;
        }

        int nrows = PQntuples(res);

        for (int i = 0; i < nrows; i++) {
            AggregateRow row;
            row.key.county_fips = county_fips;
            row.key.window_start = PQgetvalue(res, i, 0);
            row.key.window_end = PQgetvalue(res, i, 1);
            
            int col = 2;
            
            // Population
            row.population_count = parseInt(PQgetvalue(res, i, col++));
            // Households (distinct household hh_id)
            row.household_count = parseInt(PQgetvalue(res, i, col++));
            row.population_min = parseDouble(PQgetvalue(res, i, col++));
            row.population_max = parseDouble(PQgetvalue(res, i, col++));
            row.household_min = parseDouble(PQgetvalue(res, i, col++));
            row.household_max = parseDouble(PQgetvalue(res, i, col++));
            if (row.population_count > 0) {
                row.population_mean = static_cast<double>(row.population_count);
                row.population_median = static_cast<double>(row.population_count);
            }
            
            // Income
            row.income_count = parseInt(PQgetvalue(res, i, col++));
            row.income_sum = parseDouble(PQgetvalue(res, i, col++));
            row.income_min = parseDouble(PQgetvalue(res, i, col++));
            row.income_max = parseDouble(PQgetvalue(res, i, col++));
            if (row.income_count > 0) {
                row.income_mean = row.income_sum / row.income_count;
            }
            
            // Income buckets (7 buckets × 4 fields each)
            row.income_bucket_under25k.count = parseInt(PQgetvalue(res, i, col++));
            row.income_bucket_under25k.sum = parseDouble(PQgetvalue(res, i, col++));
            row.income_bucket_under25k.min = parseDouble(PQgetvalue(res, i, col++));
            row.income_bucket_under25k.max = parseDouble(PQgetvalue(res, i, col++));
            if (row.income_bucket_under25k.count > 0) {
                row.income_bucket_under25k.mean = row.income_bucket_under25k.sum / row.income_bucket_under25k.count;
            }
            
            row.income_bucket_25_50k.count = parseInt(PQgetvalue(res, i, col++));
            row.income_bucket_25_50k.sum = parseDouble(PQgetvalue(res, i, col++));
            row.income_bucket_25_50k.min = parseDouble(PQgetvalue(res, i, col++));
            row.income_bucket_25_50k.max = parseDouble(PQgetvalue(res, i, col++));
            if (row.income_bucket_25_50k.count > 0) {
                row.income_bucket_25_50k.mean = row.income_bucket_25_50k.sum / row.income_bucket_25_50k.count;
            }
            
            row.income_bucket_50_75k.count = parseInt(PQgetvalue(res, i, col++));
            row.income_bucket_50_75k.sum = parseDouble(PQgetvalue(res, i, col++));
            row.income_bucket_50_75k.min = parseDouble(PQgetvalue(res, i, col++));
            row.income_bucket_50_75k.max = parseDouble(PQgetvalue(res, i, col++));
            if (row.income_bucket_50_75k.count > 0) {
                row.income_bucket_50_75k.mean = row.income_bucket_50_75k.sum / row.income_bucket_50_75k.count;
            }
            
            row.income_bucket_75_100k.count = parseInt(PQgetvalue(res, i, col++));
            row.income_bucket_75_100k.sum = parseDouble(PQgetvalue(res, i, col++));
            row.income_bucket_75_100k.min = parseDouble(PQgetvalue(res, i, col++));
            row.income_bucket_75_100k.max = parseDouble(PQgetvalue(res, i, col++));
            if (row.income_bucket_75_100k.count > 0) {
                row.income_bucket_75_100k.mean = row.income_bucket_75_100k.sum / row.income_bucket_75_100k.count;
            }
            
            row.income_bucket_100_150k.count = parseInt(PQgetvalue(res, i, col++));
            row.income_bucket_100_150k.sum = parseDouble(PQgetvalue(res, i, col++));
            row.income_bucket_100_150k.min = parseDouble(PQgetvalue(res, i, col++));
            row.income_bucket_100_150k.max = parseDouble(PQgetvalue(res, i, col++));
            if (row.income_bucket_100_150k.count > 0) {
                row.income_bucket_100_150k.mean = row.income_bucket_100_150k.sum / row.income_bucket_100_150k.count;
            }
            
            row.income_bucket_150_200k.count = parseInt(PQgetvalue(res, i, col++));
            row.income_bucket_150_200k.sum = parseDouble(PQgetvalue(res, i, col++));
            row.income_bucket_150_200k.min = parseDouble(PQgetvalue(res, i, col++));
            row.income_bucket_150_200k.max = parseDouble(PQgetvalue(res, i, col++));
            if (row.income_bucket_150_200k.count > 0) {
                row.income_bucket_150_200k.mean = row.income_bucket_150_200k.sum / row.income_bucket_150_200k.count;
            }
            
            row.income_bucket_200k_plus.count = parseInt(PQgetvalue(res, i, col++));
            row.income_bucket_200k_plus.sum = parseDouble(PQgetvalue(res, i, col++));
            row.income_bucket_200k_plus.min = parseDouble(PQgetvalue(res, i, col++));
            row.income_bucket_200k_plus.max = parseDouble(PQgetvalue(res, i, col++));
            if (row.income_bucket_200k_plus.count > 0) {
                row.income_bucket_200k_plus.mean = row.income_bucket_200k_plus.sum / row.income_bucket_200k_plus.count;
            }
            
            // Age
            row.age_count = parseInt(PQgetvalue(res, i, col++));
            row.age_sum = parseInt(PQgetvalue(res, i, col++));
            row.age_min = parseDouble(PQgetvalue(res, i, col++));
            row.age_max = parseDouble(PQgetvalue(res, i, col++));
            if (row.age_count > 0) {
                row.age_mean = static_cast<double>(row.age_sum) / row.age_count;
            }
            
            // Age buckets (5 buckets × 4 fields each)
            row.age_bucket_0_18.count = parseInt(PQgetvalue(res, i, col++));
            row.age_bucket_0_18.sum = parseDouble(PQgetvalue(res, i, col++));
            row.age_bucket_0_18.min = parseDouble(PQgetvalue(res, i, col++));
            row.age_bucket_0_18.max = parseDouble(PQgetvalue(res, i, col++));
            if (row.age_bucket_0_18.count > 0) {
                row.age_bucket_0_18.mean = row.age_bucket_0_18.sum / row.age_bucket_0_18.count;
            }
            
            row.age_bucket_19_35.count = parseInt(PQgetvalue(res, i, col++));
            row.age_bucket_19_35.sum = parseDouble(PQgetvalue(res, i, col++));
            row.age_bucket_19_35.min = parseDouble(PQgetvalue(res, i, col++));
            row.age_bucket_19_35.max = parseDouble(PQgetvalue(res, i, col++));
            if (row.age_bucket_19_35.count > 0) {
                row.age_bucket_19_35.mean = row.age_bucket_19_35.sum / row.age_bucket_19_35.count;
            }
            
            row.age_bucket_36_50.count = parseInt(PQgetvalue(res, i, col++));
            row.age_bucket_36_50.sum = parseDouble(PQgetvalue(res, i, col++));
            row.age_bucket_36_50.min = parseDouble(PQgetvalue(res, i, col++));
            row.age_bucket_36_50.max = parseDouble(PQgetvalue(res, i, col++));
            if (row.age_bucket_36_50.count > 0) {
                row.age_bucket_36_50.mean = row.age_bucket_36_50.sum / row.age_bucket_36_50.count;
            }
            
            row.age_bucket_51_65.count = parseInt(PQgetvalue(res, i, col++));
            row.age_bucket_51_65.sum = parseDouble(PQgetvalue(res, i, col++));
            row.age_bucket_51_65.min = parseDouble(PQgetvalue(res, i, col++));
            row.age_bucket_51_65.max = parseDouble(PQgetvalue(res, i, col++));
            if (row.age_bucket_51_65.count > 0) {
                row.age_bucket_51_65.mean = row.age_bucket_51_65.sum / row.age_bucket_51_65.count;
            }
            
            row.age_bucket_66_plus.count = parseInt(PQgetvalue(res, i, col++));
            row.age_bucket_66_plus.sum = parseDouble(PQgetvalue(res, i, col++));
            row.age_bucket_66_plus.min = parseDouble(PQgetvalue(res, i, col++));
            row.age_bucket_66_plus.max = parseDouble(PQgetvalue(res, i, col++));
            if (row.age_bucket_66_plus.count > 0) {
                row.age_bucket_66_plus.mean = row.age_bucket_66_plus.sum / row.age_bucket_66_plus.count;
            }
            
            // Unemployment
            row.employed_count = parseInt(PQgetvalue(res, i, col++));
            row.unemployed_count = parseInt(PQgetvalue(res, i, col++));
            row.employed_min = (row.employed_count > 0) ? 1.0 : 0.0;
            row.employed_max = (row.employed_count > 0) ? 1.0 : 0.0;
            row.employed_mean = (row.employed_count > 0) ? 1.0 : 0.0;
            row.unemployed_min = (row.unemployed_count > 0) ? 1.0 : 0.0;
            row.unemployed_max = (row.unemployed_count > 0) ? 1.0 : 0.0;
            row.unemployed_mean = (row.unemployed_count > 0) ? 1.0 : 0.0;
            
            // Quality
            row.rows_scanned = parseInt(PQgetvalue(res, i, col++));
            row.rows_used = parseInt(PQgetvalue(res, i, col++));
            row.rows_dropped = parseInt(PQgetvalue(res, i, col++));

            out.aggregateRows.push_back(std::move(row));
        }

        // Aggregate row counting statistics across all result rows
        out.status.rows_scanned = 0;
        out.status.rows_used = 0;
        out.status.rows_dropped = 0;
        for (const auto &row : out.aggregateRows) {
            out.status.rows_scanned += row.rows_scanned;
            out.status.rows_used += row.rows_used;
            out.status.rows_dropped += row.rows_dropped;
        }

        PQclear(res);

        auto t1 = std::chrono::steady_clock::now();
        out.status.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        return out;
    }
};

std::unique_ptr<Adapter> createSchemaCAdapter() {
    return std::make_unique<SchemaCAdapter>();
}