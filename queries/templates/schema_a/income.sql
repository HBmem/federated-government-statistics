-- income.sql (Schema A)
-- Returns mergeable income partials and bucket histogram for valid residents in timeframe.
-- Buckets match Config/buckets.json income_usd.

WITH params AS (
  SELECT CAST(:start_date AS DATE) AS start_date, CAST(:end_date AS DATE) AS end_date
),
base AS (
  SELECT
    h.county_fips,
    r.resident_id,
    h.income_usd,
    r.moved_in_date,
    r.move_out_date,
    r.death_date,
    r.verification_source
  FROM resident r
  JOIN household h ON h.household_id = r.household_id
),
labeled AS (
  SELECT
    b.*,

    (b.county_fips IS NULL OR b.resident_id IS NULL) AS missing_required,
    (b.verification_source IS NULL OR b.verification_source = '') AS missing_verification,

    (b.moved_in_date IS NULL OR b.moved_in_date > (SELECT end_date FROM params)) AS invalid_moved_in,
    (b.move_out_date IS NOT NULL AND b.move_out_date < (SELECT start_date FROM params)) AS invalid_move_out,
    (b.death_date IS NOT NULL AND b.death_date <= (SELECT end_date FROM params)) AS invalid_death,

    (b.income_usd IS NULL) AS missing_income,
    (b.income_usd IS NOT NULL AND b.income_usd < 0) AS negative_income
  FROM base b
),
valid AS (
  SELECT
    *,
    NOT (
      missing_required OR missing_verification OR
      invalid_moved_in OR invalid_move_out OR invalid_death OR
      missing_income OR negative_income
    ) AS is_valid,
    CASE
      WHEN missing_required THEN 'missing_required'
      WHEN missing_verification THEN 'missing_verification'
      WHEN invalid_moved_in OR invalid_move_out OR invalid_death THEN 'inconsistent_residency'
      WHEN missing_income THEN 'missing_required'
      WHEN negative_income THEN 'negative_income'
      ELSE NULL
    END AS primary_drop_reason
  FROM labeled
),
agg AS (
  SELECT
    COUNT(*)::BIGINT AS rows_scanned,
    COUNT(*) FILTER (WHERE is_valid)::BIGINT AS rows_used,
    COUNT(*) FILTER (WHERE NOT is_valid)::BIGINT AS rows_dropped,

    COUNT(*) FILTER (WHERE primary_drop_reason = 'missing_required')::BIGINT AS drop_missing_required,
    COUNT(*) FILTER (WHERE primary_drop_reason = 'missing_verification')::BIGINT AS drop_missing_verification,
    COUNT(*) FILTER (WHERE primary_drop_reason = 'inconsistent_residency')::BIGINT AS drop_inconsistent_residency,
    COUNT(*) FILTER (WHERE primary_drop_reason = 'negative_income')::BIGINT AS drop_negative_income,

    0::BIGINT AS drop_invalid_date,
    0::BIGINT AS drop_invalid_age,
    0::BIGINT AS drop_other,

    -- Metric partials
    SUM(income_usd) FILTER (WHERE is_valid)::NUMERIC AS income_sum,
    COUNT(*) FILTER (WHERE is_valid)::BIGINT AS income_count,
    MIN(income_usd) FILTER (WHERE is_valid)::NUMERIC AS income_min,
    MAX(income_usd) FILTER (WHERE is_valid)::NUMERIC AS income_max,

    -- Histogram buckets
    COUNT(*) FILTER (WHERE is_valid AND income_usd BETWEEN 0 AND 24999)::BIGINT AS inc_lt_25k,
    COUNT(*) FILTER (WHERE is_valid AND income_usd BETWEEN 25000 AND 49999)::BIGINT AS inc_25_50k,
    COUNT(*) FILTER (WHERE is_valid AND income_usd BETWEEN 50000 AND 74999)::BIGINT AS inc_50_75k,
    COUNT(*) FILTER (WHERE is_valid AND income_usd BETWEEN 75000 AND 99999)::BIGINT AS inc_75_100k,
    COUNT(*) FILTER (WHERE is_valid AND income_usd BETWEEN 100000 AND 149999)::BIGINT AS inc_100_150k,
    COUNT(*) FILTER (WHERE is_valid AND income_usd BETWEEN 150000 AND 199999)::BIGINT AS inc_150_200k,
    COUNT(*) FILTER (WHERE is_valid AND income_usd >= 200000)::BIGINT AS inc_200k_plus
  FROM valid
)
SELECT * FROM agg;