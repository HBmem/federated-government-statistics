-- income.sql (Schema C)
-- Returns mergeable numeric income partials and histogram buckets.

WITH params AS (
  SELECT CAST(:start_date AS DATE) AS start_date, CAST(:end_date AS DATE) AS end_date
),
base AS (
  SELECT doc_id, county_fips, profile_json FROM citizen
),
extracted AS (
  SELECT
    b.doc_id,
    b.county_fips,
    NULLIF(b.profile_json #>> '{residency,verification_source}', '') AS verification_source,

    -- Income extraction (numeric)
    CASE
      WHEN (b.profile_json #>> '{household,income_usd}') ~ '^-?\d+(\.\d+)?$'
        THEN (b.profile_json #>> '{household,income_usd}')::NUMERIC
      ELSE NULL
    END AS income_usd,
    (b.profile_json #>> '{household,income_usd}') AS income_raw,

    CASE
      WHEN (b.profile_json #>> '{residency,moved_in_date}') ~ '^\d{4}-\d{2}-\d{2}$'
        THEN (b.profile_json #>> '{residency,moved_in_date}')::DATE
      ELSE NULL
    END AS moved_in_date,
    CASE
      WHEN (b.profile_json #>> '{residency,move_out_date}') ~ '^\d{4}-\d{2}-\d{2}$'
        THEN (b.profile_json #>> '{residency,move_out_date}')::DATE
      ELSE NULL
    END AS move_out_date,
    CASE
      WHEN (b.profile_json #>> '{residency,death_date}') ~ '^\d{4}-\d{2}-\d{2}$'
        THEN (b.profile_json #>> '{residency,death_date}')::DATE
      ELSE NULL
    END AS death_date,

    (b.profile_json #>> '{residency,moved_in_date}') AS moved_in_raw,
    (b.profile_json #>> '{residency,move_out_date}') AS move_out_raw,
    (b.profile_json #>> '{residency,death_date}') AS death_raw
  FROM base b
),
labeled AS (
  SELECT
    e.*,
    (e.doc_id IS NULL OR e.county_fips IS NULL OR e.county_fips = '') AS missing_required,
    (e.verification_source IS NULL OR e.verification_source = '') AS missing_verification,

    (e.income_raw IS NOT NULL AND e.income_raw <> '' AND e.income_usd IS NULL) AS invalid_income_format,
    (e.income_usd IS NOT NULL AND e.income_usd < 0) AS negative_income,

    (e.moved_in_raw IS NOT NULL AND e.moved_in_raw <> '' AND e.moved_in_date IS NULL) AS invalid_moved_in_format,
    (e.move_out_raw IS NOT NULL AND e.move_out_raw <> '' AND e.move_out_date IS NULL) AS invalid_move_out_format,
    (e.death_raw IS NOT NULL AND e.death_raw <> '' AND e.death_date IS NULL) AS invalid_death_format,

    (e.moved_in_date IS NULL OR e.moved_in_date > (SELECT end_date FROM params)) AS invalid_moved_in,
    (e.move_out_date IS NOT NULL AND e.move_out_date < (SELECT start_date FROM params)) AS invalid_move_out,
    (e.death_date IS NOT NULL AND e.death_date <= (SELECT end_date FROM params)) AS invalid_death
  FROM extracted e
),
valid AS (
  SELECT
    l.*,
    NOT (
      l.missing_required OR l.missing_verification OR
      l.invalid_income_format OR l.negative_income OR
      l.invalid_moved_in_format OR l.invalid_move_out_format OR l.invalid_death_format OR
      l.invalid_moved_in OR l.invalid_move_out OR l.invalid_death
    ) AS is_valid,
    CASE
      WHEN l.missing_required THEN 'missing_required'
      WHEN l.missing_verification THEN 'missing_verification'
      WHEN l.invalid_income_format THEN 'invalid_date'
      WHEN l.negative_income THEN 'negative_income'
      WHEN l.invalid_moved_in_format OR l.invalid_move_out_format OR l.invalid_death_format THEN 'invalid_date'
      WHEN l.invalid_moved_in OR l.invalid_move_out OR l.invalid_death THEN 'inconsistent_residency'
      ELSE NULL
    END AS primary_drop_reason
  FROM labeled l
),
agg AS (
  SELECT
    COUNT(*)::BIGINT AS rows_scanned,
    COUNT(*) FILTER (WHERE is_valid)::BIGINT AS rows_used,
    COUNT(*) FILTER (WHERE NOT is_valid)::BIGINT AS rows_dropped,

    COUNT(*) FILTER (WHERE primary_drop_reason = 'missing_required')::BIGINT AS drop_missing_required,
    COUNT(*) FILTER (WHERE primary_drop_reason = 'missing_verification')::BIGINT AS drop_missing_verification,
    COUNT(*) FILTER (WHERE primary_drop_reason = 'invalid_date')::BIGINT AS drop_invalid_date,
    COUNT(*) FILTER (WHERE primary_drop_reason = 'inconsistent_residency')::BIGINT AS drop_inconsistent_residency,
    COUNT(*) FILTER (WHERE primary_drop_reason = 'negative_income')::BIGINT AS drop_negative_income,

    0::BIGINT AS drop_invalid_age,
    0::BIGINT AS drop_other,

    SUM(income_usd) FILTER (WHERE is_valid)::NUMERIC AS income_sum,
    COUNT(*) FILTER (WHERE is_valid)::BIGINT AS income_count,
    MIN(income_usd) FILTER (WHERE is_valid)::NUMERIC AS income_min,
    MAX(income_usd) FILTER (WHERE is_valid)::NUMERIC AS income_max,

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