-- age.sql (Schema C)
-- Returns mergeable age partials and histogram buckets.
-- Age computed at :end_date from JSON dob.

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

    CASE
      WHEN (b.profile_json #>> '{demographics,dob}') ~ '^\d{4}-\d{2}-\d{2}$'
        THEN (b.profile_json #>> '{demographics,dob}')::DATE
      ELSE NULL
    END AS dob_date,
    (b.profile_json #>> '{demographics,dob}') AS dob_raw,

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
with_age AS (
  SELECT
    e.*,
    EXTRACT(YEAR FROM age((SELECT end_date FROM params), e.dob_date))::INT AS age_years
  FROM extracted e
  WHERE e.dob_date IS NOT NULL
),
labeled AS (
  SELECT
    w.*,
    (w.doc_id IS NULL OR w.county_fips IS NULL OR w.county_fips = '') AS missing_required,
    (w.verification_source IS NULL OR w.verification_source = '') AS missing_verification,

    (w.dob_raw IS NOT NULL AND w.dob_raw <> '' AND w.dob_date IS NULL) AS invalid_dob_format,
    (w.moved_in_raw IS NOT NULL AND w.moved_in_raw <> '' AND w.moved_in_date IS NULL) AS invalid_moved_in_format,
    (w.move_out_raw IS NOT NULL AND w.move_out_raw <> '' AND w.move_out_date IS NULL) AS invalid_move_out_format,
    (w.death_raw IS NOT NULL AND w.death_raw <> '' AND w.death_date IS NULL) AS invalid_death_format,

    (w.moved_in_date IS NULL OR w.moved_in_date > (SELECT end_date FROM params)) AS invalid_moved_in,
    (w.move_out_date IS NOT NULL AND w.move_out_date < (SELECT start_date FROM params)) AS invalid_move_out,
    (w.death_date IS NOT NULL AND w.death_date <= (SELECT end_date FROM params)) AS invalid_death,

    (w.age_years < 0 OR w.age_years > 120) AS invalid_age
  FROM with_age w
),
valid AS (
  SELECT
    l.*,
    NOT (
      l.missing_required OR l.missing_verification OR
      l.invalid_dob_format OR l.invalid_moved_in_format OR l.invalid_move_out_format OR l.invalid_death_format OR
      l.invalid_moved_in OR l.invalid_move_out OR l.invalid_death OR
      l.invalid_age
    ) AS is_valid,
    CASE
      WHEN l.missing_required THEN 'missing_required'
      WHEN l.missing_verification THEN 'missing_verification'
      WHEN l.invalid_dob_format OR l.invalid_moved_in_format OR l.invalid_move_out_format OR l.invalid_death_format THEN 'invalid_date'
      WHEN l.invalid_moved_in OR l.invalid_move_out OR l.invalid_death THEN 'inconsistent_residency'
      WHEN l.invalid_age THEN 'invalid_age'
      ELSE NULL
    END AS primary_drop_reason
  FROM labeled l
),
agg AS (
  SELECT
    (SELECT COUNT(*) FROM base)::BIGINT AS rows_scanned,
    COUNT(*) FILTER (WHERE is_valid)::BIGINT AS rows_used,
    ((SELECT COUNT(*) FROM base) - COUNT(*) FILTER (WHERE is_valid))::BIGINT AS rows_dropped,

    COUNT(*) FILTER (WHERE primary_drop_reason = 'missing_required')::BIGINT AS drop_missing_required,
    COUNT(*) FILTER (WHERE primary_drop_reason = 'missing_verification')::BIGINT AS drop_missing_verification,
    COUNT(*) FILTER (WHERE primary_drop_reason = 'invalid_date')::BIGINT AS drop_invalid_date,
    COUNT(*) FILTER (WHERE primary_drop_reason = 'inconsistent_residency')::BIGINT AS drop_inconsistent_residency,
    COUNT(*) FILTER (WHERE primary_drop_reason = 'invalid_age')::BIGINT AS drop_invalid_age,

    0::BIGINT AS drop_negative_income,
    0::BIGINT AS drop_other,

    SUM(age_years) FILTER (WHERE is_valid)::BIGINT AS age_sum,
    COUNT(*) FILTER (WHERE is_valid)::BIGINT AS age_count,
    MIN(age_years) FILTER (WHERE is_valid)::INT AS age_min,
    MAX(age_years) FILTER (WHERE is_valid)::INT AS age_max,

    COUNT(*) FILTER (WHERE is_valid AND age_years BETWEEN 0 AND 4)::BIGINT AS age_0_4,
    COUNT(*) FILTER (WHERE is_valid AND age_years BETWEEN 5 AND 9)::BIGINT AS age_5_9,
    COUNT(*) FILTER (WHERE is_valid AND age_years BETWEEN 10 AND 14)::BIGINT AS age_10_14,
    COUNT(*) FILTER (WHERE is_valid AND age_years BETWEEN 15 AND 17)::BIGINT AS age_15_17,
    COUNT(*) FILTER (WHERE is_valid AND age_years BETWEEN 18 AND 24)::BIGINT AS age_18_24,
    COUNT(*) FILTER (WHERE is_valid AND age_years BETWEEN 25 AND 34)::BIGINT AS age_25_34,
    COUNT(*) FILTER (WHERE is_valid AND age_years BETWEEN 35 AND 44)::BIGINT AS age_35_44,
    COUNT(*) FILTER (WHERE is_valid AND age_years BETWEEN 45 AND 54)::BIGINT AS age_45_54,
    COUNT(*) FILTER (WHERE is_valid AND age_years BETWEEN 55 AND 64)::BIGINT AS age_55_64,
    COUNT(*) FILTER (WHERE is_valid AND age_years BETWEEN 65 AND 74)::BIGINT AS age_65_74,
    COUNT(*) FILTER (WHERE is_valid AND age_years BETWEEN 75 AND 84)::BIGINT AS age_75_84,
    COUNT(*) FILTER (WHERE is_valid AND age_years >= 85)::BIGINT AS age_85_plus
  FROM valid
)
SELECT * FROM agg;