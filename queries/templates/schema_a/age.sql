-- age.sql (Schema A)
-- Returns mergeable age partials and bucket histogram for valid residents in timeframe.
-- Age is computed at :end_date.

WITH params AS (
  SELECT CAST(:start_date AS DATE) AS start_date, CAST(:end_date AS DATE) AS end_date
),
base AS (
  SELECT
    h.county_fips,
    r.resident_id,
    r.dob,
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
    (b.dob IS NULL) AS missing_dob,

    (b.moved_in_date IS NULL OR b.moved_in_date > (SELECT end_date FROM params)) AS invalid_moved_in,
    (b.move_out_date IS NOT NULL AND b.move_out_date < (SELECT start_date FROM params)) AS invalid_move_out,
    (b.death_date IS NOT NULL AND b.death_date <= (SELECT end_date FROM params)) AS invalid_death
  FROM base b
),
with_age AS (
  SELECT
    l.*,
    EXTRACT(YEAR FROM age((SELECT end_date FROM params), l.dob))::INT AS age_years
  FROM labeled l
  WHERE l.dob IS NOT NULL
),
valid AS (
  SELECT
    w.*,
    (w.age_years < 0 OR w.age_years > 120) AS invalid_age,
    NOT (
      w.missing_required OR w.missing_verification OR w.missing_dob OR
      w.invalid_moved_in OR w.invalid_move_out OR w.invalid_death OR
      (w.age_years < 0 OR w.age_years > 120)
    ) AS is_valid,
    CASE
      WHEN w.missing_required OR w.missing_dob THEN 'missing_required'
      WHEN w.missing_verification THEN 'missing_verification'
      WHEN w.invalid_moved_in OR w.invalid_move_out OR w.invalid_death THEN 'inconsistent_residency'
      WHEN (w.age_years < 0 OR w.age_years > 120) THEN 'invalid_age'
      ELSE NULL
    END AS primary_drop_reason
  FROM with_age w
),
agg AS (
  SELECT
    (SELECT COUNT(*) FROM base)::BIGINT AS rows_scanned,
    COUNT(*) FILTER (WHERE is_valid)::BIGINT AS rows_used,
    ((SELECT COUNT(*) FROM base) - COUNT(*) FILTER (WHERE is_valid))::BIGINT AS rows_dropped,

    COUNT(*) FILTER (WHERE primary_drop_reason = 'missing_required')::BIGINT AS drop_missing_required,
    COUNT(*) FILTER (WHERE primary_drop_reason = 'missing_verification')::BIGINT AS drop_missing_verification,
    COUNT(*) FILTER (WHERE primary_drop_reason = 'inconsistent_residency')::BIGINT AS drop_inconsistent_residency,
    COUNT(*) FILTER (WHERE primary_drop_reason = 'invalid_age')::BIGINT AS drop_invalid_age,

    0::BIGINT AS drop_invalid_date,
    0::BIGINT AS drop_negative_income,
    0::BIGINT AS drop_other,

    -- Metric partials
    SUM(age_years) FILTER (WHERE is_valid)::BIGINT AS age_sum,
    COUNT(*) FILTER (WHERE is_valid)::BIGINT AS age_count,
    MIN(age_years) FILTER (WHERE is_valid)::INT AS age_min,
    MAX(age_years) FILTER (WHERE is_valid)::INT AS age_max,

    -- Age histogram buckets
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