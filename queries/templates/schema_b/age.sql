-- age.sql (Schema B)
-- Returns mergeable age partials and histogram buckets.
-- DOB parsing assumes dob_str is mostly ISO, but bad data may break it.

WITH params AS (
  SELECT CAST(:start_date AS DATE) AS start_date, CAST(:end_date AS DATE) AS end_date
),
base AS (
  SELECT
    person_uuid,
    county,
    dob_str,
    moved_in_str,
    move_out_str,
    death_str,
    verification_source
  FROM person_record
),
parsed AS (
  SELECT
    b.*,

    -- Parse DOB: allow ISO and common US formats.
    CASE
      WHEN dob_str ~ '^\d{4}-\d{2}-\d{2}$' THEN to_date(dob_str, 'YYYY-MM-DD')
      WHEN dob_str ~ '^\d{2}/\d{2}/\d{4}$' THEN to_date(dob_str, 'MM/DD/YYYY')
      WHEN dob_str ~ '^\d{2}-\d{2}-\d{4}$' THEN to_date(dob_str, 'MM-DD-YYYY')
      ELSE NULL
    END AS dob_date,

    CASE
      WHEN moved_in_str ~ '^\d{4}-\d{2}-\d{2}$' THEN to_date(moved_in_str, 'YYYY-MM-DD')
      WHEN moved_in_str ~ '^\d{2}/\d{2}/\d{4}$' THEN to_date(moved_in_str, 'MM/DD/YYYY')
      WHEN moved_in_str ~ '^\d{2}-\d{2}-\d{4}$' THEN to_date(moved_in_str, 'MM-DD-YYYY')
      ELSE NULL
    END AS moved_in_date,

    CASE
      WHEN move_out_str ~ '^\d{4}-\d{2}-\d{2}$' THEN to_date(move_out_str, 'YYYY-MM-DD')
      WHEN move_out_str ~ '^\d{2}/\d{2}/\d{4}$' THEN to_date(move_out_str, 'MM/DD/YYYY')
      WHEN move_out_str ~ '^\d{2}-\d{2}-\d{4}$' THEN to_date(move_out_str, 'MM-DD-YYYY')
      ELSE NULL
    END AS move_out_date,

    CASE
      WHEN death_str ~ '^\d{4}-\d{2}-\d{2}$' THEN to_date(death_str, 'YYYY-MM-DD')
      WHEN death_str ~ '^\d{2}/\d{2}/\d{4}$' THEN to_date(death_str, 'MM/DD/YYYY')
      WHEN death_str ~ '^\d{2}-\d{2}-\d{4}$' THEN to_date(death_str, 'MM-DD-YYYY')
      ELSE NULL
    END AS death_date
  FROM base b
),
with_age AS (
  SELECT
    p.*,
    EXTRACT(YEAR FROM age((SELECT end_date FROM params), p.dob_date))::INT AS age_years
  FROM parsed p
  WHERE p.dob_date IS NOT NULL
),
labeled AS (
  SELECT
    w.*,

    (w.person_uuid IS NULL OR w.county IS NULL OR w.county = '') AS missing_required,
    (w.verification_source IS NULL OR w.verification_source = '') AS missing_verification,

    (w.dob_str IS NOT NULL AND w.dob_str <> '' AND w.dob_date IS NULL) AS invalid_dob_format,
    (w.moved_in_str IS NOT NULL AND w.moved_in_str <> '' AND w.moved_in_date IS NULL) AS invalid_moved_in_format,
    (w.move_out_str IS NOT NULL AND w.move_out_str <> '' AND w.move_out_date IS NULL) AS invalid_move_out_format,
    (w.death_str IS NOT NULL AND w.death_str <> '' AND w.death_date IS NULL) AS invalid_death_format,

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