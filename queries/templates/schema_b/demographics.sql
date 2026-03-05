-- demographics.sql (Schema B)
-- Mergeable counts by gender_code, race_code, eth_flag for valid residents in timeframe.

WITH params AS (
  SELECT CAST(:start_date AS DATE) AS start_date, CAST(:end_date AS DATE) AS end_date
),
base AS (
  SELECT
    person_uuid,
    county,
    gender_code,
    race_code,
    eth_flag,
    moved_in_str,
    move_out_str,
    death_str,
    verification_source
  FROM person_record
),
parsed AS (
  SELECT
    b.*,
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
labeled AS (
  SELECT
    p.*,
    (p.person_uuid IS NULL OR p.county IS NULL OR p.county = '') AS missing_required,
    (p.verification_source IS NULL OR p.verification_source = '') AS missing_verification,

    (p.moved_in_str IS NOT NULL AND p.moved_in_str <> '' AND p.moved_in_date IS NULL) AS invalid_moved_in_format,
    (p.move_out_str IS NOT NULL AND p.move_out_str <> '' AND p.move_out_date IS NULL) AS invalid_move_out_format,
    (p.death_str IS NOT NULL AND p.death_str <> '' AND p.death_date IS NULL) AS invalid_death_format,

    (p.moved_in_date IS NULL OR p.moved_in_date > (SELECT end_date FROM params)) AS invalid_moved_in,
    (p.move_out_date IS NOT NULL AND p.move_out_date < (SELECT start_date FROM params)) AS invalid_move_out,
    (p.death_date IS NOT NULL AND p.death_date <= (SELECT end_date FROM params)) AS invalid_death
  FROM parsed p
),
valid AS (
  SELECT
    l.*,
    NOT (
      l.missing_required OR l.missing_verification OR
      l.invalid_moved_in_format OR l.invalid_move_out_format OR l.invalid_death_format OR
      l.invalid_moved_in OR l.invalid_move_out OR l.invalid_death
    ) AS is_valid,
    CASE
      WHEN l.missing_required THEN 'missing_required'
      WHEN l.missing_verification THEN 'missing_verification'
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

    0::BIGINT AS drop_invalid_age,
    0::BIGINT AS drop_negative_income,
    0::BIGINT AS drop_other,

    -- Gender counts
    COUNT(*) FILTER (WHERE is_valid AND gender_code = 'M')::BIGINT AS sex_m,
    COUNT(*) FILTER (WHERE is_valid AND gender_code = 'F')::BIGINT AS sex_f,
    COUNT(*) FILTER (WHERE is_valid AND gender_code = 'X')::BIGINT AS sex_x,

    -- Race code counts
    COUNT(*) FILTER (WHERE is_valid AND race_code = 'W')::BIGINT AS race_white,
    COUNT(*) FILTER (WHERE is_valid AND race_code = 'B')::BIGINT AS race_black,
    COUNT(*) FILTER (WHERE is_valid AND race_code = 'A')::BIGINT AS race_asian,
    COUNT(*) FILTER (WHERE is_valid AND race_code = 'P')::BIGINT AS race_pacific,
    COUNT(*) FILTER (WHERE is_valid AND race_code = 'N')::BIGINT AS race_native,
    COUNT(*) FILTER (WHERE is_valid AND race_code NOT IN ('W','B','A','P','N'))::BIGINT AS race_other,

    -- Ethnicity flag counts
    COUNT(*) FILTER (WHERE is_valid AND eth_flag = 'Y')::BIGINT AS eth_hispanic,
    COUNT(*) FILTER (WHERE is_valid AND eth_flag <> 'Y')::BIGINT AS eth_non_hispanic
  FROM valid
)
SELECT * FROM agg;