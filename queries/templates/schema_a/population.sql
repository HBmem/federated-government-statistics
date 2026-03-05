-- population.sql (Schema A)
-- Returns population counts for residents valid in the requested timeframe.

WITH params AS (
  SELECT
    CAST(:start_date AS DATE) AS start_date,
    CAST(:end_date   AS DATE) AS end_date
),
base AS (
  SELECT
    h.county_fips,
    r.resident_id,
    r.dob,
    r.active_status,
    r.moved_in_date,
    r.move_out_date,
    r.death_date,
    r.verification_source
  FROM resident r
  JOIN household h ON h.household_id = r.household_id
),
validated AS (
  SELECT
    b.*,

    (b.county_fips IS NULL OR b.resident_id IS NULL) AS missing_required,
    (b.verification_source IS NULL OR b.verification_source = '') AS missing_verification,

    (b.moved_in_date IS NULL OR b.moved_in_date > (SELECT end_date FROM params)) AS invalid_moved_in,
    (b.move_out_date IS NOT NULL AND b.move_out_date < (SELECT start_date FROM params)) AS invalid_move_out,
    (b.death_date IS NOT NULL AND b.death_date <= (SELECT end_date FROM params)) AS invalid_death
  FROM base b
),
labeled AS (
  SELECT
    v.*,
    NOT (
      v.missing_required OR
      v.missing_verification OR
      v.invalid_moved_in OR
      v.invalid_move_out OR
      v.invalid_death
    ) AS is_valid,
    CASE
      WHEN v.missing_required THEN 'missing_required'
      WHEN v.missing_verification THEN 'missing_verification'
      WHEN v.invalid_moved_in OR v.invalid_move_out OR v.invalid_death THEN 'inconsistent_residency'
      ELSE NULL
    END AS primary_drop_reason
  FROM validated v
),
agg AS (
  SELECT
    COUNT(*)::BIGINT AS rows_scanned,
    COUNT(*) FILTER (WHERE is_valid)::BIGINT AS rows_used,
    COUNT(*) FILTER (WHERE NOT is_valid)::BIGINT AS rows_dropped,

    COUNT(*) FILTER (WHERE primary_drop_reason = 'missing_required')::BIGINT AS drop_missing_required,
    COUNT(*) FILTER (WHERE primary_drop_reason = 'missing_verification')::BIGINT AS drop_missing_verification,
    COUNT(*) FILTER (WHERE primary_drop_reason = 'inconsistent_residency')::BIGINT AS drop_inconsistent_residency,

    0::BIGINT AS drop_invalid_date,
    0::BIGINT AS drop_invalid_age,
    0::BIGINT AS drop_negative_income,
    0::BIGINT AS drop_other,

    -- Metric outputs
    COUNT(*) FILTER (WHERE is_valid)::BIGINT AS population_count
  FROM labeled
)
SELECT * FROM agg;