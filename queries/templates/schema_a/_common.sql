-- _common.sql (Schema A)
-- Shared CTE logic used by Schema A metric templates.
-- Parameter placeholders expected:
--   :start_date (YYYY-MM-DD)
--   :end_date   (YYYY-MM-DD)

-- Notes:
-- - This file is intended to be pasted into templates, not executed alone.
-- - Workers may implement file concatenation, but if not, we duplicate the block.

WITH params AS (
  SELECT
    CAST(:start_date AS DATE) AS start_date,
    CAST(:end_date   AS DATE) AS end_date
),
base AS (
  -- Base scan: join resident to household for county_fips.
  SELECT
    h.county_fips,
    r.resident_id,
    r.dob,
    r.sex,
    r.race,
    r.ethnicity,
    h.hh_size,
    h.income_usd,
    r.employment_status,
    r.has_job_flag,
    r.active_status,
    r.moved_in_date,
    r.move_out_date,
    r.death_date,
    r.last_verified_date,
    r.verification_source
  FROM resident r
  JOIN household h ON h.household_id = r.household_id
),
validated AS (
  -- Compute drop reasons with a simple priority order.
  SELECT
    b.*,
    (SELECT start_date FROM params) AS start_date,
    (SELECT end_date FROM params)   AS end_date,

    -- Required fields checks
    (b.county_fips IS NULL OR b.resident_id IS NULL) AS missing_required,
    (b.verification_source IS NULL OR b.verification_source = '') AS missing_verification,

    -- Residency timeframe logic
    (
      b.moved_in_date IS NULL OR
      b.moved_in_date > (SELECT end_date FROM params)
    ) AS invalid_moved_in,

    (
      b.move_out_date IS NOT NULL AND
      b.move_out_date < (SELECT start_date FROM params)
    ) AS invalid_move_out,

    (
      b.death_date IS NOT NULL AND
      b.death_date <= (SELECT end_date FROM params)
    ) AS invalid_death,

    -- Age sanity check (computed at end_date)
    (
      b.dob IS NULL OR
      EXTRACT(YEAR FROM age((SELECT end_date FROM params), b.dob)) < 0 OR
      EXTRACT(YEAR FROM age((SELECT end_date FROM params), b.dob)) > 120
    ) AS invalid_age,

    -- Income sanity check
    (
      b.income_usd IS NOT NULL AND b.income_usd < 0
    ) AS negative_income
  FROM base b
),
labeled AS (
  SELECT
    v.*,

    -- One canonical "is_valid" for metrics that require full validity.
    -- Some metrics may choose to be less strict, but default is strict.
    NOT (
      v.missing_required OR
      v.missing_verification OR
      v.invalid_moved_in OR
      v.invalid_move_out OR
      v.invalid_death OR
      v.invalid_age OR
      v.negative_income
    ) AS is_valid,

    -- Drop reason bucket counters (non-exclusive counters are fine, but we track a primary reason too).
    CASE
      WHEN v.missing_required THEN 'missing_required'
      WHEN v.missing_verification THEN 'missing_verification'
      WHEN v.invalid_moved_in OR v.invalid_move_out OR v.invalid_death THEN 'inconsistent_residency'
      WHEN v.invalid_age THEN 'invalid_age'
      WHEN v.negative_income THEN 'negative_income'
      ELSE NULL
    END AS primary_drop_reason
  FROM validated v
)