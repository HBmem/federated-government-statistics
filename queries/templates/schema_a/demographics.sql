-- demographics.sql (Schema A)
-- Returns mergeable demographic counts for valid residents in timeframe:
-- - counts by sex, race, ethnicity

WITH params AS (
  SELECT CAST(:start_date AS DATE) AS start_date, CAST(:end_date AS DATE) AS end_date
),
base AS (
  SELECT
    h.county_fips,
    r.resident_id,
    r.sex,
    r.race,
    r.ethnicity,
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
    (b.death_date IS NOT NULL AND b.death_date <= (SELECT end_date FROM params)) AS invalid_death
  FROM base b
),
valid AS (
  SELECT
    *,
    NOT (missing_required OR missing_verification OR invalid_moved_in OR invalid_move_out OR invalid_death) AS is_valid,
    CASE
      WHEN missing_required THEN 'missing_required'
      WHEN missing_verification THEN 'missing_verification'
      WHEN invalid_moved_in OR invalid_move_out OR invalid_death THEN 'inconsistent_residency'
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

    0::BIGINT AS drop_invalid_date,
    0::BIGINT AS drop_invalid_age,
    0::BIGINT AS drop_negative_income,
    0::BIGINT AS drop_other,

    -- Sex counts
    COUNT(*) FILTER (WHERE is_valid AND sex = 'M')::BIGINT AS sex_m,
    COUNT(*) FILTER (WHERE is_valid AND sex = 'F')::BIGINT AS sex_f,
    COUNT(*) FILTER (WHERE is_valid AND sex = 'X')::BIGINT AS sex_x,

    -- Race counts (add more categories as needed)
    COUNT(*) FILTER (WHERE is_valid AND race = 'White')::BIGINT AS race_white,
    COUNT(*) FILTER (WHERE is_valid AND race = 'Black')::BIGINT AS race_black,
    COUNT(*) FILTER (WHERE is_valid AND race = 'Asian')::BIGINT AS race_asian,
    COUNT(*) FILTER (WHERE is_valid AND race = 'Pacific')::BIGINT AS race_pacific,
    COUNT(*) FILTER (WHERE is_valid AND race = 'Native')::BIGINT AS race_native,
    COUNT(*) FILTER (WHERE is_valid AND race NOT IN ('White','Black','Asian','Pacific','Native'))::BIGINT AS race_other,

    -- Ethnicity counts
    COUNT(*) FILTER (WHERE is_valid AND ethnicity = 'Hispanic')::BIGINT AS eth_hispanic,
    COUNT(*) FILTER (WHERE is_valid AND ethnicity <> 'Hispanic')::BIGINT AS eth_non_hispanic
  FROM valid
)
SELECT * FROM agg;