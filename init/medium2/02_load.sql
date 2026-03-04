-- 02_load.sql (medium2)
-- Loads Schema B CSV into person_record.

COPY person_record(
  person_uuid, county, dob_str, gender_code, race_code, eth_flag,
  income_bracket, household_size, full_address, updated_ts_utc,
  moved_in_str, move_out_str, death_str, active_flag,
  last_verified_str, verification_source, employment_status, has_job_flag
)
FROM '/docker-entrypoint-initdb.d/personal_record.csv'
WITH (FORMAT csv, HEADER true, NULL '');

ANALYZE person_record;