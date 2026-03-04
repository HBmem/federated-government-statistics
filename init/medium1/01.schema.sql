-- 01_schema.sql (medium1)
-- Creates Schema B (single-table) for scan-heavy workloads with string parsing.

CREATE TABLE IF NOT EXISTS person_record (
  person_uuid UUID PRIMARY KEY,
  county TEXT,
  dob_str TEXT,
  gender_code TEXT,
  race_code TEXT,
  eth_flag TEXT,
  income_bracket TEXT,
  household_size TEXT,
  full_address TEXT,
  updated_ts_utc TIMESTAMP,
  moved_in_str TEXT,
  move_out_str TEXT,
  death_str TEXT,
  active_flag TEXT,
  last_verified_str TEXT,
  verification_source TEXT,
  employment_status TEXT,
  has_job_flag TEXT
);

-- Indexes that reflect typical query patterns in the project.
CREATE INDEX IF NOT EXISTS idx_person_county ON person_record(county);
CREATE INDEX IF NOT EXISTS idx_person_updated ON person_record(updated_ts_utc);
CREATE INDEX IF NOT EXISTS idx_person_active ON person_record(active_flag);