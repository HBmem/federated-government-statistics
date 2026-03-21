-- Creates Schema A (multi-table) for join-heavy workloads.

-- pgcrypto is used to support UUID utilities if needed.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Household table contains county_fips and household-level attributes.
CREATE TABLE IF NOT EXISTS household (
  household_id UUID PRIMARY KEY,
  county_fips TEXT NOT NULL,
  hh_size INT,
  income_usd NUMERIC,
  housing_type TEXT,
  created_at_utc TIMESTAMP
);

-- Address table stores address information referenced by resident.
CREATE TABLE IF NOT EXISTS address (
  address_id UUID PRIMARY KEY,
  street TEXT,
  city TEXT,
  state TEXT,
  zip TEXT
);

-- Resident table stores person-level attributes and residency/employment fields.
CREATE TABLE IF NOT EXISTS resident (
  resident_id UUID PRIMARY KEY,
  household_id UUID REFERENCES household(household_id),
  first_name TEXT,
  last_name TEXT,
  dob DATE,
  sex CHAR(1),
  race TEXT,
  ethnicity TEXT,
  address_id UUID REFERENCES address(address_id),
  moved_in_date DATE,
  move_out_date DATE,
  death_date DATE,
  active_status BOOLEAN,
  last_verified_date DATE,
  verification_source TEXT,
  employment_status TEXT,
  has_job_flag BOOLEAN
);

-- Indexes to improve filtering/join performance.
CREATE INDEX IF NOT EXISTS idx_household_county ON household(county_fips);
CREATE INDEX IF NOT EXISTS idx_resident_household ON resident(household_id);
CREATE INDEX IF NOT EXISTS idx_resident_active ON resident(active_status);
CREATE INDEX IF NOT EXISTS idx_resident_move_out ON resident(move_out_date);
CREATE INDEX IF NOT EXISTS idx_resident_death ON resident(death_date);