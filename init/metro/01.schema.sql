CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE household (
  household_id UUID PRIMARY KEY,
  county_fips TEXT NOT NULL,
  hh_size INT,
  income_usd NUMERIC,
  housing_type TEXT,
  created_at_utc TIMESTAMP
);

CREATE TABLE address (
  address_id UUID PRIMARY KEY,
  street TEXT,
  city TEXT,
  state TEXT,
  zip TEXT
);

CREATE TABLE resident (
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

CREATE INDEX idx_household_county ON household(county_fips);
CREATE INDEX idx_resident_household ON resident(household_id);
