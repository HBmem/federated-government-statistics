-- 02_load.sql (metro)
-- Loads CSV data files shipped alongside init scripts.

-- These CSV files are expected to be present in the init directory mounted at:
-- /docker-entrypoint-initdb.d

-- Load address rows.
COPY address(address_id, street, city, state, zip)
FROM '/docker-entrypoint-initdb.d/address.csv'
WITH (FORMAT csv, HEADER true);

-- Load household rows.
COPY household(household_id, county_fips, hh_size, income_usd, housing_type, created_at_utc)
FROM '/docker-entrypoint-initdb.d/household.csv'
WITH (FORMAT csv, HEADER true);

-- Load resident rows.
-- Empty strings in date columns will be interpreted as NULL if the CSV uses empty fields.
-- If your generator uses empty strings, Postgres COPY treats them as empty string by default;
-- for DATE columns, empty string is not valid. To handle that, we use NULL ''.
COPY resident(
  resident_id, household_id, first_name, last_name, dob,
  sex, race, ethnicity, address_id,
  moved_in_date, move_out_date, death_date, active_status,
  last_verified_date, verification_source,
  employment_status, has_job_flag
)
FROM '/docker-entrypoint-initdb.d/resident.csv'
WITH (FORMAT csv, HEADER true, NULL '');

-- Analyze to improve query planning for the first runs.
ANALYZE address;
ANALYZE household;
ANALYZE resident;