-- Same schema as rural1 for consistent comparisons.

CREATE TABLE IF NOT EXISTS citizen (
  doc_id TEXT PRIMARY KEY,
  county_fips TEXT,
  profile_json JSONB,
  created_at_utc TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_citizen_county ON citizen(county_fips);
CREATE INDEX IF NOT EXISTS idx_citizen_profile_gin ON citizen USING GIN (profile_json);