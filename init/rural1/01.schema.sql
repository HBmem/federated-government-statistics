-- 01_schema.sql (rural1)
-- Creates Schema C (single-table with JSONB) for JSON extraction workloads.

CREATE TABLE IF NOT EXISTS citizen (
  doc_id TEXT PRIMARY KEY,
  county_fips TEXT,
  profile_json JSONB,
  created_at_utc TIMESTAMP
);

-- Indexes useful for schema discovery and query performance.
CREATE INDEX IF NOT EXISTS idx_citizen_county ON citizen(county_fips);

-- GIN index speeds up JSON path/key queries when extracting nested fields.
CREATE INDEX IF NOT EXISTS idx_citizen_profile_gin ON citizen USING GIN (profile_json);