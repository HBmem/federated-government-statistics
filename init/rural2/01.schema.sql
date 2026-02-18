CREATE TABLE citizen (
  doc_id TEXT PRIMARY KEY,
  county_fips TEXT,
  profile_json JSONB,
  created_at TIMESTAMP
);

CREATE INDEX idx_citizen_county ON citizen(county_fips);
CREATE INDEX idx_profile_gin ON citizen USING GIN (profile_json);
