-- Loads Schema C CSV into citizen.

COPY citizen(doc_id, county_fips, profile_json, created_at_utc)
FROM '/docker-entrypoint-initdb.d/citizen.csv'
WITH (FORMAT csv, HEADER true, NULL '', QUOTE '"', ESCAPE '"');

ANALYZE citizen;