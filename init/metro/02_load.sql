\copy household FROM '/docker-entrypoint-initdb.d/household.csv' WITH (FORMAT csv, HEADER true);
\copy address   FROM '/docker-entrypoint-initdb.d/address.csv'   WITH (FORMAT csv, HEADER true);
\copy resident  FROM '/docker-entrypoint-initdb.d/resident.csv'  WITH (FORMAT csv, HEADER true);
