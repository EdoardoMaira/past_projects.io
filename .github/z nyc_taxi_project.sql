-- ============================================================
-- NYC YELLOW TAXI 2024 — FULL SQL PIPELINE (MySQL)
-- Edoardo Maira
--
-- Sections are separated by *****. Run top to bottom.
-- Only thing to change: LOCAL_PATH below (the folder holding the CSVs).
-- ============================================================


-- ************************************************************
-- PART 1 — DATABASE, SCHEMA, BULK LOAD
-- ************************************************************

SET GLOBAL local_infile = 1;

CREATE DATABASE IF NOT EXISTS nyc_taxi;
USE nyc_taxi;

DROP TABLE IF EXISTS yellow_trips;

-- Column order must match the CSV exactly: LOAD DATA fills by position.
CREATE TABLE yellow_trips (
    VendorID                INT,
    tpep_pickup_datetime    DATETIME,
    tpep_dropoff_datetime   DATETIME,
    passenger_count         DOUBLE,
    trip_distance           DOUBLE,
    RatecodeID              DOUBLE,
    store_and_fwd_flag      VARCHAR(1),
    PULocationID            INT,
    DOLocationID            INT,
    payment_type            INT,
    fare_amount             DOUBLE,
    extra                   DOUBLE,
    mta_tax                 DOUBLE,
    tip_amount              DOUBLE,
    tolls_amount            DOUBLE,
    improvement_surcharge   DOUBLE,
    total_amount            DOUBLE,
    congestion_surcharge    DOUBLE,
    airport_fee             DOUBLE
);

-- All 12 months go into the same table. ~3 minutes, ~41M raw rows.
-- Note: local_infile must be enabled on both client and server,
-- and the client timeout raised well above the default 30s.

LOAD DATA LOCAL INFILE 'LOCAL_PATH/yellow_tripdata_2024-01.csv'
INTO TABLE yellow_trips
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 LINES;

LOAD DATA LOCAL INFILE 'LOCAL_PATH/yellow_tripdata_2024-02.csv'
INTO TABLE yellow_trips
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 LINES;

LOAD DATA LOCAL INFILE 'LOCAL_PATH/yellow_tripdata_2024-03.csv'
INTO TABLE yellow_trips
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 LINES;

LOAD DATA LOCAL INFILE 'LOCAL_PATH/yellow_tripdata_2024-04.csv'
INTO TABLE yellow_trips
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 LINES;

LOAD DATA LOCAL INFILE 'LOCAL_PATH/yellow_tripdata_2024-05.csv'
INTO TABLE yellow_trips
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 LINES;

LOAD DATA LOCAL INFILE 'LOCAL_PATH/yellow_tripdata_2024-06.csv'
INTO TABLE yellow_trips
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 LINES;

LOAD DATA LOCAL INFILE 'LOCAL_PATH/yellow_tripdata_2024-07.csv'
INTO TABLE yellow_trips
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 LINES;

LOAD DATA LOCAL INFILE 'LOCAL_PATH/yellow_tripdata_2024-08.csv'
INTO TABLE yellow_trips
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 LINES;

LOAD DATA LOCAL INFILE 'LOCAL_PATH/yellow_tripdata_2024-09.csv'
INTO TABLE yellow_trips
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 LINES;

LOAD DATA LOCAL INFILE 'LOCAL_PATH/yellow_tripdata_2024-10.csv'
INTO TABLE yellow_trips
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 LINES;

LOAD DATA LOCAL INFILE 'LOCAL_PATH/yellow_tripdata_2024-11.csv'
INTO TABLE yellow_trips
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 LINES;

LOAD DATA LOCAL INFILE 'LOCAL_PATH/yellow_tripdata_2024-12.csv'
INTO TABLE yellow_trips
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 LINES;

SHOW WARNINGS;

-- ************************************************************
-- PART 2 — CLEANING
-- Never touch the raw table: build a filtered copy instead.
-- ************************************************************

DROP TABLE IF EXISTS yellow_trips_clean;

CREATE TABLE yellow_trips_clean AS
SELECT * FROM yellow_trips
WHERE tpep_pickup_datetime >= '2024-01-01'
  AND tpep_pickup_datetime <  '2025-01-01'
  AND fare_amount    >= 0
  AND trip_distance  > 0
  AND passenger_count > 0;

-- Sanity checks: 41,169,720 raw vs 35,636,331 clean (~13% dropped).
SELECT COUNT(*) FROM yellow_trips;
SELECT COUNT(*) FROM yellow_trips_clean;

SELECT * FROM yellow_trips_clean LIMIT 100;


-- ************************************************************
-- PART 3 — TAXI ZONE LOOKUP
-- Location IDs are meaningless integers without this table.
-- ************************************************************

DROP TABLE IF EXISTS taxi_zone_lookup;

CREATE TABLE taxi_zone_lookup (
    LocationID    INT,
    Borough       VARCHAR(50),
    Zone          VARCHAR(100),
    service_zone  VARCHAR(50)
);

LOAD DATA LOCAL INFILE 'LOCAL_PATH/taxi_zone_lookup.csv'
INTO TABLE taxi_zone_lookup
FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 LINES;

-- Should be 265 rows.
SELECT COUNT(*) FROM taxi_zone_lookup;

-- The join keys, side by side.
SELECT DOLocationID FROM yellow_trips LIMIT 1000;
SELECT LocationID FROM taxi_zone_lookup;


-- ************************************************************
-- PART 4 — TEMPORAL AND BEHAVIOURAL ANALYSIS
-- ************************************************************

-- Index on pickup time. Run once; every time-based query below leans on it.
CREATE INDEX idx_pickup ON yellow_trips_clean (tpep_pickup_datetime);


-- Q1. Trips per month.
SELECT
    MONTH(tpep_pickup_datetime) AS month,
    COUNT(*)                    AS num_trips
FROM yellow_trips_clean
GROUP BY MONTH(tpep_pickup_datetime)
ORDER BY month;


-- Q2. Trips by weekday (1 = Sunday ... 7 = Saturday).
-- TLC reports Thursday busiest, Monday quietest.
SELECT
    DAYOFWEEK(tpep_pickup_datetime) AS weekday,
    COUNT(*)                        AS num_trips
FROM yellow_trips_clean
GROUP BY DAYOFWEEK(tpep_pickup_datetime)
ORDER BY weekday;


-- Q3. Passenger count. Benchmark: ~78% single-passenger.
SELECT
    passenger_count,
    COUNT(*)                                         AS num_trips,
    ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM yellow_trips_clean
GROUP BY passenger_count
ORDER BY passenger_count;


-- Q4. Payment mix on the clean table (1 = card, 2 = cash).
SELECT
    payment_type,
    COUNT(*)                                         AS num_trips,
    ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM yellow_trips_clean
GROUP BY payment_type
ORDER BY num_trips DESC;


-- Q5. Fare, tip and tip share by payment type.
-- Cash tips never reach the meter, so avg_tip = 0 for cash is an
-- artefact of recording, not behaviour. Note avg_fare is near-identical.
SELECT
    payment_type,
    COUNT(*)                                                 AS num_trips,
    ROUND(AVG(fare_amount), 2)                               AS avg_fare,
    ROUND(AVG(tip_amount), 2)                                AS avg_tip,
    ROUND(AVG(total_amount), 2)                              AS avg_total,
    ROUND(100 * AVG(tip_amount / NULLIF(fare_amount, 0)), 2) AS avg_tip_pct
FROM yellow_trips_clean
WHERE fare_amount > 0
GROUP BY payment_type
ORDER BY num_trips DESC;


-- Q6. Hour of day: volume against average distance.
-- The two move inversely — this is the core result of the project.
SELECT
    HOUR(tpep_pickup_datetime)   AS hour_of_day,
    COUNT(*)                     AS num_trips,
    ROUND(AVG(trip_distance), 2) AS avg_distance
FROM yellow_trips_clean
GROUP BY HOUR(tpep_pickup_datetime)
ORDER BY hour_of_day;


-- Q7. Month-over-month growth. LAG pulls the previous month onto the row.
WITH monthly AS (
    SELECT
        MONTH(tpep_pickup_datetime) AS month,
        COUNT(*)                    AS num_trips
    FROM yellow_trips_clean
    GROUP BY MONTH(tpep_pickup_datetime)
)
SELECT
    month,
    num_trips,
    LAG(num_trips) OVER (ORDER BY month) AS prev_month_trips,
    ROUND(100 * (num_trips - LAG(num_trips) OVER (ORDER BY month))
          / LAG(num_trips) OVER (ORDER BY month), 2) AS mom_growth_pct
FROM monthly
ORDER BY month;


-- ************************************************************
-- PART 5 — GEOGRAPHY
-- ************************************************************

-- Q8. Top 10 pickup zones.
SELECT z.Zone, z.Borough, COUNT(*) AS num_trips
FROM yellow_trips_clean t
JOIN taxi_zone_lookup z ON t.PULocationID = z.LocationID
GROUP BY z.Zone, z.Borough
ORDER BY num_trips DESC
LIMIT 10;


-- Q9. Pickups by borough.
SELECT
    z.Borough,
    COUNT(*)                                         AS num_trips,
    ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM yellow_trips_clean t
JOIN taxi_zone_lookup z ON t.PULocationID = z.LocationID
GROUP BY z.Borough
ORDER BY num_trips DESC;


-- Q10. Airport trips: longer, and far more expensive.
SELECT
    z.Zone,
    COUNT(*)                       AS num_trips,
    ROUND(AVG(t.trip_distance), 2) AS avg_distance,
    ROUND(AVG(t.fare_amount), 2)   AS avg_fare,
    ROUND(AVG(t.total_amount), 2)  AS avg_total
FROM yellow_trips_clean t
JOIN taxi_zone_lookup z ON t.PULocationID = z.LocationID
WHERE z.service_zone = 'Airports'
GROUP BY z.Zone
ORDER BY num_trips DESC;


-- ************************************************************
-- PART 6 — BENCHMARK CHECK: THE CASH ANOMALY
-- My cash share (14.66%) sits well above the ~10% usually cited.
-- Two things to rule out: the cleaning filters, and the denominator.
-- ************************************************************

-- Payment mix on the RAW table, before any filtering.
-- Cash is 13.46% here, so cleaning nudges the share up but does not
-- cause the gap — it was already there in the unfiltered data.
SELECT payment_type, COUNT(*) AS raw_trips
FROM yellow_trips
GROUP BY payment_type;


-- Denominator hypothesis: payment_type 0 (Flex Fare) is 9.94% of records
-- and is neither card nor cash. Does excluding it explain the gap?
-- It does not: cash rises to 14.94%. Dropping rows from a denominator
-- can only push a share up. Hypothesis rejected — the divergence is
-- most likely definitional or tied to the reference year of the
-- official figure, and I leave it unresolved rather than paper over it.
SELECT
    payment_type,
    COUNT(*) AS trips,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_all,
    ROUND(100.0 * COUNT(*) / SUM(CASE WHEN payment_type <> 0 THEN COUNT(*) END) OVER (), 2) AS pct_excl_flex
FROM yellow_trips
GROUP BY payment_type
ORDER BY payment_type;