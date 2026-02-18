# federated-government-statistics

## Data Generation


## Data Aggregation
The cordinator now will request the following aggregated data from the worker nodes.
- Time window
    - start date
    - end date
- Group by
    - county fips
    - time window(start and end time for month and year)
- Metrics
    - population
        - count
    - household
        - count
    - income
        - sum
        - count
    - income buckets
        - count
        - sum
    - age
        - sum
        - count
    - age buckets
        - count
        - sum
    - unemployment rate
        - count with jobs
        - count wihtout jobs
- Quality
    - rows scanned
        - count
    - rows used
        - count
    - rows dropped
        -count

# First Test
The first test will be performed on 50,000 rows split between 5 databses. 

| Databases | Counties | Size |
| --------- | -------- | ---- |
| metro | King | 25,000 |
| medium1 | Pierce | 10,000 |
| medium2 | Snohomish | 10,000 |
| rural1 | Ferry | 2,000 |
| rural2 | San Juan | 3,000 |


