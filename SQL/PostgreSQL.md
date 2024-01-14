# Sample Data
Contains details of the world's fastest production cars by 0 to 60 mph acceleration time. Each row contains one car model, and the table is named cars.

![Screenshot 2024-01-13 at 9 08 25 PM](https://github.com/mattamx/DataScience_quick_guides/assets/107958646/e3b3e3b5-3b32-488c-90d3-bc9c491f34b1)

# Querying tables
Get all the columns from a table using `SELECT*`
```sql
SELECT *
    FROM cars
```
Get a column from a table by name using `SELECT col`
```sql
SELECT model
    FROM cars
```
Get multiple columns from a table by name using `SELECT col1, col2`
```sql
SELECT make, model
    FROM cars
```
Override column names with `SELECT col AS new_name`
```sql
SELECT make, mode, propulstion_type AS engine_type
    FROM cars
```
Arrange the rows in ascending order of values in a column with `ORDER BY col`
```sql
SELECT make, model, time_to_60_mph_s
    FROM cars
    ORDER BY time_to_60_mph_s
```
Arrange the wors in descending order of values in a column with `ORDER BY col DESC`
```sql
SELECT make, model, model_year
    FROM cars
    ORDER BY model_year DESC
```
Limit the number of rows returned with `LIMIT n`
```python
SELECT *
    FROM cars
LIMIT 2
```
Get unique values with `SELECT DISTINCT`
```sql
SELECT DISTINCT propulsion_type
    FROM cars
```

# Filtering data
## Filtering on numeric columns
Get rows where a number is greater than a value with `WHERE col > n`
```sql
SELECT make, model, time_to_60_mps_s
    FROM cars
    WHERE time_to_60_mph_s > 2.1
```
Get rows where a number is greater than or equal to a value with `WHERE col >= n`
```sql
SELECT make, model, time_to_60_mph_s
    FROM cars
    WHERE time_to_60_mph_s >= 2.1
```
Get rows where a number is less than or equal to a value with `WHERE col <= n`
```sql
SELECT make, model, time_to_60_mph_s
    FROM cars
    WHERE time_to_60_mph_s <= 2.1
```
Get rows where a number is equal to a value with `WHERE col = n`
```sql
SELECT make, model, time_to_60_mph_s
    FROM cars
    WHERE time_to_60_mph_s = 2.1
```
Get rows where a number is not equal to a value with `WHERE col <> n` or `WHERE col != n`
```sql
SELECT make, model, time_to_60_mph_s
    FROM cars
    WHERE time_to_60_mph_s <> 2.1
```
Get rows where a number is between two values (inclusive) with `WHERE col BETWEEN m AND n`
```sql
SELECT make, model, time_to_60_mph_s
    FROM cars
    WHERE time_to_60_mph_s BETWEEN 1.9 and 2.1
```
## Filtering on text columns
Get rows where text is equal to a value with `WHERE col = 'x'`
```sql
SELECT make, model, propulsion_type
    FROM cars
    WHERE propulsion_type = 'Hybrid'
```
Get rows where text is one of several values with `WHERE col IN ('x','y')`
```sql
SELECT make, model, propulsion_type
    FROM cars
    WHERE propulsion_type IN ('Electric', 'Hybrid')
```
Get rows where text contains specific letters with `WHERE col LIKE '%abc%'` (% represents any character)
```sql
SELECT make, model, propulsion_type
    FROM cars
    WHERE propulsion_type LIKE %ic%
```
For case sensitive matching, use `WHERE col ILIKE '%abc%'`
```sql
SELECT make, model, propulsion_type
    FROM cars
    WHERE propulsion_type ILIKE %ic%
```
## Filtering on multiple columns
Get the rows where one condition and another condition holds with `WHERE condn1 AND condn2`
```sql
SELECT make, model, propulsion_type, model_year
    FROM cars
    WHERE propulsion_type = 'Hybrid'
        AND model_year < 2020
```
Get the rows where one condition or another condition holds with `WHERE condn1 OR condn2`
```sql
SELECT make, model, propulsion_type, model_year
    FROM cars
    WHERE propulsion_type = 'Hybrid'
        OR model_year < 2020
```
## Filtering on missing data
Get rows where values are missing with `WHERE col IS NULL`
```sql
SELECT make, model, limited_production_count
    FROM cars
    WHERE limited_production_count IS NULL
```
Get rows where values are not missing with `WHERE col IS NOT NULL`
```sql
SELECT make, model, limited_production_count
    FROM cars
    WHERE limited_production_count IS NOT NULL
```

# Aggregating data
## Simple aggregations
Get the total number of rows `SELECT COUNT(*)`
```sql
SELECT COUNT(*)
    FROM cars
```
Get the total value of a column with `SELECT SUM(col)`
```sql
SELECT SUM(limited_production_count)
    FROM cars
```
Get the mean value of a column with `SELECT AVG(col)`
```sql
SELECT AVG(time_to_60_mph_s)
    FROM cars
```
Get the minimum value of a column with `SELECT MIN(col)`
```sql
SELECT MIN(time_to_60_mph_s)
    FROM cars
```
Get the maximum value of a column with `SELECT MAX(col)`
```sql
SELECT MAX(time_to_60_mph_s)
    FROM cars
```
## Grouping, filtering and sorting
Get summaries grouped by values with `GROUP BY col`
```sql
SELECT propulsion_type, COUNT(*)
    FROM cars
    GROUP BY propulsion_type
```
Get summaries grouped by values, in order of summaries with `GROUP BY col ORDER BY smmry`
```sql
SELECT propulsion_type, AVG(time_to_60_mps_s) AS mean_time_to_60_mps_s
    FROM cars
    GROUP BY propulsion_type
    ORDER BY mean_time_to_60_mps_s
```
Get rows where values in a group meet a criterion with `GROUP BY col HAVING condn`
```sql
SELECT propulsion_type, AVG(time_to_60_mps_s) AS mean_time_to_60_mps_s
    FROM cars
    GROUP BY propulsion_type
    HAVING mean_time_to_60_mps_s > 2
```
Filter before and after grouping with WHERE condn_before `GROUP BY col HAVING condn_after`
```sql
SELECT propulsion_type, AVG(time_to_60_mps_s) AS mean_time_to_60_mps_s
    FROM cars
WHERE limited_production_count IS NOT NULL
    GROUP BY propulsion_type
    HAVING mean_time_to_60_mps_s > 2
```
# PostgreSQL-Specific Syntax
Limit the number of rows returned, offset from the top with `LIMIT m OFFSET n`
```sql
SELECT *
    FROM cars
LIMIT 2 OFFSET 3
```
PostgreSQL allows text concatenation with the `||` operator
```sql
SELECT make || ' ' || model AS make_and_model
    FROM cars
```
Get the current date with `CURRENT_DATE` and the current datetime with `NOW()` or `CURRENT_TIME`
```sql
SELECT NOW(), CURRENT_DATE, CURRENT_TIME
```
List available tables by selecting from `pg_cataglog.pg_tables`
```sql
SELECT * FROM pg_cataglog.pg_tables
```
