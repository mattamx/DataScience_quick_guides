# Sample data
![Screenshot 2024-01-14 at 10 44 15 AM](https://github.com/mattamx/DataScience_quick_guides/assets/107958646/243e428e-bb4c-464e-be9c-6fe90bf65da4)

# Querying a single table
Fetch al columns from the country table
```sql
SELECT *
FROM country;
```
Fetch id and name columns from the city table
```sql
SELECT id, name
FROM city;
```
Fetch city names sorted by the rating column in the default ascinding order
```sql
SELECT name
FROM city
ORDER BY rating [ASC];
```
Fetch city names sorted by the rating column in descending order
```sql
SELECT name
FROM city
ORDER BY rating DESC;
```

# Aliases
## Columns
```sql
SELECT name AS city_name
FROM city;
```
## Tables
```sql
SELECT co.name, ci.name
FROM city AS ci
JOIN country AS co
  ON ci.country_id = co.id;
```

# Filtering the output
## Comparison operators
Fetch names of cities that have a rating above 3
```sql
SELECT name
FROM city
WHERE rating > 3;
```
Fetch names of cities that are neither Berlin or Madrid
```sql
SELECT name
FROM city
WHERE name != 'Berlin'
  AND name != 'Madrid';
```
## Text operators
Fetch names of cities that start with a 'P' or end with an 'S'
```sql
SELECT name
FROM city
WHERE name LIKE 'P%'
  OR name LIKE '%s';
```
Fetch names of cities that start with any letter followed by 'ublin'
```sql
SELECT name
FROM city
WHERE name LIKE '_ublin';
```
## Other operators
Fetch names of cities that have a population between 500K and 5M
```sql
SELECT name
FROM city
WHERE population BETWEEN 500000 AND 5000000;
```
Fetch names of cities that don't miss a rating value
```sql
SELECT name
FROM city
WHERE rating IS NOT NULL;
```
Fetch names of cities that are in countries with IDs 1, 4, 7 or 8
```sql
SELECT name
FROM city
WHERE country_id IN (1, 4, 7, 8);
```

# Querying multiple tables
## Inner join
Returns rows that have matching values in both tables.
![Screenshot 2024-01-14 at 2 33 54 PM](https://github.com/mattamx/DataScience_quick_guides/assets/107958646/211026a4-ac4b-4907-ba52-ef15309236be)
```sql
SELECT city.name, country.name
FROM city
[INNER] JOIN country
  ON city.country_id = country.id;
```
## Left join
Returns all rows from the left table with corresponding rows from the right table. If there is not matching row, **NULLs** are returned as values from the second table.
![Screenshot 2024-01-14 at 2 36 11 PM](https://github.com/mattamx/DataScience_quick_guides/assets/107958646/f1c65bae-c5af-4c68-960f-5672664e81dd)
```sql
SELECT city.name, country.name
FROM city
LEFT JOIN country
  ON city.country_id = country.id;
```
## Right join
Returns all rows from the right table with corresponding rows from the left table. If there's no matching row, **NULLs** are returned as values from the left table.
```sql
SELECT city.name, country.name
FROM city
LEFT JOIN country
  ON city.country_id = country.id;
```
## Full join
Returns all rows from both tables - if there are no matching rows in the second table, **NULLs** are returned.
![Screenshot 2024-01-14 at 2 41 32 PM](https://github.com/mattamx/DataScience_quick_guides/assets/107958646/d53ff101-b9aa-4ae2-97a8-bc8b6da9b85a)
```sql
SELECT city.name, country.name
FROM city
FULL [OUTER] JOIN country
  ON city.country_id = country.id;
```
## Cross join
Returns all possible combinations of rows from both tables. There are two syntaxes available.
![Screenshot 2024-01-14 at 2 41 56 PM](https://github.com/mattamx/DataScience_quick_guides/assets/107958646/c5da6b92-e30a-4ec3-b98b-36cc3ef24d72)
```sql
SELECT city.name, country.name
FROM city
CROSS JOIN country;

SELECT city.name, country.name
FROM city, country;
```
## Natural join
Joins tables by all columns with the same name. Rarely used in practice.
![Screenshot 2024-01-14 at 2 42 21 PM](https://github.com/mattamx/DataScience_quick_guides/assets/107958646/8b7dbd37-1e0f-4ec1-a13d-17bbf42fd9ba)
```sql
SELECT city.name, country.name
FROM city
NATURAL JOIN country;
```

# Aggregation and Grouping
GROUP BY groups together rows that have the same values in specified columns. It computes summaries (aggregates) for each unique combination of values.
![Screenshot 2024-01-14 at 2 43 39 PM](https://github.com/mattamx/DataScience_quick_guides/assets/107958646/6fed2081-fffa-47e3-8500-9958e6a04157)
## Aggregate functions
- avg(expr): average value for rows within the group
- count(expr): count of values for rows within the group
- max(expr): maximum value within the group
- min(expr): minimum value within the group
- sum(expr): sum of values within the group

## Example queries
Find out the number of cities
```sql
SELECT COUNT(*)
FROM city;
```
Find out the number of cities with non-null ratings
```sql
SELECT COUNT(rating)
FROM city;
```
Find out the number of distinctive country values
```sql
SELECT COUNT(DISTINCT country_id)
FROM city;
```
Find out the smalles and the greatest country populations
```sql
SELECT MIN(population), MAX(population)
FROM country;
```
Find out the total population of cities in respective countries
```sql
SELECT country_id, SUM(population)
FROM city
GROUP BY country_id;
```
Find out the average rating for cities in respective countries if the average is above 3.0
```sql
SELECT country_id, AVG(rating)
FROM city
GROUP BY country_id
HAVING AVG(rating) > 3.0;
```

# Subqueries
A subquery is a query that is nested inside another query, or inside another query. There are different types of subqueries.
## Single value
The simples subquery returns exactly one column and exactly one row. It can be used with comparison operators.

This query finds cities with the same rating as Paris.
```sql
SELECT name FROM city
WHERE rating = (
      SELECT rating
      FROM city
      WHERE name = 'Paris'
);
```
