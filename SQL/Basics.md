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
