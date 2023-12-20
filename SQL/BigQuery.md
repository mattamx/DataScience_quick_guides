## Import

```python
from google.cloud import bigquery
```

### Client Object
The first step in the workflow is to create a `Client` object. This `Client` object will play a central role in retrieving information from BigQuery datasets.
```python
# Create a "Client" object
client = bigquery.Client()
```

### Dataset
In BigQuery, each dataset is contained in a corresponding project. We'll work with a dataset of posts on Hacker News, a website focusing on computer science and cybersecurity news. In this case, our `hacker_news` dataset is contained in the `bigquery-public-data` project. 

To access the dataset:
- We begin by constructing a reference to the dataset with the `dataset()` method.
- Next, we use the `get_dataset()` method, along with the reference we just constructed, to fetch the dataset.

```python
# Construct a reference to the "hacker_news" dataset
dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)
```

Every dataset is just a collection of tables. You can think of a dataset as a spreadsheet file containing multiple tables, all composed of rows and columns.

We use the `list_tables()` method to list the tables in the dataset.

```python
# List all the tables in the "hacker_news" dataset
tables = list(client.list_tables(dataset))

# Print names of all tables in the dataset (there are four!)
for table in tables:  
    print(table.table_id)
```

Similar to how we fetched a dataset, we can fetch a table. In the code cell below, we fetch the `full` table in the `hacker_news` dataset.

```python
# Construct a reference to the "full" table
table_ref = dataset_ref.table("full")

# API request - fetch the table
table = client.get_table(table_ref)
```

![image](https://github.com/mattamx/DataScience_guides/assets/107958646/6248226b-ca29-43d1-9829-9b739cfe0478)

### Table schema
The structure of a table is called its **schema**. We need to understand a table's schema to effectively pull out the data we want.

In this example, we'll investigate the `full` table that we fetched above.

```python
# Print information on all the columns in the "full" table in the "hacker_news" dataset
table.schema
```

Each `SchemaField` tells us about a specific column (which we also refer to as a **field**). In order, the information is:

- The **name** of the column
- The **field type** (or datatype) in the column
- The **mode** of the column ('NULLABLE' means that a column allows NULL values, and is the default)
- A **description** of the data in that column

The first field has the SchemaField:

`SchemaField('by', 'string', 'NULLABLE', "The username of the item's author.",())`

This tells us:

- the field (or column) is called by,
- the data in this field is strings,
- NULL values are allowed, and
- it contains the usernames corresponding to each item's author.

We can use the `list_rows()` method to check just the first five lines of of the `full` table to make sure this is right. (Sometimes databases have outdated descriptions, so it's good to check.) This returns a BigQuery `RowIterator` object that can quickly be converted to a pandas DataFrame with the `to_dataframe()` method.

```python
# Preview the first five lines of the "full" table
client.list_rows(table, max_results=5).to_dataframe()
```

The `list_rows()` method will also let us look at just the information in a specific column. If we want to see the first five entries in the `by` column, for example, we can do that!

```python
# Preview the first five entries in the "by" column of the "full" table
client.list_rows(table, selected_fields=table.schema[:1], max_results=5).to_dataframe()
```

### Submitting a query
```python
# Query to get the score column from every row where the type column has value "job"
query = """
        SELECT score, title
        FROM `bigquery-public-data.hacker_news.full`
        WHERE type = "job" 
        """
```
We begin by setting up the query with the query() method.
```python
# Set up the query
query_job = client.query(query)
```
Next, we run the query and convert the results to a pandas DataFrame.
```python
# API request - run the query, and return a pandas DataFrame
variable = query_job.to_dataframe()
```

### Working with big datasets
To begin, you can estimate the size of any query before running it. Here is an example using the (*very large!*) Hacker News dataset. To see how much data a query will scan, we create a `QueryJobConfig` object and set the `dry_run` parameter to True.
```python
# Query to get the score column from every row where the type column has value "job"
query = """
        SELECT score, title
        FROM `bigquery-public-data.hacker_news.full`
        WHERE type = "job" 
        """

# Create a QueryJobConfig object to estimate size of query without running it
dry_run_config = bigquery.QueryJobConfig(dry_run=True)

# API request - dry run query to estimate costs
dry_run_query_job = client.query(query, job_config=dry_run_config)

print("This query will process {} bytes.".format(dry_run_query_job.total_bytes_processed))
```

You can also specify a parameter when running the query to limit how much data you are willing to scan. Here's an example with a low limit.
```python
# Only run the query if it's less than 1 MB
ONE_MB = 1000*1000
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=ONE_MB)

# Set up the query (will only run if it's less than 1 MB)
safe_query_job = client.query(query, job_config=safe_config)

# API request - try to run the query, and return a pandas DataFrame
safe_query_job.to_dataframe()
```
### Group By, Having & Count
We'll work with the `comments` table and begin by printing the first few rows.
```python
from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "hacker_news" dataset
dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# Construct a reference to the "comments" table
table_ref = dataset_ref.table("comments")

# API request - fetch the table
table = client.get_table(table_ref)

# Preview the first five lines of the "comments" table
client.list_rows(table, max_results=5).to_dataframe()
```

Let's use the table to see which comments generated the most replies. Since:

- the `parent` column indicates the comment that was replied to, and
- the `id` column has the unique ID used to identify each comment,

we can **GROUP BY** the parent column and **COUNT()** the id column in order to figure out the number of comments that were made as responses to a specific comment.

Furthermore, since we're only interested in popular comments, we'll look at comments with more than ten replies. So, we'll only return groups **HAVING** more than ten ID's.

```python
# Query to select comments that received more than 10 replies
query_popular = """
                SELECT parent, COUNT(id)
                FROM `bigquery-public-data.hacker_news.comments`
                GROUP BY parent
                HAVING COUNT(id) > 10
                """
```

Now that our query is ready, let's run it and store the results in a pandas DataFrame:

```python
# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 10 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query_popular, job_config=safe_config)

# API request - run the query, and convert the results to a pandas DataFrame
popular_comments = query_job.to_dataframe()

# Print the first five rows of the DataFrame
popular_comments.head()
```

#### Note on using Group By
Note that because it tells SQL how to apply aggregate functions (like **COUNT()**), it doesn't make sense to use **GROUP BY** without an aggregate function. Similarly, if you have any **GROUP BY** clause, then all variables must be passed to either a

1. **GROUP BY** command, or
2. an aggregation function.
   
Consider the query below:
```python
query_good = """
             SELECT parent, COUNT(id)
             FROM `bigquery-public-data.hacker_news.comments`
             GROUP BY parent
             """
```

Note that there are two variables: `parent` and `id`.

- `parent` was passed to a **GROUP BY** command (in `GROUP BY parent`), and
- `id` was passed to an aggregate function (in `COUNT(id)`).

And this query won't work, because the `author` column isn't passed to an aggregate function or a **GROUP BY** clause:
```python
query_bad = """
            SELECT author, parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            """
```

If make this error, you'll get the error message `SELECT list expression references column (column's name) which is neither grouped nor aggregated at`.

### Analytic Functions
Analytic functions allow us to perform complex calculations with relatively straightforward syntax. For instance, we can quickly calculate moving averages and running totals, among other quantities.

#### Syntax
To understand how to write analytic functions, we'll work with a small table containing data from two different people who are training for a race. The `id` column identifies each runner, the `date` column holds the day of the training session, and time shows the `time` (in minutes) that the runner dedicated to training. Say we'd like to calculate a moving average of the training times for each runner, where we always take the average of the current and previous training sessions. We can do this with the following query:
![image](https://github.com/mattamx/DataScience_guides/assets/107958646/428a5380-3017-4462-972d-a78e42b7f06c)


All analytic functions have an **OVER** clause, which defines the sets of rows used in each calculation. The **OVER** clause has three (optional) parts:

- The **PARTITION BY** clause divides the rows of the table into different groups. In the query above, we divide by `id` so that the calculations are separated by runner.
- The **ORDER BY** clause defines an ordering within each partition. In the sample query, ordering by the `date` column ensures that earlier training sessions appear first.
- The final clause (`ROWS BETWEEN 1 PRECEDING AND CURRENT ROW`) is known as a **window frame** clause. It identifies the set of rows used in each calculation. We can refer to this group of rows as a **window**. (Actually, analytic functions are sometimes referred to as **analytic window functions** or simply **window functions**!)

![image](https://github.com/mattamx/DataScience_guides/assets/107958646/7ce82c0d-5f7a-47b1-aaee-f9af6e9bf756)

#### (More on) window frame clauses

There are many ways to write window frame clauses:

- `ROWS BETWEEN 1 PRECEDING AND CURRENT ROW` - the previous row and the current row.
- `ROWS BETWEEN 3 PRECEDING AND 1 FOLLOWING` - the 3 previous rows, the current row, and the following row.
- `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` - all rows in the partition.

Of course, this is not an exhaustive list, and you can imagine that there are many more options! In the code below, you'll see some of these clauses in action.

#### Three types of analytic functions
The example above uses only one of many analytic functions. BigQuery supports a wide variety of analytic functions, and we'll explore a few here. For a complete listing, you can take a look at the [documentation](https://cloud.google.com/bigquery/docs/reference/standard-sql/analytic-function-concepts).

##### 1) Analytic aggregate functions
As you might recall, **AVG()** (from the example above) is an aggregate function. The **OVER** clause is what ensures that it's treated as an analytic (aggregate) function. **Aggregate functions** take all of the values within the window as input and return a single value.

- **MIN()** (or **MAX()**) - Returns the minimum (or maximum) of input values
- **AVG()** (or **SUM()**) - Returns the average (or sum) of input values
- **COUNT()** - Returns the number of rows in the input

##### 2) Analytic navigation functions
**Navigation functions** assign a value based on the value in a (usually) different row than the current row.

- **FIRST_VALUE()** (or **LAST_VALUE()**) - Returns the first (or last) value in the input
- **LEAD()** (and **LAG()**) - Returns the value on a subsequent (or preceding) row

##### 3) Analytic numbering functions
**Numbering functions** assign integer values to each row based on the ordering.

- **ROW_NUMBER()** - Returns the order in which rows appear in the input (starting with 1)
- **RANK()** - All rows with the same value in the ordering column receive the same rank value, where the next row receives a rank value which increments by the number of rows with the previous rank value.

### Writing Efficient Queries
Most database systems have a **query optimizer** that attempts to interpret/execute your query in the most effective way possible. But several strategies can still yield huge savings in many cases.

##### Some useful functions
We will use two functions to compare the efficiency of different queries:

- `show_amount_of_data_scanned()` shows the amount of data the query uses.
- `show_time_to_run()` prints how long it takes for the query to execute.

#### Strategies
##### 1) Only select the columns you want.
It is tempting to start queries with **SELECT * FROM**.... It's convenient because you don't need to think about which columns you need. But it can be very inefficient.

This is especially important if there are text fields that you don't need, because text fields tend to be larger than other fields.

```python
star_query = "SELECT * FROM `bigquery-public-data.github_repos.contents`"
show_amount_of_data_scanned(star_query)

basic_query = "SELECT size, binary FROM `bigquery-public-data.github_repos.contents`"
show_amount_of_data_scanned(basic_query)
```
```
Data processed: 2682.118 GB
Data processed: 2.531 GB
```
In this case, we see a 1000X reduction in data being scanned to complete the query, because the raw data contained a text field that was 1000X larger than the fields we might need.

##### 2) Read less data.
Both queries below calculate the average duration (in seconds) of one-way bike trips in the city of San Francisco.

```python
more_data_query = """
                  SELECT MIN(start_station_name) AS start_station_name,
                      MIN(end_station_name) AS end_station_name,
                      AVG(duration_sec) AS avg_duration_sec
                  FROM `bigquery-public-data.san_francisco.bikeshare_trips`
                  WHERE start_station_id != end_station_id 
                  GROUP BY start_station_id, end_station_id
                  LIMIT 10
                  """
show_amount_of_data_scanned(more_data_query)

less_data_query = """
                  SELECT start_station_name,
                      end_station_name,
                      AVG(duration_sec) AS avg_duration_sec                  
                  FROM `bigquery-public-data.san_francisco.bikeshare_trips`
                  WHERE start_station_name != end_station_name
                  GROUP BY start_station_name, end_station_name
                  LIMIT 10
                  """
show_amount_of_data_scanned(less_data_query)
```
```
Data processed: 0.076 GB
Data processed: 0.06 GB
```
Since there is a 1:1 relationship between the station ID and the station name, we don't need to use the `start_station_id` and `end_station_id` columns in the query. By using only the columns with the station IDs, we scan less data.

##### 3) Avoid N:N JOINs.
Most of the JOINs that you have executed in this course have been **1:1 JOINs**. In this case, each row in each table has at most one match in the other table.

![image](https://github.com/mattamx/DataScience_guides/assets/107958646/06804ec9-fdfd-43bd-896d-0903e7c200f7)

Another type of JOIN is an **N:1 JOIN**. Here, each row in one table matches potentially many rows in the other table.

![image](https://github.com/mattamx/DataScience_guides/assets/107958646/54e48221-dfb6-42f6-a191-80a46d1f6645)

Finally, an **N:N JOIN** is one where a group of rows in one table can match a group of rows in the other table. Note that in general, all other things equal, this type of JOIN produces a table with many more rows than either of the two (original) tables that are being JOINed.

![image](https://github.com/mattamx/DataScience_guides/assets/107958646/39111364-232a-43a7-acc9-f94e1e47e414)

Now we'll work with an example from a real dataset. Both examples below count the number of distinct committers and the number of files in several GitHub repositories.

```python
big_join_query = """
                 SELECT repo,
                     COUNT(DISTINCT c.committer.name) as num_committers,
                     COUNT(DISTINCT f.id) AS num_files
                 FROM `bigquery-public-data.github_repos.commits` AS c,
                     UNNEST(c.repo_name) AS repo
                 INNER JOIN `bigquery-public-data.github_repos.files` AS f
                     ON f.repo_name = repo
                 WHERE f.repo_name IN ( 'tensorflow/tensorflow', 'facebook/react', 'twbs/bootstrap', 'apple/swift', 'Microsoft/vscode', 'torvalds/linux')
                 GROUP BY repo
                 ORDER BY repo
                 """
show_time_to_run(big_join_query)

small_join_query = """
                   WITH commits AS
                   (
                   SELECT COUNT(DISTINCT committer.name) AS num_committers, repo
                   FROM `bigquery-public-data.github_repos.commits`,
                       UNNEST(repo_name) as repo
                   WHERE repo IN ( 'tensorflow/tensorflow', 'facebook/react', 'twbs/bootstrap', 'apple/swift', 'Microsoft/vscode', 'torvalds/linux')
                   GROUP BY repo
                   ),
                   files AS 
                   (
                   SELECT COUNT(DISTINCT id) AS num_files, repo_name as repo
                   FROM `bigquery-public-data.github_repos.files`
                   WHERE repo_name IN ( 'tensorflow/tensorflow', 'facebook/react', 'twbs/bootstrap', 'apple/swift', 'Microsoft/vscode', 'torvalds/linux')
                   GROUP BY repo
                   )
                   SELECT commits.repo, commits.num_committers, files.num_files
                   FROM commits 
                   INNER JOIN files
                       ON commits.repo = files.repo
                   ORDER BY repo
                   """

show_time_to_run(small_join_query)
```
```
Time to run: 13.028 seconds
Time to run: 4.413 seconds
```
The first query has a large N:N JOIN. By rewriting the query to decrease the size of the JOIN, we see it runs much faster.
