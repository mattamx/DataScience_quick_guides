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
To begin,you can estimate the size of any query before running it. Here is an example using the (*very large!*) Hacker News dataset. To see how much data a query will scan, we create a `QueryJobConfig` object and set the `dry_run` parameter to True.
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
