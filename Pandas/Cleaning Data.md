# Data type constraints
Strings to integers
```python
# import csv file and output header
df = pd.read_csv('name.csv')
df.head(2)

# get data types of columns
df.dtypes

# get DataFrame information
df.info()
```

```python
# print column sum
df['col'].sum() # values are stored as strings

# remove specific string from the column
df['col'] = df['col'].str.strip('specific string')
df['col'] = df['col'].astype('int') # convert to a numeric column

# verify that the column is now an integer
assert df['col'].dtype == 'int'
```
The assert statement
```python
# this will pass
assert 1+1 == 2

# this will not pass
assert 1+1 == 3 # AssertionError
```
Numeric or categorical?
```python
# column contains values corresponding to perceived categories
df['col'].describe()

# convert to categorical
df['col'] = df['col'].astype('category')
df.describe()
```

# Data range constraints
How to deal with out of range data?
- Dropping data
- Setting custom minimums and maximums
- Treat as missing and impute
- Setting custom value depending on business assumptions

## Data range example
```python
import pandas as pd

# specific subsetting
df[df['col'] > value]

# dropping values using filtering
df = df[df['col'] <= value]

# dropping values using .drop()
df.drop(df[df['col'] > value].index, inplace=True)

# asserting results
assert df['col'].max() <= value

# converting column above the range to the limit value
df.loc[df['col'] > value, 'col'] = limit value

# asserting statement
assert df['col'].max() <= value # no output means it passed
```
## Date range example
```python
import datetime as dt
import pandas as pd

# output data types
df.dtypes

# convert to date
df['col'] = pd.to_datetime(df['col'])dt.date

today_date = dt.date.today()
```
Dropping the data
```python
# dropping values using filtering
df = df[df['col'] < today_date]

# dropping values using .drop()
df.drop(df[df['col'] > today_date].index, inplace=True)
```
Hardcode dates with upper limit
```python
# dropping values using filtering
df.loc[df['col'] > today_date, 'col'] = today_date

# asserting the statement
assert df.col.max().date() <= today_date

```

# Uniqueness constraints
Duplicate values
- All columns have the same values
- Most columns have the same values

Why do they happen?
- Data entry & human error
- Join or merge errors
- Bugs and design errors

## Finding duplicate values
- The `.duplicated()` method
  - `subset`: list of column names to check for duplication
  - `keep`: whether to keep **first** (`first`), **last** (`last`), **all** (False) duplicate values
```python
# printing the header
df.head()

# get duplicates across all columns
duplicates = df.duplicated()
print(duplicated)

$ get duplicate rows
duplicates = df.duplicated()
df[duplicates]
```
Checking specific columns for duplication
```python
column_names = ['col1', 'col2', 'col3']
duplicates = df.duplicated(subset = column_names, keep=False)

# output duplicate values
df[duplicates]

# sorting duplicate values by a specific column
df[duplicates].sort_values(by = 'col1')
```

## Treating duplicate values
- The `.drop_duplicates()` method
  - `subset`: list of column names to check for duplication
  - `keep`: whether to keep **first** (`first`), **last** (`last`), **all** (False) duplicate values
  - `inplace`: drop duplicated rows directly inside the DataFrame without creating a new object (`True`)
```python
# drop duplicates
df.drop_duplicates(inplace=True)

# output duplicate values
column_names = ['col1', 'col2', 'col3']
duplicates = df.duplicated(subset = column_names, keep=False)
df[duplicates].sort_values(by = 'col1')
```
The `.groupby()` and `.agg()` methods
```python
# group by column names and produce statistical summaries
column_names = ['col1', 'col2', 'col3']
summaries = {'col4': 'max', 'col5':'mean'}
df = df.groupby(by= column_names).agg(summaries).reset_index()

# make sure aggregation is done
duplicates = df.duplicated(subset = column_names, keep=False)
df[duplicates].sort_values(by= 'col1')
```

# Membership constraints
Categories and membership constraints
- Predefined finite set of categories

Why could we have these problems?
- Data entry errors
  - free text
  - dropdowns
- Parsing errors

How to treat these problems?
- Dropping data
- Remapping categories
- Inferring categories
  
```python
# read data and print it
data = pd.read_csv('data.csv')
print(data)

# correct possible values for categorical column
categories
```
## Inconsistent categories
Finding inconsistent categories
```python
inconsistent_categories = set(data['categorical_col']).difference(categories['categorical_col'])
print(inconsistent_categories)

# get and print rows with inconsistent categories
inconsistent_rows = df['categorical_col'].isin(inconsistent_categories)
data[inconsistent_rows]
```
Dropping inconsistent categories
```python
inconsistent_categories = set(data['categorical_col']).difference(categories['categorical_col'])
inconsistent_rows = df['categorical_col'].isin(inconsistent_categories)
inconsistent_data = data[inconsistent_rows]

# drop inconsistent categories and get consistent data only
consistent_data = data[~inconsistent_rows]
```

# Categorical variables
1) Value inconsistency
- Inconsistent fields: `string`, `String`, `strinG`, sTring`
- _Trailing white spaces: _`string `, ' string '
2) Collapsing too many categories to few
- Creating new groups: multiple categories from continuous data
- Mapping groups to new ones: mapping multipe categories to 2
3) Making sure data is of type `category`

## Value consistency
```python
# capitalize
data['col'] = data['col'].str.upper()
data['col'].value_counts() # checking changes

# lowercase
data['col'] = data['col'].str.lower()
data['col'].value_counts() # checking changes

# trailing spaces
data['col'] = data['col'].str.strip()
data['col'].value_counts() # checking changes
```

## Collapsing data into categories

Creating categories out of data columns
```python
# using qcut()
import pandas as pd

group_names = ['group1', 'group2', 'group3']
df['new_col'] = pd.qcut(df['numerical_col'], q = 3, labels = group_names)

# pring new column
df[['new_col', 'numerical_col']]
```

```python
# using cut() - create category ranges and names
ranges = [0, value, value, np.inf] # always start with 0
group_names = ['group1', 'group2', 'group3']
df['new_col'] = pd.qcut(df['numerical_col'], bins = ranges, labels = group_names)

# pring new column
df[['new_col', 'numerical_col']]
```
Mapping categories to fewer ones
- Reducing categories in categorical column
  
```python
# create mapping dictionary and replace
mapping = {'cat1': 'group1', 'cat2': 'group1','cat3': 'group1', 'cat4': 'group2', 'cat5': 'group2',}
df['col'] = df['col'].replace(mapping)
df['col'].unique()
```

# Cleaning text data
Common text data problems:
1) Data inconsistency
- A phone number with + or 00
2) Fixed length violations
- Passwords need to be at least 8 characters
3) Typos
- Periods, commas and other characters in the string

```python
df = pd.read_csv('name.csv')
print(df)

# replacing a character with another
df['col'] = df['col'[.str.replace("+", "00")
print(df)

# further replacing a character with nothing
df['col'] = df['col'[.str.replace("-", "")
print(df)
```

```python

```

```python

```

```python

```

