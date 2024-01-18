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

# replacing a string based on length
df_filter = df['col'].str.len()
df.loc[df_filter < value, 'col'] = np.nan

# finding the length of each row in the column
sanity_check = df['col'].str.len()

# asserting minimum length is a specific value
asser sanity_check.min() >= value

# asserting all values do not have any specific strings
asser df['col'].str.contains("+|-").any() == False

# replacing letter in a column with nothing
df['col'] = df['col'].str.replace(r'\D+', '') # regular expression
df.head()
```

# Uniformity
| Column | Unit |
| :---------------: | --------------- |
| Temperature | `32°C` is also `89.6°F`
| Weight | `70 Kg` is also `11 st.`
| Date | `26-11-2019` is also `26, November, 2019`
| Money | `100$` is also `10763.90¥`

```python
temperatures = pd.read_csv('temperature.csv')
temperatures.head()
```
Checking for non-uniformity with matplotlib
```python
import matplotlib.pyplot as plt

# creating scatterplot
plt.scatter(x= 'Date', y= 'Temperature', data= temperatures)
plt.title('Temperature in Celsius March 2019 - NYC')
plt.xlabel('Dates')
plt.ylabel('Temperature in Celsius')
plt.show()
```
Treating temperature data
```python
# conversion to celsius (for perceived fahrenheit outliers)
temp_fah = temperatures.loc[temperatures['Temperature'] > 40, 'Temperature']
temp_cels = (temp_fah - 32) * (5/9)
temperatures.loc[temperatures['Temperature'] > 40, 'Temperature'] = temp_cels

# aserting conversion is correct
assert temperatures['Temperature'].max() < 40
```
Treating date data
`datetime` is useful for representing dates
| Date | datetime format |
| :---------------: | --------------- |
| 25-12-2019 | `%d-%m-%Y`
| December 25th 2019 | `%c`
| 12-25-2019 | `%m-%d-%Y`

`pandas.to_datetime()`
- Can recognize most formats automatically
- Sometimes fails with erroneous or unrecognizable formats

Treating ambiguous date data
- Is `2019-03-08 in August or March?
  - Convert to `NA` and treat accordingly
  - Infer format by understanding data source
  - Infer format by understanding previous and subsequent data in DataFrame
  
```python
# converting to date time - but won't work
birthdays['Birthday'] = pd.to_datetime(birthdays['Birthday'])

# will work
birthdays['Birthday'] = pd.to_datetime(birthdays['Birthday'],
                        # attempt to infer format of each date
                        infer_datetime_format = True,
                        # return NA for rows where conversion failed
                        errors = 'coerce')

# applying a specific format
birthdays['Birthday'] = birthdays['Birthday'].dt.strftime('%d-%m-%Y')
```

# Cross field validation
The use of **multiple** fields in a dataset to sanity check data integrity

What to do when we catch inconsistencies?
- Dropping data
- Set to missing and impute
- Apply rules from domain knowledge

```python
sum_classes = df[['col1', 'col2', 'col3']].sum(axis=1)
total_equivalence = sum_classes == df['col'] # comparing the sum to the total column for a sanity check

# find and filter out rows with inconsistent totals
inconsistent_totals = df[total_equivalence]
consistent_totals = df[~total_equivalence]
```

```python
import pandas as pd
import datetime as dt

# convert to date time and get today's date
df['col'] = pd.to_datetime(df['col])
today = dt.date.today()

# for each row in the date column, calculate year difference
manual_difference = today.year - df['col'].dt.year

# find instances where years match
year_equivalence = manual_difference == df['col1'] # finding matches

# find and filter out rows with inconsistent
inconsistent_year = df[year_equivalence]
consistent_year = df[~year_equivalence]
```

# Completeness
Missing data: occurs when no data is stored for a variable in an observation
- Can be represented as `NA`, `nan`, `0`, `.`
- Technical errors
- Human errors
```python
# return missing values
df.isna() # boolean table

# summary of missingness
df.isna().sum()
```
## Missingno
Useful package for visualizing and understanding missing data
```python
import missingno as msno
import matplotlib.pyplot as plt

# visualize missingness
msno.matrix(df)
plt.show()
```
Isolating missing and complete values aside
```python
missing = df[df['col'].isna()]
complete = df[~df['col'].isna()]

complete.describe()
missing.describe()
```
Visualizing sorted missingness as a check
```python
sorted_df = df.sort_values(by='col')
msno.matrix(sorted_df)
plt.show()
```
## Missingness types
- Missing Completely at Random (MCAR)
  - No systematic relationship between data and other values
    - Data entry errors when inputting data
- Missing at Random (MAR)
  - Systemic relationship between missing data and other observed values
    - Missing ozon data for high temperatures
- Missing Not at Random (MNAR)
  - Systemic relationship between missing data and unobserved values
    - Missing temperature values for high temperatures
   
How to deal with missing data?

**Simple approaches:**
1. Drop missing data
2. Impute with statistical measures *(mean, median, mode..)*
   
**More complex approaches:**
1. Imputing using an algorithmic approach
2. Impute with machine learning models
   
```python
# dropping missing values
df_dropped = df.dropna(subset=['col'])
df_dropped.head()
```
Replacing with statistical measures
```python
col_mean = df['col'].mean()
df_imputed = df.fillna({'col': col_mean})
df_imputed.head()
```

# Comparing strings

Minimum edit distance algorithms
| Algorithm | Operations |
| :---------------: | --------------- |
| Damerau-Levenshtein | insertion, substitution, deletion, transposition
| Levenshtein | insertion, substitution, deletion
| Hamming | substitution only
| Jaro distance | transposition only

**Possible packages:** `nltk`, `thefuzz`, `textdistance`

Simple string comparison
```python
from thefuzz import fuzz

# comparing reeding vs reading
fuzz.WRatio('Reeding', 'Reading') # 86
```
Partial strings and different orderings
```python
# partial string comparison
fuzz.WRatio('Houston Rockets', 'Rockets') # 90

# partial string comparison with different order
fuzz.WRatio('Houston Rockets vs Los Angeles Lakers', 'Lakers vs Rockets') # 86
```
Comparison with arrays
```python
from thefuzz import process

# define string and array of possible matches
string = 'Houston Rockets vs Los Angeles Lakers'
choices = pd.Series(['Rockets vs Lakers', 'Lakers vs Rockets',
                    'Houston vs Los Angeles', 'Heat vs Bulls'])

process.extract(string, choices, limit = 2)
```
Collapsing categories with string similarity
- Use `.replace()` to collapse `"eur"` into `"Europe"`
- What if there are too many variations?
- `"EU"` , `"eur"` , `"Europ"` , `"Europa"` , `"Erope"` , `"Evropa"`

Collapsing categories with string matching
```python
print(df['col'].unique())
categories # 2 categories

# for each correct category
for element in categories['col']:
  # find potential matches with typos
    matches = process.extract(element, df['col'], limit = df.shape[0])
    # for each potential match match
    for potential_match in matches:
      # if high similarity score
        if potential_match[1] >= 80:
          # replace typo with correct category
          df.loc[df['col'] == potential_match[0], 'col'] = element
```

# Generating pairs

`recordlinkage` package

- Data A and Data B
  - Generate pairs
  - Compare pairs
  - Score pairs
- Link data
```python
import recordlinkage

# create indexing object
indexer = recordlinkage.Index()

# generate pairs blocked on column
indexer.block('col')
pairs = indexer.index(df1, df2)

print(pairs)
```
Comparing the DataFrames
```python
# generate pairs
pairs = indexer.index(df1, df2)

# create the compare object
compare_cl = recordlinkage.Compare()

# find exact matches for pairs fo column 1 and column 2
compare_cl.extract('col1', 'col1', label = 'col1')
compare_cl.extract('col2', 'col2', label = 'col2')

# find similar matches for pairs of column 3 and column 4 using string similarity
compare_cl.string('col3', 'col3', label = 'col3')
compare_cl.string('col4', 'col4', label = 'col4')

# find matches
potential_matches = compare_cl.computer(pairs, df1, df2)

print(potential_matches)

# finding the only pairs we want
potential_matches[potential_matches.sum(axis = 1) => 2]
```

## Linking DataFrames

What we have already done
```python
# import recordlinkage and generate full pairs
import recordlinkage
indexer = recordlinkage.Index()
indexer.block('col')
full_pairs = indexer.index(df1, df2)

# comparison step
compare_cl = recordlinkage.Compare()
compare_cl.extract('col1', 'col1', label = 'col1')
compare_cl.extract('col2', 'col2', label = 'col2')
compare_cl.string('col3', 'col3', label = 'col3')
compare_cl.string('col4', 'col4', label = 'col4')

potential_matches = compare_cl.computer(full_pairs, df1, df2)

```
Probable matches
```python
matches = potential_matches[potential_matches.sum(axis = 1) => 3]
print(matches)

# get the indices
matches.index

# get indices from df2 only
duplicate_rows = matches.index.get_level_values(1)
```
Linking DataFrames
```python
# finding duplicates in df2
df2_duplicates = df2[df2.index.isin(duplicate_rows)]

# finding new rows in df2
df2_new = df2[~df2.index.isin(duplicate_rows)]

# link the DataFrames
full_df = df1.append(df2_new)
```

```python
# import recordlinkage, generate pairs and compare across columns
...

# generate potential matches
potential_matches = compare_cl.compute(full_pairs, df1, df2)

# isolate matches with matching values for 3 or more columns
matches = potential_matches[potential_matches.sum(axis = 1) >= 3]

# get index for matching df2 rows only
duplicate_rows = matches.index.get_level_values(1)

# find new rows in df2
df2_new = df2[~df2.index.isin(duplicate_rows)]

# link the DataFrames
full_df = df1.append(df2_new)
```
