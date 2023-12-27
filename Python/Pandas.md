# Modules
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
```

```python
# Row index
df.index
# Column index
df.columns
```
## Series
Ordered, one-dimensional array of data with an index.
```python
s1 = Series(range(0,4)) # -> 0, 1, 2, 3
s2 = Series(range(1,5)) # -> 1, 2, 3, 4
s3 = s1 + s2 # -> 1, 3, 5, 7
s4 = Series(['a','b']) * 3 # -> 'aaa', 'bbb'
```
## Index
Provides axis labels for the Series and DataFrame objects. Can only contain hashable objects.
```python
# get Index from Series and DataFrame
idx = s.index 
idx = df.columns # column index
idx = df.index # row index
```

```python
# Index attributes
b = idx.is_monotonic_decreasing
b = idx.is_monotonic_increasing
b = idx.has_duplicates
i = idx.nlevels # multi-level indexes
# Index methods
a = idx.values() # get as numpy array
l = idx.tolist() # get as python list
idx = idx.astype(dtype) # change data type
b = idx.equals(o) # check for equality
idx = idx.union(o) # union of two indexes
i = idx.nunique() # number of unique labels
label = idx.min() # minimum label
label = idx.max() # maximum label
```
## Loading Data
```python
df = pd.read_csv('file.csv')
df = pd.read_csv('file.csv', header=0, index_col=0, quotechar='"', sep=':', na_values= ['na', '-', '.', ''])
```
Inline CSV text to a DataFrame
```python
from io import StringIO
data = """, Animal, Cuteness, Desirable,
row-1, dog, 8.7, True
row-2, bat, 2.6, False """
df = read_csv(StringIO(data), header=0, index_col=0, skipinitialspace=True)
```
Excel file to a DataFrame
```python
workbook = pd.ExcelFile('file.xlsx')
dictionary = {}
for sheet_name in workbook.sheet_names:
  df = workbook.parse(sheet_name)
  dictionary[sheet_name] = df
```
MySQL to a DataFrame
```python
import pymysql
from sqlalchemy import create_engine
engine = create_engine('mysql+pymysql://' + 'USER:PASSWORD@localhost/DATABASE')
df = pd.read_sql_table('table', engine)
```
Combining Series into a DataFrame
```python
s1 = Series(range(0,6))
s2 = s1 * s1
s2.index = s2.index + 2 # misalign indexes
df = pd.concat([s1,s2], axis=1)
# Example 2
s3 = Series({'Tom':1, 'Dick':4, 'Har':9})
s4 = Series({'Tom':3, 'Dick':2, 'Mar':5})
df = pd.concat({'A':s3, 'B':s4}, axis=1)
```
## Saving a DataFrame
```python
# to CSV
df.to_csv('name.csv', encoding='utf-8')
# to Excel
from pandas import ExcelWriter
writer = ExcelWriter('filename.xlsx')
df1.to_excel(writer, 'Sheet1')
df2.to_excel(writer, 'Sheet2')
writer.save()
# to MySQL
import pymysql
from sqlalchemy import create_engine
e = create_engine('mysql+pymysql://' + 'USER:PASSWORD@localhost/DATABA
df.to_sql('TABLE', e, if_exists='replace')
# to a Python Dictionary
dictionary = df.to_dict()
# to a Python String
string = df.to_string()
```
## DataFrame Contents
```python
df.info() # index and data types
n = 4
dfh = df.head(n) # get first n rows
dft = df.tail(n) # get last n rows
dfs = df.describe() # summary stats cols
top_left_corner_df = df.iloc[:5,:5]
```
Non-indexing attributes
```python
dfT = df.T # transpose rows and columns
l = df.axes # list rows and col indexes
(r,c) = df.axes # from above
s = df.dtypes # Series column data types
b = df.empty # True for empty DataFrame
i = df.ndim # number of axes (2)
t = df.shape # (row-count, column-count)
(r,c) = df.shape # from above
i = df.size # column-count
a = df.values # get a numpy array for df
```
Utility methods
```python
dfc = df.copy() # copy a DataFrame
dfr = df.rank() # rank each col (default
dfs = df.sort() # sort each col (default)
dfc = df.astype(dtype) # type conversion
```
Iteration methods
```python
df.iteritems()
df.iterrows()
# example: iterating over columns
for (name, series) in df.iteritems():
  print('Col name: ' + str(name)
  print('First value: ' + str(series.iat[0]) + '\n')
```
Math on the whole DataFrame
```python
df = df.abs() # absolute value
df = df.add(o) # add df, Series or value
s = df.count() # non NA/Null values
df = df.cummax() # (cols default axis)
df = df.cummin() # (cols default axis)
df = df.cumsum() # (cols default axis)
df = df.cumprod() # (cols default axis)
df = df.diff() # 1st diff (col def axis)
df = df.div(o) # div by df, Series, value
df = df.dot(o) # matrix dot product
df = df.max() # max of axis (col def)
s = df.mean() # mean (col default axis)
s = df.median() # median (col default axis)
s = df.min() # min of axis (col def)
df = df.mul(o) # mul by df Series val
s = df.sum() # sum axis (cols default)
```
Filters
```python
df = df.filter(items=['a','b']) # by col
df = df.filter(items=[5], axis=0) # by row
df = df.filter(like='x') # keep x in col
df = df.filter(regex='x') # regex in col
df = df.select(crit-(lamba x: not x%5)) # rows
```
## Working with Columns
Column index and labels
```python
idx = df.columns # get col index
label = df.columns[0] # 1st col label
lst = df.columns.tolist() # get as a list
# Change column labels
df.rename(columns={'old':'new'}, inplace=True)
df = df.rename(columns={'a':1, 'b':'x'})
```
Selecting columns
```python
s = df['colName'] # select col to Series
df = df[['colName']] # select col to df
df = df[['a', 'b']] # select 2 or more
df = df[['c', 'a', 'b']] # change order
s = df[df.columns[0]] # select by number
df = df[df.columns[0, 3, 4]] # by number
s = df.pop('c') # get col and drop from df
# Selecting columns with Python attributes
s = df.a # same as s = df['a']
# cannot create new columns by attribute
df.existing_col = df.a / df.b
df['new_col'] = df.a / df.b
```
Adding new columns
```python
df['new_col'] = range(len(df))
df['new_col'] = np.repeat(np.nan, len(df))
df['random'] = np.random.rand(len(df))
df['index_as_col'] = df.index
df1[['b','c']] = df2[['e','f']]
df3 = df1.append(other=df2)
```

```python
# Swap column contents - change colum order
df[['B','A']] = df[['A','B']]
# Dropping columns
df = df.drop('col1', axis=1)
df.drop('col1', axis=1, inplace=True)
df = df.drop(['col1', 'col2'], axis=1)
s = df.pop('col') # drops from frame
del col['col']
df.drop(df.columns[0], inplace=True)
```
Arithmetic
```python
# Vectorized arithmetic on columns
df['proportion'] = df['count'] / df['total']
df['percent'] = df['proportion'] * 100
# Apple numpy math functions to columns
df['log_data'] = np.log(df['col1'])
df['rounded'] = np.round(df['col2'], 2)
```

```python
# Columns set based on criteria
df['b'] = df['a'].where(df['a'] > 0, other = 0)
df['d'] = df['a'].where(df.b != 0, other = df.c)
# Data type conversions
s = df['col'].astype(str) # Series type
na = df['col'].values # numpy array
pl = df['col'].tolist() # python list
```
Column-wide methods/attributes
```python
value = df['col'].dtype # type of data
value = df['col'].size # col dimensions
value = df['col'].count() # non-NA count
value = df['col'].sum()
value = df['col'].prod()
value = df['col'].min()
value = df['col'].max()
value = df['col'].mean()
value = df['col'].median()
value = df['col'].cov(df['col2'])
s = df['col'].describe()
s = df['col'].value_counts()
```

```python
# Index label for mix/max values
label = df['col1'].idxmin()
label = df['col1'].idxmax()
```
Element-wise methods
```python
s = df['col'].isnull()
s = df['col'].notnull() # not isnull()
s = df['col'].astype(float)
s = df['col'].round(decimals=0)
s = df['col'].diff(periods=1)
s = df['col'].shift(periods=1)
s = df['col'].to_datetime()
s = df['col'].fillna(0) # replace NaN with 0
s = df['col'].cumsum()
s = df['col'].cumprod()
s = df['col'].pct_change(periods=4)
s = df['col'].rolling_sum(periods=4, window=4)
```

```python
# Append column of row sums to a DataFrame
df['Total'] = df.sum(axis=1)
# Multiply every column in DataFrame by Series
df = df.mul(s, axis=0) # on matched rows
# Selecting columns with .loc, .iloc and .ix
df = df.loc[:, 'col1':'col2'] # inclusive
df = df.iloc[:, 0:2] # exclusive
# Get interger pisition of a column index label
j = df.columns.get_loc('col_name')
```

```python

```

```python

```
