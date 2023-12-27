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
