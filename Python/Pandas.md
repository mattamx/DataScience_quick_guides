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
Swap column contents - change colum order
```python
df[['B','A']] = df[['A','B']]
```
Dropping columns
```python
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
Columns set based on criteria
```python
df['b'] = df['a'].where(df['a'] > 0, other = 0)
df['d'] = df['a'].where(df.b != 0, other = df.c)
```
Data type conversions
```python
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
Index label for mix/max values
```python
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
## Working with Rows
Get the row index and labels
```python
idx = df.index # get row index
label = df.index[0] # 1st row label
lst = df.index.tolist() # get as a list
```
Change the (row) index
```python
df.index = idx # new adhoc index
df.index = range(len(df)) # set with list
df = df.reset_index() # replace old with new
df['b'] = df # old index stored as a column in df
df = df.reindex(index=range(len(df)))
df = df.set_index(key=['r1', 'r2', 'etc'])
df.rename(index={'old':'new'}, inplace=True)
```
Adding rows
```python
df = original_df.append(more_rows_in_df) # both DataFrames should have the same column labels
```
Dropping rows (by name)
```python
df = df.drop('row_label')
df = df.drop(['row1','row2']) # multi-row
```
Boolean row selection by values in a column
```python
df = df[df['col2'] >= 0]
df = df[(df['col3'] >= 1) | (df['col1'] < 0)]
df = df[df['col2'].isin([1,2,5,7,11])]
df = df[~df['col2'].isin([1,2,5,7,11])]
df = df[df['col'].str.contains('helo')]
```
Selecting rows using isin() over multiple columns
```python
data = {1:[1,2,3], 2:[1,4,9], 3:[1,8,27]}
df = pd.DataFrame(data)
# multi-column isin()
lf = {1:[1,3], 3:[8,27]} # look for
f = df[df[list(lf)].isin(lf).all(axis=1)]
# Selecting rows using an index
idx = df[df['col'] >= 2].index
print(df.ix[idx])
```
Select a slice of rows by integer position
```python
df = df[:] # copy DataFrame
df = df[0:2] # rows 0 and 1
df = df[-1:] # the last row
df = df[2:3] # row 2 (the third row)
df = df[:-1] # all but the last row
df = df[::2] # every 2nd row (0 2 ..)
# Selecting a slice of rows by label/index
df = df['a':'c'] # rows 'a' through 'c' # doesn't work on integer labelled rows
```
Append a row of column totals to a DataFrame
```python
# Option 1: use dictionary comprehension
sums = {col: df[col].sum() for col in df}
sums_df = DataFrame(sums, index=['Total'])
df.append(sums_df)
# Option 2: All done with pandas
df = df.append(DataFrame(df.sum(), columns=['Total']).T)
```
Iterating over DataFrame Rows
```python
for (index, row) in df.iterrows(): # pass
```
Sorting DataFrame rows values
```python
df = df.sort(df.columns[0], ascending=False)
df.sort(['col1','col2'], inplace=True)
```
Random selection of rows
```python
import random as r
k = 20 # pick a number
selection = r.sample(range(len(df)), k)
df_sample = df.iloc[selection,:]
```

Sort DataFrame by its row index
```python
df.sort_index(inplace=True) # sort by row
df = df.sort_index(ascending=False)
```
Drop duplicates in the row index
```python
df['index'] = df.index # 1 create new column
df = df.drop_duplicates(cols='index', take_last=True) # 2 use new column
del df['index'] # 3 delete the column
df.sort_index(inplace=True) # 4 tidy up
```
Test if two DataFrames have same row index
```python
len(a) == len(b) and all(a.index == b.index)
```
Get the integer prosition of a row and a col index label
```python
i = df.index.get_loc('row_label')
```
Get integer position of rows that meet condition
```python
a = np.where(df['col'] >= 2) # numpy array
```
## Working with Cells
Selecting a cell by row and column labels
```python
value = df.at['row', 'col']
value = df.loc['row', 'col']
value = df['col'].at['row'] # tricky

```
Setting a cell by row and column labels
```python
df.at['row', 'col']
df.loc['row', 'col']
df['col'].at['row'] # tricky
```
Selecting and slicing on labels
```python
df = df.loc['row1':'row3', 'col1':'col3'] # inclusive
```
Setting a cross-section by labels
```python
df.loc['A':'C', 'col1':'col3'] = np.nan
df.loc[1:2, 'col1':'col3'] = np.zeros((2,2))
df.loc[1:2 , 'A':'C'] = othr.loc[1:2, 'A':'C']
```
Selecting a cell by integer position
```python
value = df.iat[9, 3] # [row,col]
value = df.iloc[0, 0] # [row,col]
value = df.iloc[len(df)-1, len(df.columns)-1]
```
Selecting a range of cells by int position
```python
df = df.iat[2:4, 2:4] # subset of the df
df = df.iloc[:5, :5] # top left corner
s = df.iloc[5, :] # return row as Series
df = df.iloc[5:6, :] # return row as row
```
Setting cell by integer position
```python
df.iloc[0, 0] = value # [row,col]
df.iat[7, 8] = value
```
Setting cell range by integer position
```python
df.iloc[0:3, 0:5] = value
df.iloc[0:3, 0:5] = np.ones((2, 3))
df.iloc[0:3, 0:5] = np.zeros((2, 3))
df.iloc[0:3, 0:5] = np.array([[1, 1, 1], [2, 2, 2]])
```
.ix for mixed label and integer position indexing
```python
value = df.ix[5, 'col1']
df = df.ix[1:5, 'col1':'col3']
```

## Joining/Combining DataFrames
There are three ways to join two DataFrames:
- merge (a database/SQL-like join operation)
- concat (stack side by side or one on top of each other)
- combine_first (splice the two together, choosing values from one over the other)

Merge on indexes
```python
df_new = pd.merge(left=df1, right=df2, how='outer', left_index=True, right_index=True)
# How: 'left', 'right', 'outer', 'inner'
# How: outer = union/all; inner = intersection
```
Merge on columns
```python
df_new = pd.merge(left=df1, right=df2, how='left', left_on='col1', right_on='col2') # when joining on columns, the indexes on the passde DataFrame are ignored
```
Join on indexes
```python
# DataFrame.join() joins on indexes by default
df_new = df1.join(other=df2, on='col1', how='outer')
df_new = df1.join(other=df2, on=['a', 'b'], how='outer')
```
Concatenation
```python
df = pd.concat([df1,df2], axis=0) # top/bottom
df = df1.append(df2,df3]) # top/bottom
df = pd.concat([df1,df2], axis=1) # left/right
# Can lead to duplicate rows or cols
```
Combine_first
```python
df = df1.combine_first(other=df2)
# multi-combine with python reduce()
df = reduce(lambda x, y: x_combine_first(y), [df1,df2,df3,df4,df5]) # index of the combined DataFrame will be the union of the indexes from df1 and df2
```

## Groupby: Split-Apply-Combine
Grouping
```python
gb = df.groupby('cat') # by one column
gb = df.groupby(['c1','c2']) # by 2 columns
gb = df.groupby(level=0) # multi-index gb
gb = df.groupby(level=['a','b']) # multi-index gb
print(gb.groups) # contains a dictionary of mapping of the groups
```
Iterating groups
```python
for name, group in gb:
  print(name)
  print(group)
```
Selecting a group
```python
dfa = df.groupby('cat').get_group('a')
dfb = df.groupby('cat').get_group('b')
```
Applying an aggregation function
```python
# apply to a column
s = df.groupby('cat')['col1'].sum()
s = df.groupby('cat')['col1'].agg(np.sum)
# apply to every column in the DataFrame
s = df.groupby('cat').agg(np.sum)
df_summary = df.groupby('cat')['col1'].describe()
df_row_1s = df.groupby('cat')['col1'].head(1)
```
Applying multiple aggregation functions
```python
gb = df.groupby('cat')
# apply multiple functions to one column
dfx = gb['col2'].agg([np.sum, np.meam])
# apply multiple functions to multiple columns
dfy = gb.agg({'cat': np.count_nonzero,
              'col1': [np.sum, np.meam, np.std],
              'col2': [np.min, np.max]
              )
```
Transforming functions
```python
# tranform to group z-scores, which have a group mean of 0 and std dev of 1
zscore = lambda x: (x-x.mean()) / x.std()
dfx = df.groupby('cat').transform(zscore)
# replace missing data with group mean
mean_r = lambda x: x.fillna(x.mean())
dfm = df.groupby('cat').transform(mean_r)
```
Applying filtering functions
```python
# Allows you to make selections based on whether each group meets specific criteria
# select groups with more than 10 members
eleven = lambda x: (len(x['col1']) >= 11)
df11 = df.groupby('cat').filter(eleven)
```
Group by a row index (non-hierarchical)
```python
df = df.set_index(keys='cat')
s = df.groupby(level=0)['col1'].sum()
dfg = df.groupby(level=0).sum()
```
