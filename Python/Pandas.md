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
engine = create_engine('mysql+pymysql"//' + 'USER:PASSWORD@localhost/DATABASE')
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

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```
