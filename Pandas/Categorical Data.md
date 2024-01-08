# Categorical Types
Categorical
- Finite number of groups (or categories)
- These categories are usually fixed or known (eye color, hair color, etc.)
- Known as qualitive data

Numerical
- Known as quantitative data
- Expressed using a numerical value
- Is usually a measurement (heigh, weight, IQ, etc.)

```python
df = pd.read_csv("data.csv")
df.dtypes
```

```python
# default dtype
df["col"].dtype # dtype('0')
# set as categorical
df["col"] = df["col"].astype('category')
```

Creating a categorical Series
```python
my_data = ['A', 'B', 'C', 'B', 'C', 'A']

my_series1 = pd.Series(my_data, dtype='category') # [A, B, C]
my_series2 = pd.Categorical(my_data, categories=['C', 'B', 'A'], ordered=True) # [C < B < A]
```

```python
# memory saving from converting to categorical
df['col'].nbytes
```

Specify dtypes when reading data
```python
# create a dictionary
df_dtypes = {'key' : 'value'}
# set the `dtype` parameter
df = pd.read_csv('data.csv', dtype=df_dtypes)
# check the `dtype`
df['col'].dtype
```

# Grouping data by category
```python
# splitting data
df = pd.read_csv('data.csv')
filter1 = df[df['col'] == 'condition']
filter2 = df[df['col'] == 'condition']
# replaced by
groupby_object = df.groupby(by=['col'])
```

Applying a function to .groupby()
```python
groupby_object.mean()
# one liner
df.groupby(by=['col']).mean()
```

Specifying columns
```python
# option 1, preferred with large datasets
df.groupby(by=['col'])['col1','col2'].sum()
# option 2
df.groupby(by=['col']).sum()['col1','col2']
```

Groupby multiple columns
```python
df.groupby(by=['col1', 'col2']).size()
```

# Setting category variables
The .cat accessor object:
`Series.cat.method_name`

Common parameters:
- `new_categories`: a list of categories
- `inplace`: Boolean - whether or not the update should overwrite the Series
- `ordered`: whether or not the categorical is treated as an ordered categorical

Setting Series categories
- Setting: `cat.set_categories()`
  - Can be used to set the order of categories
  - All values not specified in this method are dropped

```python
# set categories
df['col'] = df['col'].cat.set_categories(new_categories=['cat1', 'cat2', 'cat3'])
# check value counts
df['col'].value_counts(dropna=False)
```

Setting order
```python
df['col'] = df['col'].cat.set_categories(new_categories=['cat1', 'cat2', 'cat3'], ordered=True) # ['shorts' < 'medium' < 'long']
```

Missing categories
```python
df['col'].value_counts(dropna=False) # NaN could mean different things
```

Adding categories
- Adding: `cat.add_categories()`
  - Does not change the value of any data in the DataFrame
  - Categories not listed in this method are left alone

```python
# add categories
df['col'] = df['col'].astype('category')
df['col'] = df['col'].cat.add_categories(new_categories=['cat1', 'cat2'])
# check categories
df['col'].cat.categories
# new categories
df['col'].value_counts(dropna=False)
```

Removing categories
- Removing: `cat.remove_categories()`
  - Values matching categories listed are set to `NaN`
    
```python
df['col'] = df['col'].astype('category')
df['col'] = df['col'].cat.remove_categories(removals=['category'])
# check categories
df['col'].cat.categories
```

# Updating categories
The `rename_categories` method:
```python
series.cat.rename_categories(new_categories=dict)
```
Make a dictionary:
```python
my_changes = {'cat':'new_cat'}
```
Rename the category:
```python
df['col'] = df['col'].cat.rename_categories(my_changes)
```
