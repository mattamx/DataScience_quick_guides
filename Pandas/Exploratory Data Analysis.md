# Initial exploration

A first look with .head()
```python
df = pd.read_csv("data.csv")
df.head() # shows 10 records
```
Gathering more .info()
```python
df.info() # concise summary of the df
```
A closer look at columns
```python
df.value_counts("col") #  frequency of each distinct row in the df
```
.describe() numerical columns
```python
df.describe() # descriptive statistics
```

# Data validation
Validating data types
```python
df.dtypes
```
Updating data types

Types:
- String: `str`
- Integer: `int`
- Float: `float`
- Dictionary: `dict`
- List: `list`
- Boolean: `bool`
```python
df["col"] = df["col"].astype(int) # updating to integer
df.dtypes
```
Validating categorical data
```python
books["col"].isin(["string1" ,"string2"]) # Boolean table
```
