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
df["col"].isin(["string1" ,"string2"]) # Boolean table

~df["col"].isin(["string1" ,"string2"]) # Boolean table (tilde; 'is not in')

df[df["col"].isin(["string1" ,"string2"]).head() # Table subset
```
Validating numerical data
```python
df.select_dtypes("number").head()

df["col"].min()

df["col"].max()

sns.boxplot(data=df, x="col")
plt.show()
```

# Data Summarization
Exploring groups of data
- `.groupby()` groups data by category
- Aggregating function indicates how to summarize grouped data
```python
df.groupby("col").mean()
```
Aggregating functions
- Sum: `.sum()`
- Count: `.count()`
- Minimum: `.min()`
- Maximum: `.max()`
- Variance: `.var()`
- Standard Deviation: `.std()`

Aggragating ungrouped data
- `.agg()` applies aggregating functions across a DataFrame
```python
df.agg(["mean", "std"])
```
Specifying aggragations for columns
```python
df.agg({"col1": ["mean", "std"], "col2": ["median"]})
```
Named summary columns
```python
df.groupby("col1").agg(
  mean_col = ("col2", "mean"),
  std_col = ("col2", "std"),
  median_col = ("col3", "std")
```
Visualizing categorical summaries
```python
sns.barplot(data=df, x="col1", y="col2")
plt.show()
```

# Addressing missing data
- Affects distributions
  - Missing data for specific column
- Less representative of the population
  - Certain groups disproportionately represented, e.g., lacking data on specific column
- Can result in drawing incorrect conclusions

Checking for missing values
```python
print(df.isna().sum())
```

Strategies for addressing missing values
- Drop missing values
  - 5% or less of total values
- Impute mean, median, mode
  - Depends on distribution and context
- Impute by sub-group

Dropping missing values
```python
threshold = len(df) * 0.05
print(threshold)

cols_to_drop = df.columns[df.isna().sum() <= threshold]
print(cols_to_drop)

df.dropna(subset=cols_to_drop, inplace=True)
```
Imputting a summary statistic
```python
cols_with_missing_values = df.columns[df.isna().sum() > 0]
print(cols_with_missing_values)

for col in cols_with_missing_values[:-1]:
    df[col].fillna(df[col].mode()[0])
```
Checking for remaining missing values
```python
print(df.isna().sum()
```
Imputing by sub-group
```python
df_dict = df.groupby("col1")["col2"].median().to_dict()
print(df_dict)

df["col2"] = df["col2"].fillna(df["col1"].map(df_dict))
```

# Converting and analyzing categorical data
Previewing the data
```python
print(df.select_dtypes("object").head())

print(df["col"].value_counts())

print(df["col"].nunique())
```
Extracting value from categories
- Current format limits ability to generate insights
- `pandas.Series.str.contains()`
  - Search a column for a specific string or multiple strings
```python
df["col"].str.contains("string") # Boolean table
```
Finding multiple phrases in strings
- Words of interest: "string1" or "string2"
```python
df["col"].str.contrains("string1|string2") # Boolean table
```
Finding multiple phrases in strings
- Words of interest: Any that start with Data
```python
df["col"].str.contains("^Data") # Boolean table
```
Finding multiple phrases in strings
```python

```
Creating the categorical column
```python

```
Previewing categories
```python

```
