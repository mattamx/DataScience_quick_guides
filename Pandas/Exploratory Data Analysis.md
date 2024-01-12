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
- Words of interest: Any that start with a specific string
```python
df["col"].str.contains("^specific string") # Boolean table
```
Finding multiple phrases in strings
```python
example_categories = ["string1", "string2", "string3", "string4"]

category1 = "string1|string2"
category2 = "string3|string4"
category3 = "string5|string6|string7|string8"

example_conditions = [
          (df["col"].str.contains(category1)),
          (df["col"].str.contains(category2))
          (df["col"].str.contains(category3))
```
Creating the categorical column
```python
df["new_col"] = np.select(example_conditions, example_categories, default="Other")

# preview
print(df[["col","new_col"]].head())
```
Visualizing new column frequency
```python
sns.countplot(data=df, x="new_col")
plt.show()
```

# Working with numerical data

Converting strings to numbers
```python
pd.Series.str.replace("characters to remove", "characters to replace them with")

# fixing a string
df["col"] = df["col"].str.replace(",", "") # replacing commas
print(df["col"].head())

# converting to float type
df["col"] = df["col"].astype(float)

# converting to required metric
df["new_col"] = df["col"] * (percentage or decimal)

# preview
print(df[["col", "new_col"]].head())
```
Adding summary statistics into a DataFrame
```python
df.groupby("col")["col1"].mean()

# standard deviation example using lambda
df["std_dev"] = df.groupby("col")["col1"].transform(lambda x: x.std()) # for each x, transform to the respective standard deviation

# preview
print(df[["col","std_dev"]].value_counts())

# median example using lambda
df["median_by_col"] = df.groupby("col")["col1"].transform(lambda x: x.median()) # for each x, transform to the respective median

# preview
print(df[["col","median_by_col"]].head())
```

# Handling outliers
- An observation far away from other data points
- Should always consider why there is an outlier

Using descriptive statistics
```python
print(df["col"].describe())
```
Using the interquartile range 
- Interquartile range (IQR)
  - IQR = 75th - 25th percentile
  - Upper Outliers > 75th percentile + (1.5 * IQR)
  - Lower Outliers < 25th percentile - (1.5 * IQR)

IQR in box plots
```python
sns.boxplot(data=df, y="col")
plt.show()
```
Identifying thresholds
```python
# 75th percentile
seventy_fifth = df["col"].quantile(0.75)

# 25th percentile
twenty_fifth = df["col"].quantile(0.25)

# Interquartile range
df_iqr = seventy_fifth - twenty_fifth

print(df_iqr)
```
Identifying outliers
```python
# Upper threshold
upper = seventy_fifth + (1.5 * df_iqr)

# Lower threshold
lower = twenty_fifth - (1.5 * df_iqr)

print(upper, lower)
```
Subsetting the data
```python
df[(df["col"] < lower) | (df["col"] > upper)] \
      [["col1", "col2", "col3"]] # columns to display
```
Why look for outliers:
- Outliers are extreme values
  - may not accurately represent the data
- Can change the mean and standard deviation
- Statistical test and machine learning models need normally distributed data

Questions to ask:
- Why do the outliers exist?
  - Consider leaving them in the dataset depending on they why
- Is the data accurate?
  - Could there have been an error in data collection?
    - If so, remove them
   
Dropping outliers
```python
no_outliers = df[(df["col"] > lower) | (df["col"] < upper)]

print(no_outliers["col"].describe())
```

# Patterns over time
Importing DateTime data
- DateTime data needs to be explicitly declared to Pandas

```python
df = pd.read_csv('data.csv', parse_dates=["col"])
df.dtypes
```
Converting to Datetime data
- `pd.to_datetime()` converst arugments to DateTime data
```python
df["col"] = pd.to_datetime(df["col"])
df.dtypes
```
Creating DateTime data
```python
df["col"] = pd.to_datetime(df[["month", "day", "year"]])

# extracting parts of a full date using `dt.month`, `dt.day`, and `dt.year` attributes
df["month_col"] = df["col"].dt.month
df["day_col"] = df["col"].dt.day
df["year_col"] = df["col"].dt.year
```

# Correlation
- Describes direction and strength of relationship between two variables
- Can help us use variables to predict future outcomes
```python
df.corr()
```
Correlation heatmaps
```python
sns.heatmap(df.corr(), annot=True)
plt.show()
```
Correlation in context
```python
df["date_col"].min()
df["date_col"].max()
```
Scatter plots
```python
sns.scatterplot(data=df, x="col1", y="col2")
plt.show()
```
Pairplots
```python
sns.pairplot(data=df)
plt.show()

# multiple variables
sns.pairplot(data=df, vars=["col1", "col2", "col3"])
plt.show()
```
