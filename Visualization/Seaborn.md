# Getting started
```python
import seaborn as sns
import matplotlib.plt as plt
```
# Plots
## Scatter plot
```python
sns.scatterplot(x='array1', y='array2')
plt.show()
```

## Count plot
```python
sns.countplot(x='categorical_list')
plt.show()
```

### DataFrames with countplot()
```python
df = pd.read_csv('path')
sns.countplot(x='categorical list', data=df)
plt.show()
```
## Hue
```python
hue_colors = {"list_of_values":"black", "list_of_values":"red"}
sns.scatterplot(x='array1', y='array2',
                hue='values or column_name',
                hue_order=['list_of_values','list_of_values'],
                palette=hue_colors)
plt.show()
```

```python
sns.scatterplot(x='column_name_one',
                data=df,
                hue='column_name_two')
plt.show()
```

### Hue with HTML hex color codes
```python
hue_colors = {"list_of_values":"#808080", "list_of_values":"#00FF00"}
sns.scatterplot(x='array1', y='array2',
                hue='values or column_name',
                hue_order=['list_of_values','list_of_values'],
                palette=hue_colors)
plt.show()
```

## Relational plots
Creating subplots in a single figure
```python
sns.relplot(x='array1', y='array2', data=df, kind='scatter',
            col='column_name', col_wrap='number of columns per row', col_order=["list_of_values","list_of_values","list_of_values"],
            row='column_name', row_order=["list_of_values","list_of_values","list_of_values"])
plt.show()
```

```python
sns.relplot(x='array1', y='array2', data=df, kind='scatter',
            size="quantitative or categorical variable", hue="quantitative or categorical variable",
            style="column_name", alpha='value between 0 and 1')
plt.show()
```

```python
sns.relplot(x='array1', y='array2', data=df, kind='line',
            hue="quantitative or categorical variable",
            style="column_name", markers='True or False', dashes='True or False',
            ci='sd or None') # sd = standard deviation
plt.show()
```
