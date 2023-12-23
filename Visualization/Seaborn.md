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
