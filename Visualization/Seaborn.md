# Getting started
```python
import seaborn as sns
import matplotlib.plt as plt
```
# Plots
## Scatter plot
```python
sns.scatterplot(x='column_name', y='column_name')
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
sns.scatterplot(x='column_name', y='column_name',
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
sns.relplot(x='column_name', y='column_name', data=df, kind='scatter',
            col='column_name', col_wrap='number of columns per row', col_order=["list_of_values","list_of_values","list_of_values"],
            row='column_name', row_order=["list_of_values","list_of_values","list_of_values"])
plt.show()
```

```python
sns.relplot(x='column_name', y='column_name', data=df, kind='scatter',
            size="quantitative or categorical variable", hue="quantitative or categorical variable",
            style="column_name", alpha='value between 0 and 1')
plt.show()
```

```python
sns.relplot(x='column_name', y='column_name', data=df, kind='line',
            hue="quantitative or categorical variable",
            style="column_name", markers='True or False', dashes='True or False',
            ci='sd or None') # sd = standard deviation
plt.show()
```

## Categorical plots
```python
category_order = [list]
sns.catplot(x='column_name', data=df, kind='count', order=category_order)
plt.show()
```

```python
sns.catplot(x='column_name', y='column_name', data=df, kind='bar', ci='None')
plt.show()
```

```python
sns.catplot(x='column_name', y='column_name', data=df, kind='box', order=['list'],
            sym=" ", # sym omits or changes appearance of outliers
            whis= [lower, upper] # changing the whiskers
            ) 
plt.show()
```

```python
from numpy import median
sns.catplot(x='column_name', y='column_name', data=df, kind='point', hue='column_name',
            join='True or False',
            estimator=median, # better statistic if data has many outliers
            capsize='value', ci='None')
plt.show()
```

# Style
Five preset figure styles: "white", "dark", "whitegrid", "darkgrid", "ticks"
```python
sns.set_style()
```

Diverging palettes: "RdBu", "PRGn", "RdBu_r", "PRGn_r"
- underscore reverses the palette

Sequential palettes: "Greys", "Blues", "PuRd", "GnBu"
```python
sns.set_palette()
```
## Scale
Smalles to largest: "paper", "notebook", "talk", "poster"
```python
sns.set_context()
```

# Titles and labels
## FacetGrid vs AxesSubplot objects
| Object Type | Plot Types | Characteristics |  
| ------- | ------- | ------- | 
| `FacetGrid` |  `relplot()`, `catplot()` | Can create subplots | 
| `AxesSubplot` | `scatterplot()`, `countplot()`, etc. | Only creates a single plot | 

```python
g = sns.scatterplot(x='column_name', y='column_name', data=df)
type(g) # AxesSubplot
```

### Titles
```python
# FacetGrid
g = sns.scatterplot(x='column_name', y='column_name', data=df, kind='box')
g.fig.suptitle('New Title',
              y='number' # adjusts height
              )
```

```python
# AxesSubplot
g = sns.scatterplot(x='column_name', y='column_name', data=df)
g.set_suptitle('New Title',
              y='number' # adjusts height
              )
```

```python
g = sns.catplot(x='column_name', y='column_name', data=df, kind='box', col='column_name)
g.fig.suptitle('New Title',
              y='number' # adjusts height
              )
g.set_titles('This is {column_name}') # sets titles for subplots
```

### Labels
```python
g = sns.catplot(x='column_name', y='column_name', data=df, kind='box')
g.set(xlabel='New label', ylabel='New label')
```

```python
g = sns.catplot(x='column_name', y='column_name', data=df, kind='box')
plt.xticks(rotation='number') # rotates tick labels
```

Adding data labels
```python
ax = g.facet_axis(0, 0)
for c in ax.containers:
    labels = [f'{(v.get_height()):.2f}%' for v in c]
    ax.bar_label(c, labels=labels, label_type='center', color='white')
```
