# Introduction to Regression

Descriptive statistics
```python
import pandas as pd

print(df.mean())

print(df['col1'].corr(df['col2']))
```

What is regression?
- Statistical models to explore the relationship between a response variable and some explanatory variables.
- Given values of explanatory variables, you can predict the values of the response variable.

**Response variable (a.k.a. dependent variable)**
- The variable that you want to predict.

**Explanatory variables (a.k.a. independent variables)**
- The variables that explain how the response variable will change.

**Linear and Logistic regression**
- Linear Regression
  - The response variable is numeric.

- Logistic Regression
  - The response variable is logical.

- Simple linear/logistic regression
  - There is only one explanatory variable.
 
**Python packages for regression**
`statsmodels`
- Optimized for insight

`scikit-learn`
- Optimized for prediction

Visualizing pairs of variables
```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(x='col1', y='col2', data=df)

plt.show()
```
Adding a linear trend line
```python
sns.regplot(x='col1', y='col2', data=df, ci=None)
```

## Fitting a linear regression

Straight lines are defined by two things
**Intercept**
The ***y*** value at the point when ***x*** is zero.

**Slope**
The amount of the ***y*** value increases if you increase ***x*** by one.

**Equation**
***y = intercept + slope * x***

Running a model
```python
from statsmodels.formula.api import ols

model = old('col1 ~ col2', data=df) # response variable comes first

model = model.fit()

print(model.params) # gives you intercept and slope
```

## Creating explanatory variables

Visualizing 1 numeric and 1 categorical variable
```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.displot(data=df, x='col1', col='col2', col_wrap=2, bins=9)

plt.show()
```

Summary statistics: mean by groupby
```python
summary_stats = df.groupby('col1')['col2'].mean()

print(summary_stats)
```

Linear regression
```python
from statsmodels.formula.api import ols

model = ols('col1 ~ col2', data=df).fit()

print(model.params)
```

Model with or without an intercept
```python
# coefficients relative to the intercept
model_with = ols('col1 ~ col2', data=df).fit()
print(model_with)

# coefficients are the means
model_without = ols('col1 ~ col2 + 0', data=df).fit()
print(model_without)

```

```python

```

```python

```

```python

```

# Making predictions

```python

```

## Working with model objects

```python

```

## Regression to the mean

```python

```

## Transforming variables

```python

```
