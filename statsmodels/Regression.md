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

# Making predictions

Data on explanatory values to predict
```python
explanatory_data = pd.DataFrame({'col': np.arange(int, int})

print(model.predict(explanatory_data))
```
Predicting inside a DataFrame
```python
explanatory_data = pd.DataFrame({'col': np.arange(int, int}

predict_data = explanatory_data.assign(column_name = model.predict(explanatory_data))
```
Showing predictions
```python
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure()
sns.regplot(x='explan_col',y='response_col', ci=None, data=df)

sns.regplot(x='explan_col',y='response_col', data=explanatory_data, color='red', marker='s')

plt.show()
```
Extrapolating
- *Extrapolating* means making predictions outside the range of observed data.
```python
explanatory_data = pd.DataFrame({'col': [int]}

predict_data = explanatory_data.assign(column_name = model.predict(explanatory_data))

print(predict_data)
```

## Working with model objects

.params attribute
```python
from statsmodels.formula.api import ols

model = ols('response_col ~ explan_col', data=df).fit()

print(model.params)
```
.fittedvalues attribute
- Fitted values: predictions on the original dataset
```python
print(model.fittedvalues)

# or

explanatory_data = df['col']
print(model.predict(explanatory_data))
```
.resid attribute
- Residuals: actual response values minus predicted response values
```python
print(model.resid)

# or

print(df['response_col'] - model.fittedvalues)
```
.summary()
```python
model.summary()
```

## Regression to the mean

The concept
- Response value = fitted value + residual
- "The stuff you explained" + "the stuff you couldn't explain"
- Residuals exist due to problems in the model *and* fundamental randomness
- Extreme cases are often due to randomness
- *Regression to the mean* means extreme cases don't persist over time

Pearson's father son dataset example
- 1078 father/son pairs
- Do tall fathers have tall sons?

| father_height_cm | son_height_cm |
| ---------- | ---------- |
| 165.2 | 151.8 |
| 160.7 | 160.6 |
| 165.0 | 160.9 |
| 167.0 | 159.5 |
| 155.3 | 163.3 |
| ... | ... |

Scatter plot
```python
fig = plt.figure()

sns.scatterplot(x='father_height_cm', y='son_height_cm', data=father_son)

plt.axline(xy1=(150,150), slope=1, linewidth=2, color='green')

plt.axis('equal')
plt.show()
```

<img width="538" alt="Screenshot 2024-01-24 at 5 13 17 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/bcbea631-4423-4476-a0a5-47b7c473fed3">

Adding a regression line
```python
fig = plt.figure()

sns.regplot(x='father_height_cm', y='son_height_cm', data=father_son, ci=None, line_kws={'color':'black'})

plt.axline(xy1=(150,150), slope=1, linewidth=2, color='green')

plt.axis('equal')
plt.show()
```

<img width="531" alt="Screenshot 2024-01-24 at 5 14 55 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/9b78a89c-3346-4f09-84d6-63abf388db5c">


Running a regression
```python
model_son_vs_father = ols('son_height_cm ~ father_height_cm', data=father_son).fit()
print(model_son_vs_father.params)
```
Making predictions
```python
really_tall_father = pd.DataFrame({'father_height_cm':[190]})
model_son_vs_father.predict(really_tall_father)

really_short_father = pd.DataFrame({'father_height_cm':[150]})
model_son_vs_father.predict(really_short_father)
```

## Transforming variables

**Perch dataset example**

```python
perch = fish[fish['species] == 'Perch']
print(perch.head())
```
It's not a linear relationship
```python
sns.regplot(x='length_cm', y='mass_g', data=perch, ci=None)

plt.show()
```
<img width="493" alt="Screenshot 2024-01-24 at 5 19 06 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/1d06ee0a-22de-4f04-bc6e-eee2232749b8">

Plotting mass vs length cubed
```python
perch['length_cm_cubed'] = perch['length_cm'] ** 3

sns.regplot(x='length_cm_cubed', y='mass_g', data=perch, ci=None)

plt.show()
```

<img width="492" alt="Screenshot 2024-01-24 at 5 20 23 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/09d740e6-5c1e-4765-8482-e276abd133a2">

Modeling mass vs length cubed
```python
perch['length_cm_cubed'] = perch['length_cm'] ** 3

model_perch = ols('mass_g ~ length_cm_cubed', data=perch).fit()
model_perch.params
```
Predicting mass vs. length cubed
```python
explanatory_data = pd.DataFrame({'length_cm_cubed': np.arange(10, 41, 5) ** 3, 'length_cm': np.arange(10, 41, 5)})

prediction_data = explanatory_data.assign(mass_g = model_perch.predict(explanatory_data)
print(prediction_data)
```
Plotting mass vs. length cubed
```python
fig = plt.figure()

sns.regplot(x='length_cm_cubed', y='mass_g', data=perch, ci=None)
sns.scatterplot(x='length_cm_cubed', y='mass_g', data=prediction_data, color='red', marker='s')

plt.show()
```
<img width="495" alt="Screenshot 2024-01-24 at 5 24 05 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/22bb83f4-48fd-46ea-ba36-5540408a0b57">

**Facebook advertising dataset**

How advertising works:
1. Pay Facebook to show ads
2. People see the ads ('impressions')
3. Some people who see it, click it

- 936 rows
- Each row represents 1 advert

| spent_usd | n_impressions | n_clicks
| ---------- | ---------- | ---------- |
| 1.43 | 7350 | 1 |
| 1.82 | 17861 | 2 |
| 1.25 | 4259 | 1 |
| 1.29 | 4133 | 1 |
| 4.77 | 15615 | 3 |
| ... | ... |

Plot is cramped
```python
sns.regplot(x='spent_usd', y='n_impressions', data=ad_conversion, ci=None)
```
<img width="487" alt="Screenshot 2024-01-24 at 5 27 45 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/3bab8b1c-7d19-496a-9445-06c12702c373">


Square root vs square root
```python
ad_conversion['sqrt_spent_usd'] = np.sqrt(ad_conversion['spent_usd'])
ad_conversion['sqrt_n_impressions'] = np.sqrt(ad_conversion['n_impressions'])

sns.regplot(x='sqrt_spent_usd', y='sqrt_n_impressions', data=ad_conversion, ci=None)

```
<img width="491" alt="Screenshot 2024-01-24 at 5 29 01 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/344ba04f-d4c2-4b10-8f84-c280d3c08887">

Modeling and predicting
```python
model_ad = ols('sqrt_n_impressions ~ sqrt_spent_usd', data=ad_conversion).fit()

explanatory_data = pd.DataFrame({'sqrt_spent_usd': np.sqrt(np.arange(0, 601, 100)), 'spent_usd': np.arange(0, 601, 100)})

prediction_data = explanatory_data.assign(sqrt_n_impressions = model_ad.predict(explanatory_data),
                                          n_impressions = model_ad.predict(explanatory_data ** 2)

print(prediction_data)
```
<img width="583" alt="Screenshot 2024-01-24 at 5 33 22 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/99417d57-7bb9-4a10-8161-15cd6d194ee8">
