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

The fish dataset: bream
```python
bream = fish[fish['species'] == 'Bream']
```
Plotting mass vs. length
```python
sns.regplot(x='length_cm', y='mass_g', data=bream, ci=None)
plt.show()
```
<kbd> <img width="495" alt="Screenshot 2024-01-24 at 9 19 20 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/28faac40-be75-4a79-80d6-7961b23a6219"> </kbd>

Running the model
```python
mdl_mass_vs_length = ols('mass_g ~ length_cm', data=bream).fit()
print(mdl_mass_vs_length.params)
```
<kbd> <img width="325" alt="Screenshot 2024-01-24 at 9 20 16 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/eb8f6ac6-6365-4b50-80d6-242735dab235"> </kbd>


Data on explanatory values to predict
```python
explanatory_data = pd.DataFrame({'length_cm': np.arange(20, 41})

print(mdl_mass_vs_length.predict(explanatory_data))
```
Predicting inside a DataFrame
```python
explanatory_data = pd.DataFrame({'length_cm': np.arange(20, 41})

prediction_data = explanatory_data.assign(mass_g = mdl_mass_vs_length.predict(explanatory_data))
```
<kbd> <img width="279" alt="Screenshot 2024-01-24 at 9 23 17 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/3dd937aa-7830-4b8e-bbef-63c262ab5b08"> </kbd>

Showing predictions
```python
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure()
sns.regplot(x='length_cm',y='mass_g', ci=None, data=bream)

sns.regplot(x='length_cm',y='mass_g', data=explanatory_data, color='red', marker='s')

plt.show()
```
<kbd> <img width="545" alt="Screenshot 2024-01-24 at 9 23 22 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/a633786c-e6d3-4134-a7ae-0cf95a12a057"></kbd>

Extrapolating
- *Extrapolating* means making predictions outside the range of observed data.
```python
explanatory_data = pd.DataFrame({'length_cm': [10]}

prediction_data = explanatory_data.assign(mass_g = mdl_mass_vs_length.predict(explanatory_data))

print(prediction_data)
```
<kbd> <img width="499" alt="Screenshot 2024-01-24 at 9 23 54 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/31683c45-392e-42d3-9359-d74de2fec79f"></kbd>

## Working with model objects

.params attribute
```python
from statsmodels.formula.api import ols

mdl_mass_vs_length = ols('mass_g ~ length_cm', data=bream).fit()

print(mdl_mass_vs_length.params)
```
.fittedvalues attribute
- Fitted values: predictions on the original dataset
```python
print(mdl_mass_vs_length.fittedvalues)

# or

explanatory_data = bream['length_cm']
print(mdl_mass_vs_length.predict(explanatory_data))
```
.resid attribute
- Residuals: actual response values minus predicted response values
```python
print(mdl_mass_vs_length.resid)

# or

print(bream['mass_g'] - mdl_mass_vs_length.fittedvalues)
```
<kbd> <img width="473" alt="Screenshot 2024-01-24 at 9 26 05 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/30538812-3de0-443b-bd35-e0458b0b4063"></kbd>


.summary()
```python
mdl_mass_vs_length.summary()
```
<kbd> <img width="594" alt="Screenshot 2024-01-24 at 9 26 15 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/ae761889-df0e-4045-bc40-7b74895c20e6"></kbd>



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

<kbd> <img width="538" alt="Screenshot 2024-01-24 at 5 13 17 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/bcbea631-4423-4476-a0a5-47b7c473fed3"> </kbd> 

Adding a regression line
```python
fig = plt.figure()

sns.regplot(x='father_height_cm', y='son_height_cm', data=father_son, ci=None, line_kws={'color':'black'})

plt.axline(xy1=(150,150), slope=1, linewidth=2, color='green')

plt.axis('equal')
plt.show()
```

<kbd> <img width="531" alt="Screenshot 2024-01-24 at 5 14 55 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/9b78a89c-3346-4f09-84d6-63abf388db5c"></kbd> 


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
<kbd><img width="493" alt="Screenshot 2024-01-24 at 5 19 06 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/1d06ee0a-22de-4f04-bc6e-eee2232749b8"></kbd> 

Plotting mass vs length cubed
```python
perch['length_cm_cubed'] = perch['length_cm'] ** 3

sns.regplot(x='length_cm_cubed', y='mass_g', data=perch, ci=None)

plt.show()
```

<kbd><img width="492" alt="Screenshot 2024-01-24 at 5 20 23 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/09d740e6-5c1e-4765-8482-e276abd133a2"></kbd> 

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
<kbd><img width="495" alt="Screenshot 2024-01-24 at 5 24 05 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/22bb83f4-48fd-46ea-ba36-5540408a0b57"></kbd> 

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
<kbd><img width="487" alt="Screenshot 2024-01-24 at 5 27 45 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/3bab8b1c-7d19-496a-9445-06c12702c373"></kbd> 


Square root vs square root
```python
ad_conversion['sqrt_spent_usd'] = np.sqrt(ad_conversion['spent_usd'])
ad_conversion['sqrt_n_impressions'] = np.sqrt(ad_conversion['n_impressions'])

sns.regplot(x='sqrt_spent_usd', y='sqrt_n_impressions', data=ad_conversion, ci=None)

```
<kbd><img width="491" alt="Screenshot 2024-01-24 at 5 29 01 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/344ba04f-d4c2-4b10-8f84-c280d3c08887"></kbd> 

Modeling and predicting
```python
model_ad = ols('sqrt_n_impressions ~ sqrt_spent_usd', data=ad_conversion).fit()

explanatory_data = pd.DataFrame({'sqrt_spent_usd': np.sqrt(np.arange(0, 601, 100)), 'spent_usd': np.arange(0, 601, 100)})

prediction_data = explanatory_data.assign(sqrt_n_impressions = model_ad.predict(explanatory_data),
                                          n_impressions = model_ad.predict(explanatory_data ** 2)

print(prediction_data)
```
<kbd><img width="583" alt="Screenshot 2024-01-24 at 5 33 22 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/99417d57-7bb9-4a10-8161-15cd6d194ee8"></kbd> 

# Quantifying model fit

**Bream and perch models**
<kbd><img width="1084" alt="Screenshot 2024-01-24 at 8 48 40 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/623c90c1-8d3a-4bb1-b3ea-131181e0ae89"></kbd> 


**Coefficient of determination**

Sometimes called "r-squared" or "R-squared"
> The proportion of the variance in the response variable that is predictable from the explanatory variable

- `1` means a perfect fit
- `0` means the worst possible fit

.summary()
```python
mdl_bream = ols('mass_g ~ length_cm', data=bream).fit()

print(mdl_bream.summary())
```
<kbd><img width="1025" alt="Screenshot 2024-01-24 at 8 49 19 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/f467b25a-8db7-47ef-bece-682b0ffb7650"></kbd> 

.rsquared attribute
```python
print(mdl_bream.rsquared)

# or

coeff_determination = bream['length_cm'].corr(bream['mass_g']) ** 2
print(coeff_determination)
```

Residual standard error (RSE)

<kbd><img width="506" alt="Screenshot 2024-01-24 at 8 51 39 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/9df992e6-281c-4942-8f72-fe7c04b97709"></kbd> 

- A "typical" difference between a prediction and observed response
- It has the same unit as the respone variable
- MSE = RSE²

.mse_resid attribute
```python
mse = mdl_bream.mse_resid
print('mse: ', mse)

rse = np.sqrt(mse)
print('rse: ', rse)
```
Calculating RSE: residuals squared
```python
residuals_sq = mdl_bream.resid ** 2

print('residuals sq: \n', residuals_sq)
```
Calculating RSE: sum of residuals squared
```python
residuals_sq = mdl_bream.resid ** 2

resid_sum_of_sq = sum(residuals_sq)

print('resid sum of sq: ', resid_sum_of_sq)
```
Calculating RSE: degrees of freedom

- *Degrees of freedom* equals the number of observations minus the number of model coefficients

```python
residuals_sq = mdl_bream.resid ** 2

resid_sum_of_sq = sum(residuals_sq)

deg_freedom = len(bream.index) - 2

print('deg freedom: ', deg_freedom)
```
Calculating RSE: square root of ratio
```python
residuals_sq = mdl_bream.resid ** 2

resid_sum_of_sq = sum(residuals_sq)

deg_freedom = len(bream.index) - 2

rse = np.sqrt(resid_sum_of_sq/deg_freedom)

print('rse: ', rse)
```

**Interpreting RSE**

`mdl_bream` has an RSE of `74`
> The difference between predicted bream masses and observed dream masses is typically about 74g

Root-mean-square-error (RMSE)
```python
residuals_sq = mdl_bream.resid ** 2

resid_sum_of_sq = sum(residuals_sq)

deg_freedom = len(bream.index) - 2

rse = np.sqrt(resid_sum_of_sq/deg_freedom)

print('rse: ', rse)
```
```python
residuals_sq = mdl_bream.resid ** 2

resid_sum_of_sq = sum(residuals_sq)

n_obs = len(bream.index)

rmse = np.sqrt(resid_sum_of_sq/n_obs)

print('rse: ', rmse)
```

# Visualizing model fit

**Residual properties of a good fit**

- Residuals are normally distributed
- The mean of the residuals is zero

Bream and perch example

Bream: the 'good' model
```python
mdl_bream = ols('mass_g ~ length_cm' data=bream).fit()
```
<kbd><img width="496" alt="Screenshot 2024-01-24 at 9 00 28 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/9796aa72-7596-4876-ba67-4557862afd61"></kbd>


Perch: the 'bad' model
```python
mdl_perch = ols('mass_g ~ length_cm' data=perch).fit()
```
<kbd><img width="508" alt="Screenshot 2024-01-24 at 9 00 44 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/5585a943-8fd2-48fe-8d06-31b839f08d65"></kbd>

Residuals vs. fitted
<kbd><img width="1059" alt="Screenshot 2024-01-24 at 9 01 35 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/c50a0b8f-48b8-4d0c-af82-8def8b193905"></kbd>

Q-Q plot
<kbd><img width="1059" alt="Screenshot 2024-01-24 at 9 02 02 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/7d310793-e61f-42f6-bf4c-3eec2ef1fec5"></kbd>


Scale-location plot
<kbd><img width="1113" alt="Screenshot 2024-01-24 at 9 02 22 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/037d8b8a-6485-4d68-902d-aa10568b40cb"></kbd>

## residplot()
```python
sns.residplot(x='length_cm', y='mass_g', data=bream, lowess=True)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
```
<kbd><img width="499" alt="Screenshot 2024-01-24 at 9 03 43 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/4b301725-b429-4dbc-8793-a57f5628dd04"></kbd>

## qqplot()
```python
from statsmodels.api import qqplot

qqplot(data=mdl_bream.resid, fit=True, line='45')
```
<kbd><img width="518" alt="Screenshot 2024-01-24 at 9 04 23 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/db767c04-31c9-4047-b062-e22a441242a9"></kbd>

## Scale-location plot
```python
model_norm_residuals_bream = mdl_bream.get_influence().resid_stundentized_internal
model_norm_residuals_abs_sqrt_bream = np.sqrt(np.abs(model_norm_residuals_bream))

sns.regplot(x=mdl_bream.fittedvalues, y=model_norm_residuals_abs_sqrt_bream, ci=None, lowess=True)

plt.xlabel('Fitted values')
plt.ylabel('Sqrt of abs val of stdized residuals')
```
<kbd><img width="461" alt="Screenshot 2024-01-24 at 9 04 48 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/97019304-b534-45d7-a91d-9dc21be95ee4"></kbd>

# Outliers, leverage and influence

Roach dataset
```python
roach = fish[fish['species'] == 'Roach']
```

Extreme explanatory values
```python
roach['extreme_l'] = ((roach['length_cm'] < 15) | (roach['length_cm'] > 26))

fig = plt.figure()
sns.regplot(x='length_cm', y='mass_g', data=roach, ci=None)
sns.scatterplot(x='length_cm', y='mass_g', hue='extreme_l', data=roach)
```
<kbd><img width="541" alt="Screenshot 2024-01-24 at 9 09 36 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/54c917d4-a9dc-4576-a60f-9c73e2b8423e"></kbd>

Response values away from the regression line
```python
roach['extreme_m'] = roach['mass_g'] < 1

fig = plt.figure()
sns.regplot(x='length_cm', y='mass_g', data=roach, ci=None)
sns.scatterplot(x='length_cm', y='mass_g', hue='extreme_m', data=roach)
```
<kbd><img width="486" alt="Screenshot 2024-01-24 at 9 10 18 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/98f3a5dc-63a8-4bed-ab43-e1368fb44ddf"></kbd>

## Leverage and influence

- *Leverage* is a measure of how extreme the explanatory values are
- *Influence* measures how much the model would change if you left the observation out of the dataset when modeling


.get_incluence() and .summary_frame()
```python
mdl_roach = ols('mass_g ~ length_cm', data=roach).fit()
summary_roach = mdl_roach.get_incluence().summary_frame()
roach['leverage'] = summary_roach['hat_diag']

print(roach.head())
```
<kbd><img width="519" alt="Screenshot 2024-01-24 at 9 14 30 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/f37be8f2-9cb5-4b04-9443-06e87753bf09"></kbd>

### Cook's distance

- *Cook's distance* is the most common measure of influence
  
```python
roach['cooks_dist'] = summary_roach['cooks_d']
print(roach.head())
```
<kbd><img width="675" alt="Screenshot 2024-01-24 at 9 14 36 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/b1deaaaf-5094-4acc-a833-6f1a0f43aed7"></kbd>

Most influential roaches
```python
print(roach.sort_values('cooks_dist', ascending = False))
```
<kbd><img width="878" alt="Screenshot 2024-01-24 at 9 15 41 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/1a79f281-82ce-4558-ac08-cae71e7dd86c"></kbd>

Removing the most influential roach
```python
roach_not_short = roach[roach['length_cm'] != 12.9]

sns.regplot(x='length_cm', y='mass_g', data=roach, ci=None, line_kws={'color':'green'})
sns.regplot(x='length_cm', y='mass_g', data=roach_not_short, ci=None, line_kws={'color':'red'})
```
<kbd><img width="513" alt="Screenshot 2024-01-24 at 9 15 46 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/82e84114-2a88-406a-9281-439527facdbe"></kbd>

# Logistic Regression

Bank churn dataset

| has_churned | time_since_first_purchase | time_since_last_purchase |
| --------- | --------- | --------- |
| 0 | 0.3993247 | -0.5158691 |
| 1 | -0.4297957 | 0.6780654 |
| 0 | 3.7383122 | 0.4082544 |
| 0 | 0.6032289 | -0.6990435 |
| ... | ... | ... |
| *response* | *length of relationship* | *recency of activity* |

Churn vs. recency
```python
mdl_churn_vs_recency_lm = ols('has_churned ~ time_since_last_purchase', data=churn).fit()

print(mdl_churn_vs_recency_lm)

intercept, slope = mdl_churn_vs_recency_lm.params
```
Visualizing the linear model
```python
sns.scatterplot(x='time_since_last_purchase', y='has_churned', data=churn)

plt.axline(xy1=(0, intercept), slope=slope)

plt.show()
```
<kbd><img width="379" alt="Screenshot 2024-01-25 at 6 01 18 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/75e3d9c7-7e01-41fa-bb0a-832165f305d7"></kbd>

Zooming out
```python
sns.scatterplot(x='time_since_last_purchase', y='has_churned', data=churn)

plt.axline(xy1=(0, intercept), slope=slope)

plt.xlim(-10,10)
plt.ylim(-0.2, 1.2)
plt.show()
```
<kbd><img width="381" alt="Screenshot 2024-01-25 at 6 01 57 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/10157409-0e40-408a-95eb-1846569e3c3b"> </kbd>

**What is logistic regression?**

- Another type of generalized linear model
- Used whent he response variable is logical
- The responses follow logistic (S-shaped) curve

Logistic regression using logit()
```python
from statsmodels.formula.api import logit

mdl_churn_vs_recency_logit = logit('has_churned ~ time_since_last_purchase', data=churn).fit()

print(mdl_churn_vs_recency_logit.params)
```
Visualizing the logistic model
```python
sns.regplot(x='time_since_last_purchase', y='has_churned', data=churn, ci=None, logistic=True)

plt.axline(xy1=(0, intercept), slope=slope, color='black')

plt.show()
```
<kbd><img width="396" alt="Screenshot 2024-01-25 at 6 05 22 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/c6178fec-d887-4ffe-8bf8-a14540cdddef"></kbd>

Zooming Out

<kbd><img width="486" alt="Screenshot 2024-01-25 at 6 05 29 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/3f8d612a-302a-4889-8fde-1ed8a19c2ad7"></kbd>

## Predictions and odds ratios

The regplot() predictions
```python
sns.regplot(x='time_since_last_purchase', y='has_churned', data=churn, ci=None, logistic=True)

plt.show()
```
<kbd><img width="396" alt="Screenshot 2024-01-25 at 6 09 31 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/f2f340e6-c010-43b2-9eca-bb0a9b7a1a7e"></kbd>

Making predictions
```python
mdl_recency= logit('has_churned ~ time_since_last_purchase', data=churn).fit()

explanatory_data = pd.DataFrame({'time_since_last_purchase': np.arange(-1, 6.25, 0.25))

prediction_data = explanatory_data.assign(has_churned=mdl_recency.predict(explanatory_data))
```
Adding point predictions
```python
sns.regplot(x='time_since_last_purchase', y='has_churned', data=churn, ci=None, logistic=True)

sns.scatterplot(x='time_since_last_purchase', y='has_churned', data=prediction_data, color='red')

plt.show()
```
<kbd><img width="401" alt="Screenshot 2024-01-25 at 6 10 04 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/23dcd72f-5b53-4a2b-a41f-746394b1dff9"> </kbd>

Getting the most likely outcome
```python
prediction_data = explanatory_data.assign(has_churned=mdl_recency.predict(explanatory_data))

prediction_data['most_likely_outcome'] = np.round(prediction_data['has_churned'])
```
Visualizing most likely outcome
```python
sns.regplot(x='time_since_last_purchase', y='has_churned', data=churn, ci=None, logistic=True)

sns.scatterplot(x='time_since_last_purchase', y='most_likely_outcome', data=prediction_data, color='red')

plt.show()
```
<kbd><img width="390" alt="Screenshot 2024-01-25 at 6 11 32 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/9a07428c-629d-4010-a09a-69f873f79953"> </kbd>

**Odds ratios**

*Odds ratio* is the probability of something happening divided by the probability that it doesn't.

<kbd><img width="317" alt="Screenshot 2024-01-25 at 6 12 53 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/a3d0d6ee-5ddd-4a19-bba6-38cb10a0c3fa"></kbd>

<kbd><img width="400" alt="Screenshot 2024-01-25 at 6 13 22 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/5faa2058-5923-4bbb-8936-53f34a822ffe"></kbd>

Calculating odds ratio
```python
prediction_data['odds_ratio'] = prediction_data['has_churned'] / (1 - prediction_data['has_churned'])
```
Visualizing odds ratio
```python
sns.lineplot(x='time_since_last_purchase', y='odds_ratio', data=prediction_data)

plt.axhline(y=1, linestyle='dotted')

plt.show()
```
<kbd><img width="388" alt="Screenshot 2024-01-25 at 6 15 18 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/055fcea7-b1a3-42f9-9bde-61470dfa4138"></kbd>

Visualizing log odds ratio
```python
sns.lineplot(x='time_since_last_purchase', y='odds_ratio', data=prediction_data)

plt.axhline(y=1, linestyle='dotted')
plt.yscale('log')

plt.show()
```
<kbd><img width="403" alt="Screenshot 2024-01-25 at 6 15 47 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/b72756ef-657c-473e-90cd-0ea0344a5859"></kbd>

Calculating log odds ratio
```python
prediction_data['log_odds_ratio'] = np.log(prediction_data['odds_ratio'])
```

**All predictions together**

| time_since_last_purchase | has_churned | most_likely_outcome | odds_ratio | log_odds_ratio |
| ----- | ----- | ----- | ----- | ----- |
| 0 | 0.491 | 0 | 0.966 | -0.035 |
| 2 | 0.623 | 1 | 1.654 | 0.503 |
| 4 | 0.739 | 1 | 2.834 | 1.042 |
| 6 | 0.829 | 1 | 4.856 | 1.580 |
| ... | ... | ... | ... | ... |



**Comparing scales**

| Scale | Are values easy to interpret? | Are changes easy to interpret? | Is precise? |
| ----- | ----- | ----- | ----- | 
| Probability | ✔ | ✘ | ✔ |
| Most likely outcome | ✔✔ | ✔ | ✘ | 
| Odds ratio | ✔ | ✘ | ✔ | 
| Log odds ratio | ✘ | ✔ | ✔ | 


# Quantifying logistic regression fit

**The four outcomes**

|  | predicted false | predicted true |
| ----- | ----- | ----- |
| **actual false** | correct | false positive |
| **actual true** | false negative | correct |


Confusion matrix: count of outcomes
```python
actual_response = churn['has_churned']

predicted_response = np.round(mdl_recency.predict())

outcomes = pd.DataFrame({'actual_response': actual_response,'predicted_response': predicted_response})

print(outcomes.value_counts(sort=False))
```
<kbd> <img width="347" alt="Screenshot 2024-01-25 at 6 26 51 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/4a005f4a-7c34-416e-9811-99e9d0c8965b"></kbd>

Visualizing the confusion matrix
```python
conf_matrix = mdl_recency.pred_table()

print(conf_matrix)
```
<kbd><img width="223" alt="Screenshot 2024-01-25 at 6 28 15 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/4acfb042-3680-4df2-8c9e-11032f46c0fc"> </kbd>

| true negative | false positive |
| ----- | ------|
| **false negative** | **true positive** |

```python
from statsmodels.graphics.mosaicplot import mosaic

mosaic(conf_matrix)
```
<kbd><img width="392" alt="Screenshot 2024-01-25 at 6 27 49 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/2585af46-a203-4c84-803f-a196206b4261"></kbd>

**Accuracy**

*Accuracy* is the proportion of correct predictions.

<kbd><img width="349" alt="Screenshot 2024-01-25 at 6 30 46 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/e1ee2c81-0997-4367-ac18-8b00e233f0df"></kbd>

```python
TN = conf_matrix[0,0]
TP = conf_matrix[1,1]
FN = conf_matrix[1,0]
FP = conf_matrix[0,1]
```
```python
acc = (TN + TP) / (TN + TP + FN + FP)
```

**Sensitivity**

*Sensitivity* is the proportion of true positives.

<kbd> <img width="248" alt="Screenshot 2024-01-25 at 6 33 04 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/3bed7ea7-0162-49f9-b781-ca54a502db66"></kbd>
```python
sense = TP / (FN + TP)
```

**Specificity**

*Specificity* is the proportion of true negatives.

<kbd> <img width="239" alt="Screenshot 2024-01-25 at 6 33 55 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/4ba41857-a9b7-45d9-a429-4eb4cccbf9c8"> </kbd>
```python
spec = TN / (TN + FP)
```
