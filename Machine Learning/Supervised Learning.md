# Machine learning with scikit-learn

- Machine learning is the process whereby:
  - Computers are given the ability to learn to make decisions from data wihout being explicitly programmed.
 
**Supervised learning**

- The predicted values are known
- Aim: predict the target values of unseen data, given the features

Types of supervised learning:

- Classification: target variable consists of categories
- Regression: target variable is continuous

Naming conventions

- Feature = predictor variable = independent variable
- Target variable = dependent variable = response variable

Requirements

- No missing values
- Data in numeric format
- Data stored in pandas DataFrame or NumPy array

*Perform Exploratory Data Analysis (EDA) first*

scikit-learn syntax
```python
from sklearn.module import Model
model = Model()
model.fit(X, y)
predictions = model.predict(X_new)
print(predictions)
```

## The classification challenge

Classifying labels of unseen data

1. Build a model
2. Model learns from the labeled data we pass to it
3. Pass unlabeled data to the model as input
4. Model predicts the labels of the unseen data

*Labeled data = training data*

k-Nearest Neighbors

- Predict the label of a data point by:
  - Looking at the `k` closest labeled data points
  - Taking a mojority vote
 
<kbd><img width="451" alt="Screenshot 2024-02-04 at 6 45 51 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/c170534e-457a-4385-8bcf-2304006277fd">
</kbd>

Using scikit-learn to fit a classifier
```python
from sklearn.neighbors import KNeighborsClassifier
X = churn_df[['total_day_charge', 'total_eve_charge']].values
y = churn_df['churn'].values
print(X.shape, y.shape)
```
```python
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X, y)
```

Predicting on unlabeled data
```python
X_new = np.array([[56.8, 17.5],
                  [24.4, 24.1],
                  [50.1, 10.9]])

print(X_new.shape)
```
```python
predictions = knn.predict(X_new)
print('Predictions: {}'.format(predictions))
```

## Measuring model performance

- In classification, accuracy is a commonly used metric
- Accuracy:
  
<kbd><img width="465" alt="Screenshot 2024-02-04 at 6 50 04 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/08e667ea-1b6b-4ba9-ad95-302457402cc1">
</kbd>

- How do we measure accuracy?
- Could compute accuracy on the data used to fit the classifier
- NOT indicative of ability of generalize

Computing accuracy

<kbd><img width="609" alt="Screenshot 2024-02-04 at 6 51 37 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/3d57bc92-a6dd-4c1f-9867-9bfe005fa57e">
</kbd>

Train/test split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,
                                    y,
                                    test_size=0.3,
                                    random_state=12,
                                    stratify=y)
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test)) # 0.88
```

Model complexity

- Larger k = less complex model = can cause underfitting
- Smaller k = more complex model = can lead to overfitting

<kbd><img width="644" alt="Screenshot 2024-02-04 at 6 54 12 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/7941c42a-ef4a-4b1c-9383-3ced72fb0042">
</kbd>

Model complexity and over/underfitting
```python
train_accuracies = {}
test_accuracies = {}
neighbors = np.arange(1,26)
for neighbor in neighbors:
  knn = KNeighborsClassifier(n_neighbors=neighbor)
  knn.fit(X_train, y_train)
  train_accuracies[neighbor] = knn.score(X_train, y_train)
  test_accuracies[neighbor] = knn.score(X_test, y_test)
```

Plotting our results
```python
plt.figure(figsize=(8,6))
plt.title('')
plt.plot(neighbors, train_accuracies.values(), label='Training Accuracy')
plt.plot(neighbors, test_accuracies.values(), label='Testing Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
```

Model complexity curve

<kbd><img width="496" alt="Screenshot 2024-02-04 at 6 57 30 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/cf7a3485-5f22-4422-8c46-6b81fe134c49">
</kbd>

# Introduction to regression

Predicting blood glucose levels
```python
import pandas as pd
diabetes_df = pd.read_csv('diabetes.csv')
print(diabetes_df.head())
```
<kbd><img width="711" alt="Screenshot 2024-02-04 at 6 59 29 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/5431f4f0-18df-4c68-91cf-d480660331b2">
</kbd>

Creating feature and target arrays
```python
X = diabetes_df.drop('glucose', axis=1).values
y = diabetes_df.gluclose.values
print(type(X), type(y))
```

Making predictions from a single feature
```python
X_bmi = X[:, 3]
print(y.shape, X_bmi.shape)
```
```python
X_bmi = X_bmi.reshape(-1,1)
print(X_bmi.shape)
```

Plotting glucose vs. body mass index
```python
import matplotlib.pyplot as plt
plt.scatter(X_bmi, y)
plt.ylabel('Blood Glucose (md/dl)')
plt.xlabel('Body Mass Index')
plt.show()
```
<kbd><img width="525" alt="Screenshot 2024-02-04 at 7 02 21 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/c720d0f6-2f3e-4638-b5dd-95b84c669181">
</kbd>

Fitting a regression model
```python
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_bmi, y)
predictions = reg.predict(X_bmi)
plt.scatter(X_bmi, predictions)
plt.ylabel('Blood Glucose (md/dl)')
plt.xlabel('Body Mass Index')
plt.show()
```
<kbd><img width="510" alt="Screenshot 2024-02-04 at 7 03 49 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/21e11fd6-b520-419e-9811-2c7fb83b9af8">
</kbd>

## The basics of linear regression

Regression mechanics

- *y = ax + b*
  - Simple linear regressio uses one feature
    - *y* = target
    - *x* = single feature
    - *a,b* = parameters/coefficients of the model - slope, intercept

- How do we choose *a* and *b*?
  - Define an error function for any given line
  - Choose the line that minimizes the error function
- Error function = loss function = cost function

The loss function

<kbd><img width="485" alt="Screenshot 2024-02-04 at 7 06 15 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/76c290f5-a5f0-41cd-a2b9-dc505a47c333">
</kbd>

Ordinary Least Squares

<kbd><img width="380" alt="Screenshot 2024-02-04 at 7 08 00 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/a9c121bd-4ea3-4d14-bf19-352f95208b41">
</kbd>
<kbd><img width="371" alt="Screenshot 2024-02-04 at 7 08 11 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/855b10f2-37ac-4904-9079-2609dd709d9a">
</kbd>

Linear regression in higher dimensions 

<kbd><img width="197" alt="Screenshot 2024-02-04 at 7 08 56 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/e8de1d80-ce47-49b7-b00e-a55fe887f9de">
</kbd>

- To fit a linear regression model here:
  - Need to specify 3 variables: *a*<sup>1</sup>, *a*<sup>2</sup>, *b*
- In higher dimensions:
  - Known as multiple regression
  - Must specificy coefficients for each feature and variable *b*

<kbd><img width="372" alt="Screenshot 2024-02-04 at 7 09 01 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/19472dc5-434c-4b9f-b424-e39f608468ce">
</kbd>

- scikit-learn works exactly the same way:
  - Pass two arrays: features and target
 
Linear regression using all features
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)
```

R-squared

- R2: quantifies the variance in target values
  - Values range from 0 to 1

- High R2:

<kbd><img width="233" alt="Screenshot 2024-02-04 at 7 12 40 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/d26c140d-c97e-46b2-9927-438b1aff04f0">

</kbd>

- Low R2:

<kbd><img width="238" alt="Screenshot 2024-02-04 at 7 12 44 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/1c6870b1-8625-4731-81a7-63afbc5cce31">
</kbd>

R-squared in scikit-learn
```python
reg_all.score(X_test, y_test) # 0.35
```

Mean squared error and root mean squared error

<kbd><img width="228" alt="Screenshot 2024-02-04 at 7 14 22 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/7c848bca-3aea-4432-9d9e-3683ba2496af">
</kbd>

- *MSE* is measured in target units, squared

<kbd><img width="186" alt="Screenshot 2024-02-04 at 7 14 26 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/e7e8158d-df36-49e8-96d0-ffe2c30f9be7">
</kbd>

- Measure *RMSE* in the same units as the target variable

RMSE in scikit-learn
```python
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred, squared=False) # 24.02
```

## Cross-validation

- Model performance is dependent on the way we split up the data
- Not representative of the model's ability to generalize the unseen data
- Solution: Cross-validation

Cross-validation basics

<kbd><img width="744" alt="Screenshot 2024-02-04 at 7 19 32 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/c47f68f1-1ddd-4bcb-8168-adc805d99428">
</kbd>

Cross-validation and mode performance

- 5 folds: 5-fold CV
- 10 folds: 10-fold CV
- k folds: k-fold CV
- More folds: more computationally expensive

Cross-validation in scikit-learn
```python
from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits=6, shuffle=True, random_state=42)
reg = LinearRegression()
cv_results = cross_val_score(reg, X, y, cv=kf)
```
Evaluating cross-validation performance
```python
print(cv_results)

print(np.mean(cv_results), np.std(cv_results))

print(np.quantile(cv_results, [0.025, 0.975]))
```


## Regularized regression

Why regularize?

- Recall: Linear regression minimizes the loss function
- It chooses a coefficient, *a*, for each feature variable, plus *b*
- Large coefficients can lead to overfitting
- Regularization: penalize large coefficients

Ridge Regression

- Loss function = OLS loss function +
  
<kbd><img width="115" alt="Screenshot 2024-02-04 at 7 23 13 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/dedaf947-4994-45a0-826c-26bfe47be1d5">
</kbd>

- Ridge penalizes large positive or negative coefficients
- *α*: parameter we need to choose
- Picking *α* is similar to picking `k` in KNN
- Hyperparameter: variable used to optimize model parameters
- *α* controls model complexity
  - *α* = 0 = OLS(can lead to overfitting)
  - Very high *α*: can lead to underfitting

Ridge regression in scikit-learn
```python
from sklearn.linear_model import Ridge
scores = []
for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
  ridge = Ridge(alpha=alpha)
  ridge.fit(X_train, y_train)
  y_pred = ridge.predict(X_test)
  scores.append(ridge.score(X_test, y_test))
print(scores)
```

Lasso regression

- Loss function = OLS loss function +

<kbd><img width="132" alt="Screenshot 2024-02-04 at 7 26 26 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/332e9f41-9e6d-42dd-98c8-6f9542532653">
</kbd>


Lasso regression in scikit-learn
```python
from sklearn.linear_model import Lasso
scores = []
for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
  lasso = Lasso(alpha=alpha)
  lasso.fit(X_train, y_train)
  lasso_pred = lasso.predict(X_test)
  scores.append(lasso.score(X_test, y_test))
print(scores)
```

Lasso for feature selection in scikit-learn
```python
from sklearn.linear_model import Lasso
X = diabetes_df.drop('glucose', axis=1).values
y = diabetes_df.glucose.values
names = diabetes_df.drop('glucose', axis=1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_
plt.bar(names, lasso_coef)
plt.xticks(rotation=45)
plt.show()
```
<kbd><img width="473" alt="Screenshot 2024-02-04 at 7 29 06 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/5ba61b33-1bd8-48e6-a0c1-38cb234e76fa">
</kbd>

```python

```

```python

```

```python

```
