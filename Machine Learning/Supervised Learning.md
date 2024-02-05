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


# How good is your model?

Classification metrics

- Measuring model performance with accuracy:
  - Fraction of correctly classified samples
  - Not always a useful metric
 
Class imbalance

- Classification for predicting fraudulent bank transactions
  - 99% of transactions are legitimate; 1% are fraudulent
- Could build a classifier that predicts NONE of the transactions are fraudulent
  - 99% accurate
  - But terrible at actually predicting fraudulent transactions
  - Fails at its original purpose
- Class imbalance: uneven frequency of classes
- Need a different way to assess performance

Assessing classification performance

<kbd><img width="769" alt="Screenshot 2024-02-05 at 11 33 14 AM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/9b9202d8-bca3-4fed-9959-46820a339c34">
</kbd>

Precision

<kbd><img width="620" alt="Screenshot 2024-02-05 at 11 33 53 AM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/fa677139-bf91-49eb-bf2d-373677a5230e">
</kbd>

- High precision = lower false positive rate
- High precision: not many legitimate transactions are predicted to be fraudulent

Recall

<kbd><img width="601" alt="Screenshot 2024-02-05 at 11 34 00 AM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/4d50d44b-8464-450c-ac25-b0ee7ad50178">
</kbd>

- High recall = lower false negative rate
- High recall: predicted most fraudulent transactions correctly

F-1 Score

<kbd>
</kbd>

Confusion matrix in scikit-learn
```python
from sklearn.metrics import classification_report, confusion_matrix
knn = KNeighborsClassifier(n_neighbors=7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(confusion_matrix(y_test, y_pred))
```
<kbd><img width="109" alt="Screenshot 2024-02-05 at 11 36 46 AM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/d245b36c-4146-4b62-b59b-68bcf4a713b8">
</kbd>

Classification report in scikit-learn
```python
print(classification_report(y_test, y_pred))
```
<kbd><img width="490" alt="Screenshot 2024-02-05 at 11 37 45 AM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/7d91ef68-a7e9-4ab5-9ca7-fee34de48f6b">
</kbd>

## Logistic regression and the ROC curve

Logistic regression for binary classification

- Logistic regression is used for classification problems
- Logistic regression outputs probabilities
- If the probability *p* > 0.5:
  - The data is labeled `1`
- If the probability *p* < 0.5:
  - The data is labeled `0`


Linear decision boundary 

<kbd>
</kbd>

Logistic regression in scikit-learn
```python
from skleanr.linear_model import LogisticRegression
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
```

Predicting probabilities
```python
y_pred_probs = logreg.predict_proba(X_test)[:,1]
print(y_pred_probs[0]) # 0.089
```

Probability thresholds

- By default, logistic regression threshold = 0.5
- Not specific to logistic regression
  - KNN classifiers also have thresholds
- What happens if we vary the threshold?

The ROC curve

<kbd><img width="547" alt="Screenshot 2024-02-05 at 11 42 27 AM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/2992e16c-77fa-436b-ad7a-ed11a4b3927e">
</kbd>

Plotting the ROC curve
```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()
```

ROC AUC in scikit-learn
```python
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred_probs)) # 0.67
```
<kbd><img width="463" alt="Screenshot 2024-02-05 at 11 44 41 AM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/7afbd81d-d4c8-4939-ad96-a9103ff3d24c">
</kbd>

## Hyperparameter tuning

- Ridge/lasso regression: choosing `alpha`
- KNN: choosing `n_neighbors`
- Hyperparameters: parameters we specify before fitting the model
  - Like `alpha` and `n_neighbors`

Choosing the correct hyperparameters

1. Try lots of different hyperparameter values
2. Fit all of them separately
3. See how well they perform
4. Choose the best performing values

- This is called **hyperparameter tuning**
- It is essential to use cross-validation to avoid overfitting to the test set
- We can still split the data and perform cross-validation on the trainng set
- We withhold the test set for final evaluation

Grid search cross-validation

<kbd>
</kbd>

GridSearchCV in scikit-learn
```python
from sklearn.model_selection import GridSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {'alpha': np.arange(0.0001, 1, 10), 'solver': ['sag', 'lsqr']}
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid, cv=kf)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)
```
<kbd><img width="319" alt="Screenshot 2024-02-05 at 11 49 58 AM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/b5b1965d-163a-4057-826d-d3a044eeabad">
</kbd>

Limitations and an alternative approach

- 3-fold cross-validation, 1 hyperparameter, 10 total values = 30 fits
- 10-fold cross-validation, 3 hyperparameter, 30 total values = 900 fits

RandomizedSearchCV
```python
from sklearn.model_selection import RandomizedSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {'alpha': np.arange(0.0001, 1, 10), 'solver': ['sag', 'lsqr']}
ridge = Ridge()
ridge_cv = RandomizedSearchCV(ridge, param_grid, cv=kf, n_iter=2)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)
```

Evaluating on the test set
```python
test_score = ridge_cv.score(X_test, y_test)
print(test_score) # 0.75
```

# Preprocessing data

scikit-learn requirements

- Numeric data
- No missing values

- With real-world data:
  - This is rarely the case
  - We will often need to preprocess our data first

Dealing with categorical features

- scikit-learn will not accept categorical features by default
- Need to conver categorical features into numeric values
- Convert to binary features called dummy variables
- 0: observation was NOT that category
- 1: observation was that category

Dummy variables

<kbd><img width="703" alt="Screenshot 2024-02-05 at 11 54 50 AM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/28b12044-8c09-4cb5-b312-5e179cef2209">
</kbd>

Dealing with categorical features

- scikit-learn: `OneHotEncoder()`
- pandas: `get_dummies()`

Music dataset

- `popularity`: target variable
- `genre`: categorica feature
```python
print(music.info())
```
<kbd><img width="685" alt="Screenshot 2024-02-05 at 11 55 53 AM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/5cd591b2-e5ca-4a44-8e49-b46f14aa0e4b">
</kbd>

EDA w/categorical feature

<kbd><img width="495" alt="Screenshot 2024-02-05 at 11 56 40 AM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/baa46d94-564c-4938-acb2-c4900c478462">
</kbd>

Encoding dummy variables
```python
import pandas as pd
music_df = pd.read_csv('music.csv')
music_dummies = pd.get_dummies(music_df['genre'], drop_first=True)
print(music_dummies.head())

music_dummies = pd.concat([music_df, music_dummies]), axis=1)
music_dummies = music_dummies.drop('genre', axis=1)

music_dummies = pd.get_dummies(music_df, drop_first=True)
print(music_dummies.columns)
```
<kbd><img width="658" alt="Screenshot 2024-02-05 at 11 59 38 AM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/d83d4959-7723-4a5b-9304-79a142b160d1">
</kbd>

<kbd><img width="692" alt="Screenshot 2024-02-05 at 12 00 12 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/99f75227-4eb5-4191-9624-9b21d61d33d3">
</kbd>


Linear regression with dummy variables
```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
X = music_dummies.drop('popularity', axis=1).values
y = music_dummies['popularity'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
linreg = LinearRegression()
linreg_cv = cross_val_score(linreg, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
print(np.sqrt(-linreg_cv))
```

## Handling missing data

- No value for a feature in a particular row
- This can occur because:
  - There may have been no observation
  - The data might be corrupt
- We need to deal with missing data

Music dataset
```python
print(music_df.isna().sum().sort_values())
```
<kbd><img width="181" alt="Screenshot 2024-02-05 at 12 04 45 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/076b9a58-1b17-4e9e-b163-6becd5bc28de">
</kbd>

Dropping missing data
```python
music_df = music_df.dropna(subset='genre', 'popularity', 'loudness', 'liveness', 'tempo'])
print(music_df.isna().sum().sort_values())
```
<kbd><img width="170" alt="Screenshot 2024-02-05 at 12 04 50 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/80c904e8-0787-42dd-bfb9-8053ce1d725a">
</kbd>

Imputing values

- Imputation - use subject-matter expertise to replace missing data with educated guesses
- Common to use the mean
- Can also use the median, or another value
- For categorical values, we typically use the most frequent value - the mode
- Must split our data first, to avoid *data leakage*
- Imputers are known as transformers

Imputation with scikit-learn
```python
from sklearn.impute import SimpleImputer
X_cat = music_df['genre'].values.reshape(-1, 1)
X_num = music_df.drop(['genre', 'popularity'], axis=1).values
y = music_df['popularity'].values
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y, test_size=0.2, random_state=12)
X_train_num, X_test_num, y_train_num, y_test_num = train_test_split(X_num, y, test_size=0.2, random_state=12)

imp_cat = SimpleImputer(strategy='most_frequent')
X_train_cat = imp_cat.fit_transform(X_train_cat)
X_test_cat = imp_cat.transform(X_test_cat)

imp_num = SimpleImputer()
X_train_num = imp_num.fit_transform(X_train_num)
X_test_num = imp_num.transform(X_test_num)
X_train = np.append(X_train_num, X_train_cat, axis=1)
X_test = np.append(X_test_num, X_test_cat, axis=1)
```

Imputing within a pipeline
```python
from sklearn.pipeline import Pipeline
music_df = music_df.dropna(subset=['genre', 'popularity', 'loudness', 'liveness', 'tempo'])
music_df['genre'] = np.where(music_df['genre'] == 'Rock', 1, 0)
X = music_df.drop('genre', axis=1).values
y = music_df['genre'].values
```
```python
steps = [('imputation', SimpleImputer()), ('logistic_regression', LogisticRegression())]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test) # 0.75
```

## Centering and scaling

Why scale our data?

- Many models use some form of distance to inform them
- Features on larger scales can disproportionately influence the model
- Example: KNN uses distance explicitly when making predictions
- We want features to be on a similar scale
- Normalizing or standardizing (scaling and centering)

```python
print(music_df['duration_ms', 'loudness', 'speechiness']].describe())
```
<kbd><img width="469" alt="Screenshot 2024-02-05 at 12 16 23 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/d3e1d1aa-dd91-4539-a844-053016f20119">
</kbd>

How to scale our data 

- Subtract the mean and divide by variance
  - All features are centered around zero and have a variance of one
  - This is called **standardization**
- Can also subtract the minimum and divide by the range
  - Minimum zero and maximum one
- Can also *normalize* so the data ranges from -1 to +1

Scaling in scikit-learn
```python
from sklearn.preprocessing import StandardScaler
X = music_df.drop('genre', axis=1).values
y = music_df['genre'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(np.mean(X), np.std(X))
print(np.mean(X_train_scaled)), np.std(X_train_scaled))
```

Scaling in a pipeline
```python
steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=6))]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
knn_scaled = pipeline.fit(X_train, y_train)
y_pred = knn_scaled.predict(X_test)
print(knn_scaled.score(X_test, y_test)) # 0.81
```

Comparing performance using unscaled data
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
knn_unscaled = KNeighborsClassifier(n_neighbors=6).fit(X_train, y_train)
print(knn_unscaled.score(X_test, y_test)) # 0.53
```

CV and scaling in a pipeline
```python
from sklearn.model_selection import GridSearchCV
steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
parameters = {'knn__n_neighbors': np.arange(1, 50)}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
print(cv.best_score_) # 0.81
print(cv.best_params_) # {'knn__n_neighbors': 12}
```

## Evaluating multiple models

Different models for different problems

- Size of the dataset
  - Fewer features = simpler model, faster training time
  - Some models require large amounts of data to perform well
- Interpretability
  - Some models are easier to explain, which can be important for stakeholders
  - Linear regression has high interpretability, as we can understand the coefficients
- Flexibility
  - May improve accuracy, by making fewer assumptions about data
  - KNN is a more flexible model, doesn't assume any linear relationships
 
It's all in the metrics

- Regression model performance:
  - RMSE
  - R-squared
- Classification model performance:
  - Accuracy
  - Confusion matrix
  - Precision, recall, F1-score
  - ROC AUC
- Train several models and evaluate performance out of the box

A note on scaling

- Models affected by scaling:
  - KNN
  - Linear Regression (plus Ridge, Lasso)
  - Logistic Regression
  - Artificual Neural Network
- Best to scale our data before evaluating models

Evaluating classification models
```python
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
X = music.drop('genre', axis=1).values
y = music['genre'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
```python
models =
results = []
for model in models.values():
  kf = KFold(n_splits=6, random_state=42, shuffle=True)
  cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
  results.append(cv_results)
plt.boxplot(results, labels=models.keys())
plt.show()
```
<kbd><img width="489" alt="Screenshot 2024-02-05 at 12 33 40 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/2d4c1c6e-5b8f-47a9-83af-3423367e60ab">
</kbd>

Test set performance
```python
for name, model in models.items():
  model.fit(X_train_scaled, y_train)
  test_score = model.score(X_test_scaled, y_test)
  print(f"{} Test Set Accuracy: {}".format(name, test_score))
```
<kbd><img width="417" alt="Screenshot 2024-02-05 at 12 34 53 PM" src="https://github.com/mattamx/DataScience_quick_guides/assets/107958646/c7683fa6-bd5f-4d32-949a-8b993188c720">
</kbd>
