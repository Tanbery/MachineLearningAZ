# Learning Machine Learning from A to Z

- [ ] Import libraries
- [ ] Load data
- [ ] Exploratory and data analysis
- [ ] Cleaning
- [ ] Modeling

## Import Libraries

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as kr
print(f"Tensor version:{tf.__version__}")
print(f"Keras  version:{kr.__version__}")
```

## Loading data

```python
dataset = pd.read_csv('Data.tsv', delimiter = '\t', quoting = 3) #tsv: tab seperated values
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

dataset = pd.read_csv('Salary_Data.csv')
#dropping unwanted columns
X = dataset.drop(["Salary"],axis=1)
y = dataset["Salary"]
train_eda = train.drop(["Name","Cabin","Ticket"],axis=1)
```

## Data Exploraty and Analysis

```python
train_eda.head()
train_eda.shape
train_eda.describe()
train_eda.columns
train_eda.info()
train_eda.isnull().sum()

#Draw a heatmap
plt.figure(figsize=(10,8))
sns.heatmap(data = train_eda.corr(), annot = True, cmap = 'Blues')
plt.show()
#Draw a BoxPlot
sns.boxplot(x=train_eda["Fare"])

# Set the width and height of the figure
plt.figure(figsize=(16,6))
sns.lineplot(data=train_eda)

```

### Take care of missing data

```python
from sklearn.impute import SimpleImputer
imputer= SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

features = ["Sex", "SibSp", "Parch", "Embarked", "Fare","Pclass"]
x = df_train[features]
y = df_train["Survived"]
z = df_test[features]
x = pd.get_dummies(x).fillna(0)
z = pd.get_dummies(z).fillna(0)

#dropping empty rows
train_fin = train_eda.dropna()
train_fin.isnull().sum()
```

### Encode Categorical data(from string to vektor)

```python
# Creating Dummy Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# create vertors instead of LabelEncoder for X[0] France=1,0,0 Germany=0,1,0 Spain=0,0,1
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])], remainder="passthrough")
X = np.array( ct.fit_transform(X))

# Replace text values with numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

x = df_train[features]
y = df_train["Survived"]
z = df_test[features]
x = pd.get_dummies(x).fillna(0)
z = pd.get_dummies(z).fillna(0)
```

## Split data

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

## Scaling the features

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
```

## Modeling

### Regression

#### LinearRegression

```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
pred = regressor.predict(X_test)
```

#### Polynomial Regression

```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=10)

#### Transform features as poly mode as feature.
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)
```

### SVR Regression

```python
# Standard Scaler [-3,+3]
from sklearn.preprocessing import StandardScaler
# Create 2 different StandardScaler.
# Because each of them has different mean/max/min
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
```

### TREE Regression

```python
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)
```

### Random Forest Regression

```python
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)
```

### xgboost regression and classification

```python
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
```

### PCA => Principal Component Analysis

```python
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
```

### Kernel PCA

```python
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)
```

### LDA ==> Lianer Discriminat Analysis

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
```

### Metrics - Evaluating the Model Performance

```python
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

#k-Fold Cross Validation ==> Model Selection
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

#GridSearchCV  ==> Model Selection
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,param_grid = parameters,scoring = 'accuracy',cv = 10,n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
```

## Plot Datas

### Training dataset visualization

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

### Test dataset visualization

plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
