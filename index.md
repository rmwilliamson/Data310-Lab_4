# Lab 4

# Question 6

### Write your own Python code to import the Boston housing data set (from the sklearn library) and scale the data (not the target) by z-scores. If we use all the features with the Linear Regression to predict the target variable then the root mean squared error (RMSE) is...

```markdown
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as MSE

data = load_boston()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
y = data.target

X = df.values
scale = StandardScaler()
xscaled = scale.fit_transform(X)

model = LinearRegression()
model.fit(xscaled,y)

y_pred = model.predict(xscaled)
rmse = np.sqrt(MSE(y,y_pred))
```

## RMSE = 4.6792


# Question 7

### On the Boston housing data set if we consider the Lasso model with 'alpha=0.03' then the 10-fold cross-validated prediction error is:

```markdown
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline

kf = KFold(n_splits=10, shuffle=True, random_state=1234)
model = Lasso(alpha=0.03)
scale = StandardScaler()
pipe = Pipeline([('Scale', scale),('Regressor', model)])

def DoKFold(X,y,model):
  PE = [] #prediction error
  for idxtrain, idxtest in kf.split(X):
    Xtrain = X[idxtrain,:]
    Xtest = X[idxtest,:]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    pipe.fit(Xtrain,ytrain)
    yhat = pipe.predict(Xtest)
    PE.append(MSE(ytest,yhat))
  return (np.mean(PE))
  
 DoKFold(X,y,model)
 
 ```

 ## Cross-validated prediction error = 24.2606
 
 
# Question 8

### On the Boston housing data set if we consider the Elastic Net model with 'alpha=0.05' and 'l1_ratio=0.9' then the 10-fold cross-validated prediction error is:

```markdown
from sklearn.linear_model import ElasticNet

kf_en = KFold(n_splits=10, shuffle=True, random_state=1234)
model = ElasticNet(alpha=0.05,l1_ratio=0.9)
scale = StandardScaler()
pipe = Pipeline([('Scale', scale),('Regressor', model)])

def DoKFold(X,y,model):
  PE = [] #prediction error
  for idxtrain, idxtest in kf_en.split(X):
    Xtrain = X[idxtrain,:]
    Xtest = X[idxtest,:]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    pipe.fit(Xtrain,ytrain)
    yhat = pipe.predict(Xtest)
    PE.append(MSE(ytest,yhat))
  return (np.mean(PE))
  
DoKFold(X,y,model)
```

## Cross-validated prediction error = 24.3104
