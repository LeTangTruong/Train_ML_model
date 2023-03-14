# House Price Prediction Project
### 1. problem Definition:
- Goal: predict the sales price for each house
### 2. Feature Selection
- Choose features to train ML Model
- Need to use `Feature Engineering` to identify Feature needed
### 3. Spliting the datasets
#### 3.1 dataset &#8594 X,y
    - `data`: dataset
    - `x`: `data[features]`
    - `y`:target variable `SalePrice`
#### 3.2 X,y &#8594 X_train, y_train, X_valid, y_valid
### 4. Training ML Model
    

#import libraries
import pandas as pd
import numpy as np

data=pd.read_csv("./train.csv")

data.head()

data = pd.read_csv("./train.csv",index_col="Id")

data.head()

data.columns

## 2. feature selection

features=["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

## 3. Spliting dataset in X and y

X = data[features]
y = data["SalePrice"]

X.head()

y.head()

->X_train, y_train, X_valid, y_valid


from sklearn.model_selection import train_test_split
X_train,X_valid, y_train, y_valid=train_test_split(X,y,train_size=0.8,test_size=20, random_state=0)

X_train


### 4. Training ML Model

from sklearn.tree import DecisionTreeRegressor
dt_model=DecisionTreeRegressor(random_state=1)

#fit training data into model
dt_model.fit(X_train, y_train)

y_preds=dt_model.predict(X_valid.head())

y_preds

pd.DataFrame({'y':y_valid.head(),'y_preds':y_preds})

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
rf_model=RandomForestRegressor(random_state=1)
rf_model.fit(X_train, y_train)

rf_val_preds = rf_model.predict(x_valid)

rf_val_preds[:5]

 ### Predict with a new input

x_valid.head()

rf_model.predict([[6969,2021,1000,800,4,5,8]])

