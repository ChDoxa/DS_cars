#******************************DATA*******************************************
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

car= 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
from urllib.request import urlretrieve
urlretrieve(car)
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight','Acceleration', 'Year', 'Origin', 'Model']
df = pd.read_csv(car, delim_whitespace=True, names=column_names)

df.isnull().values.sum()
pd.set_option('precision', 2)
# display stats of the features
df.describe()

for i in column_names:
    print(df[df[i] =='?'].index)
    
for i in column_names:
    print(df[df[i] =='NaN'].index)
    
for i in column_names:
    print(df[df[i] ==' '].index)
    
horsepower_missing_ind = df[df.Horsepower=='?'].index

horsepower_missing_ind = df[df.Horsepower=='?'].index
df.loc[horsepower_missing_ind]
df.dtypes#Horsepower: object
df.loc[horsepower_missing_ind, 'Horsepower'] = float('nan')
df.Horsepower = df.Horsepower.apply(pd.to_numeric)
df.loc[horsepower_missing_ind, 'Horsepower'] = int(df.Horsepower.mean())

pd.set_option('precision', 2)
# display stats of the features
df.describe()

#Determine the target column
df.info()#all columns
df.describe().columns#only num columns
df.describe()
df.head()

# Seperate Numeric & Categorical Variables 
df_num = df[['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight','Acceleration']]
df_cat =df[['Model', 'Year', 'Origin']]

# Hists for numeric values  
for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show()

#***Correlation of numeric variables  
print(df_num.corr())
sns.heatmap(df_num.corr())
plt.title("Correlation of numeric variables")
plt.show()

pd.pivot_table(df, index = 'MPG', values = ['Cylinders', 'Displacement', 'Horsepower', 'Weight',
       'Acceleration'])

#***Relationship between the categorical variables
for i in df_cat.columns[1:3]:
    sns.barplot(df_cat[i].value_counts().index,df_cat[i].value_counts()).set_title(i)
    if i!= ["Model"]:
        plt.xticks(fontsize=10)
        plt.show()

#**************Simple Linear Regression******************

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from math import sqrt
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import scipy as sp

cars_df=pd.read_excel('cars.xls')
cars_df.head()
cars_df.info()
y=cars_df.MPG
X=cars_df.Horsepower
X_train,X_test,y_train,y_test=train_test_split(pd.DataFrame(X),y,test_size=0.3,random_state=42)
regressor= LinearRegression()
regressor.fit(X_train,y_train)
regressor.score(X_train,y_train)#
y_prediction=regressor.predict(X_test)
print(y_prediction)
y_prediction=regressor.predict(X_test)
RMSE=sqrt(mean_squared_error(y_true=y_test,y_pred=y_prediction))
print(RMSE)
RMSE=round(RMSE,2)
print(RMSE)
plt.plot(y_test,y_prediction,"yo")
plt.xlabel("Y TEST DATA")
plt.ylabel("Y PREDICT")
plt.title( f"Y Data - Y Prediction, RMSE={RMSE} "
          "\n Simple Linear Regression")
plt.show()


#**************Multiple Linear Regression******************
y=cars_df.MPG
X=cars_df[['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Year']]
X_train,X_test,y_train,y_test=train_test_split(X,y)
scaler=MinMaxScaler()
X_train_sc=pd.DataFrame(scaler.fit_transform(X_train),columns=X_train.columns)
X_test_sc=pd.DataFrame(scaler.transform(X_test),columns=X_test.columns)
regressor= LinearRegression()
regressor.fit(X_train_sc, y_train)
y_prediction=regressor.predict(X_test_sc)
RMSE=sqrt(mean_squared_error(y_true=y_test,y_pred=y_prediction))
print(RMSE)
RMSE=round(RMSE,2)
print(RMSE)
plt.plot(y_test,y_prediction,"yo")
plt.xlabel("Y TEST DATA")
plt.ylabel("Y PREDICT")
plt.title( f"Y Data - Y Prediction, RMSE={RMSE} "
          "\n Multiple Linear Regression")
plt.show()

#************Random Forests***********
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,mean_squared_error
from math import sqrt 
from sklearn import tree
cars_df=pd.read_excel('cars.xls')
y=cars_df.MPG
X=cars_df[['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Year','Origin']]
X=pd.get_dummies(X,drop_first=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
rf= RandomForestRegressor(n_estimators=100,max_depth=42,n_jobs=-1)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
RMSE=round((sqrt(mean_squared_error(y_true=y_test, y_pred=y_pred))),2)
print(RMSE)
plt.plot(y_test,y_pred,"yo")
plt.xlabel("Y TEST DATA")
plt.ylabel("Y PREDICT")
plt.title( f"Y Data - Y Prediction, RMSE={RMSE} "
          "\n Random Forests")
plt.show()

#**************Support Vector Machines (SVM)*************
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
X,y = make_blobs(n_samples=1000, centers=2, random_state=0, cluster_std=0.60)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42, test_size=0.3)
model=SVC(kernel='linear', C=1E10)
model.fit(X,y)
Y=model.fit(X,y)
print(Y)
y_predi=model.predict(X_test)
accuracy_score(y_test,model.predict(X_test))
accuracy_score=accuracy_score(y_test,model.predict(X_test))
print("Accuracy Score = ", accuracy_score)























