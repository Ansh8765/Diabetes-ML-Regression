from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#data.data - feaututres
#data.target - labels
#data.feauture_name - column names
data = load_diabetes()
print(data)

df = pd.DataFrame(data.data,columns=data.feature_names)
df['Target'] = data.target
print(df.head())
print(df.isnull().sum())
print(df.tail())

X = data.data
y= data.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)
print(X_train.shape,y_train.shape)

model = LinearRegression()
model.fit(X_train,y_train)
print('Training Completed')
y_pred = model.predict(X_test)
print('/nModel Evaluation    ')
y_true = y_test
mse = mean_squared_error(y_true,y_pred)
mae = mean_absolute_error(y_true,y_pred)
print('MSE:',round(mse,2))
print('MAE:',round(mae))
print('RMSE:',round(np.sqrt(mse)))
print('R2 Score',r2_score(y_true,y_pred))


plt.scatter(y_test,y_pred,color='blue',linestyle='--',marker = 'o',label='Actual')
#plt.scatter(y_pred,color='green',linestyle='.',marker = 'D')
plt.title('Actual vs Predicted')
plt.grid()
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()
plt.show()
