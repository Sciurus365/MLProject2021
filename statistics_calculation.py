# statistics
import pandas as pd
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

original_end = pd.read_csv('original_end.csv', sep=',')
cr_predbytrain = pd.read_csv('cr_predbytrain.csv', sep=',')
m_train = pd.read_csv('m_train.csv', sep=',')
cr_prebyall = pd.read_csv('cr_prebyall.csv', sep=',')
m_all = pd.read_csv('m_all.csv', sep=',')

# for train and test data
y_test = original_end['V1']
## for voice 1 by cr
y_pred = cr_predbytrain['V1']

MSE = mean_squared_error(y_test,y_pred)
print("Following are for voice 1 by cr")
print("MSE = " , MSE)
MAE = mean_absolute_error(y_test,y_pred)
print("MAE = ", MAE)
R2 = r2_score(y_test,y_pred)
print("R2 = ", R2)

## for voice 2 by cr
y_pred = cr_predbytrain['V2']

MSE = mean_squared_error(y_test,y_pred)
print("Following are for voice 2 by cr")
print("MSE = " , MSE)
MAE = mean_absolute_error(y_test,y_pred)
print("MAE = ", MAE)
R2 = r2_score(y_test,y_pred)
print("R2 = ", R2)

## for voice 3 by cr
y_pred = cr_predbytrain['V3']

MSE = mean_squared_error(y_test,y_pred)
print("Following are for voice 3 by cr")
print("MSE = " , MSE)
MAE = mean_absolute_error(y_test,y_pred)
print("MAE = ", MAE)
R2 = r2_score(y_test,y_pred)
print("R2 = ", R2)

## for voice 4 by cr
y_pred = cr_predbytrain['V4']

MSE = mean_squared_error(y_test,y_pred)
print("Following are for voice 4 by cr")
print("MSE = " , MSE)
MAE = mean_absolute_error(y_test,y_pred)
print("MAE = ", MAE)
R2 = r2_score(y_test,y_pred)
print("R2 = ", R2)

# # for voice 1 by m
y_pred = m_train['V1']

MSE = mean_squared_error(y_test,y_pred)
print("Following are for voice 1 by m")
print("MSE = " , MSE)
MAE = mean_absolute_error(y_test,y_pred)
print("MAE = ", MAE)
R2 = r2_score(y_test,y_pred)
print("R2 = ", R2)

# # for voice 2 by m
y_pred = m_train['V2']

MSE = mean_squared_error(y_test,y_pred)
print("Following are for voice 2 by m")
print("MSE = " , MSE)
MAE = mean_absolute_error(y_test,y_pred)
print("MAE = ", MAE)
R2 = r2_score(y_test,y_pred)
print("R2 = ", R2)

# # for voice 3 by m
y_pred = m_train['V3']

MSE = mean_squared_error(y_test,y_pred)
print("Following are for voice 3 by m")
print("MSE = " , MSE)
MAE = mean_absolute_error(y_test,y_pred)
print("MAE = ", MAE)
R2 = r2_score(y_test,y_pred)
print("R2 = ", R2)

# # for voice 4 by m
y_pred = m_train['V4']

MSE = mean_squared_error(y_test,y_pred)
print("Following are for voice 4 by m")
print("MSE = " , MSE)
MAE = mean_absolute_error(y_test,y_pred)
print("MAE = ", MAE)
R2 = r2_score(y_test,y_pred)
print("R2 = ", R2)








