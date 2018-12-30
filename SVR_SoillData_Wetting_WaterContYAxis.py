"""
*************************************************************************************************************************************************************************************************
Kevin Achieng
kachieng@uwyo.edu
Department of Civil and Artichectural Engineering
University of Wyoming

October 29th, 2018
December 17th, 2018
December 30th, 2018


Soil moisture (given suction) & thus Soil Water Retention Curve  for soil (e.g. Loamy Sand) subjected to monotonic wetting is predicted using 
three Support Vector Regression (SVR) models:
	(i) the radial basis function kernel (RBF), 
	(ii) linear and 
	(iii) quadratic kernels

*************************************************************************************************************************************************************************************************

"""

# 1. Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Fix a random seed  to initialize random no. generator
# initialized random number generator to ensure that the results are REPRODUCIBLE
np.random.seed(7)
# 2. Import the soil Data (measured/observed suction and soil moisture)
soilData = pd.read_csv('E:/Fall2018/SupportVectorRegression/SoilMositureData/Data/Submit/Code/Test_data/Wetting2.csv') #change this to your own directory where the SWRC data is stored
X = soilData.iloc[:, 2:3].values # Soil suction (cm): This is create x as matrix instead of vector
X.shape
n_samples=len(X)
print(X)
y = soilData.iloc[:, 0:1].values #Observed Soil moisture ($cm^3$/$cm^3$) : create y as matrix instead of vector
y_mean = soilData['Observed Water Content'].mean()


# 3. Feature Scaling (Standardizing the data i.e. xnew=(x-mean)/std)
# Feature Scaling (Standardizing) is a process of transforming variables to have values in same range so that no variable is dominated by the other.
# SVR libraries do not apply feature scaling by default. Therefore, apply feature scaling to the X,Y dataset

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() # StandardScalar() is the function (inside sklearn library) we need to implement feature scaling.
sc_y = StandardScaler()
X = sc_X.fit_transform(X) # sc_x.fit_transform(X) transforms and fits X into scaled data
y = sc_y.fit_transform(y)


# 4. Define the support vector regression models for rbf, quadratic, and linear kernels
regressor_rbf = SVR(kernel = 'rbf', C=1000, gamma=2.7) #create regressor_rbf SVR class object with kernel ‘rbf’ because this follows Gaussian process
regressor_rbf.fit(X, y) #fit scaled X and y to the object regressor_rbf
regressor_quad = SVR(kernel = 'poly', C=1000, degree=2)
regressor_quad.fit(X, y)
regressor_linear = SVR(kernel = 'linear', C=1000)
regressor_linear.fit(X, y)

# 5. Predict soil moisture using measured/observed suction
y_pred_rbf= regressor_rbf.predict(X) #Predicted y values
y_pred_rbf=y_pred_rbf.reshape(-1,1)
y_pred_quad= regressor_quad.predict(X) #Predicted y values
y_pred_quad=y_pred_quad.reshape(-1,1)
y_pred_linear= regressor_linear.predict(X) #Predicted y values
y_pred_linear=y_pred_linear.reshape(-1,1)

y_pred_rbf=sc_y.inverse_transform(y_pred_rbf)
y_pred_quad=sc_y.inverse_transform(y_pred_quad)
y_pred_linear=sc_y.inverse_transform(y_pred_linear)
print(pd.DataFrame(y_pred_rbf)) 
print(pd.DataFrame(y_pred_rbf).dtypes)

predicted_SWC=soilData.copy()
predicted_SWC['SWC_rbf']=pd.DataFrame(y_pred_rbf).iloc[:, 0:1].values
predicted_SWC['SWC_quad']=pd.DataFrame(y_pred_quad).iloc[:, 0:1].values
predicted_SWC['SWC_linear']=pd.DataFrame(y_pred_linear).iloc[:, 0:1].values

print(predicted_SWC)
predicted_SWC.to_csv('./Output_files/SVR_rbf_quad_linear_predicted_SWC_Wetting2.csv', encoding='utf-8', index=False)


print(y_pred_rbf.shape)
X = soilData.iloc[:, 2:3].values  #Suction, h(cm)
X=X.reshape(-1,1)

y = soilData['Observed Water Content'].values
y=y.reshape(-1,1)


# 6. Check the Performance of the the three SVR models using Mean Squared Error (MSE), Root MSE (RMSE), Willmot's index of agreement (d1), and and R-Squared
MSE_rbf=mean_squared_error(y, y_pred_rbf)
RMSE_rbf=np.sqrt(MSE_rbf)
R2_rbf=r2_score(y, y_pred_rbf)
print('RMSE_rbf: %.2f' % (RMSE_rbf))
print('R^2 rbf: %.2f' % (r2_score(y, y_pred_rbf)))
MSE_quad=mean_squared_error(y, y_pred_quad)
RMSE_quad=np.sqrt(MSE_quad)
R2_quad=r2_score(y, y_pred_quad)
print('RMSE_quad: %.2f' % (RMSE_quad))
print('R^2 quad: %.2f' % (r2_score(y, y_pred_quad)))
MSE_linear=mean_squared_error(y, y_pred_linear)
RMSE_linear=np.sqrt(MSE_linear)
R2_linear=r2_score(y, y_pred_linear)
print('RMSE_linear: %.2f' % (RMSE_linear))
print('R^2 linear: %.2f' % (r2_score(y, y_pred_linear)))

# Willmot's index of agreement d1
d1_rbf = 1-(((soilData['Observed Water Content']-predicted_SWC['SWC_rbf']).sum())/(abs(soilData['Observed Water Content']-y_mean)+abs(predicted_SWC['SWC_rbf']-y_mean)).sum())
d1_quad = 1-(((soilData['Observed Water Content']-predicted_SWC['SWC_quad']).sum())/(abs(soilData['Observed Water Content']-y_mean)+abs(predicted_SWC['SWC_quad']-y_mean)).sum())
d1_linear = 1-(((soilData['Observed Water Content']-predicted_SWC['SWC_linear']).sum())/(abs(soilData['Observed Water Content']-y_mean)+abs(predicted_SWC['SWC_linear']-y_mean)).sum())

#'''
# 7. plot the SWRC graph for observed and modelled (Support Vector Regression) SWRC
fig = plt.figure()
ax = plt.subplot(111)
#Plot the measured/observed data
#plt.scatter(soilData.iloc[:, 0:1].values, soilData.iloc[:, 1:2].values, color = 'red') #Un-Standardized version, Method1
plt.scatter(y, X, label='Observed Data, '+'N='+str(n_samples), facecolors='none', edgecolors='r') #Un-Standardized version, Method2

X = soilData.iloc[:, 2:3].values 
n_samples=len(X)
X=X.reshape(-1,1)
#Plot modelled SWRC
	#Semi-log plot: Linear X-axis (soil moisture) and log Y-axis (suction)  plot 
plt.semilogy(y_pred_rbf, soilData.iloc[:, 2:3], 'bo', label='RBF SVR: $R^2=$' + str(round(r2_score(soilData.iloc[:, 0:1].values, y_pred_rbf),2))+ ', RMSE='+str(round(np.sqrt(mean_squared_error(soilData.iloc[:, 0:1].values, y_pred_rbf)),3)) + '$cm^3$/$cm^3$'+', d1='+str(round(d1_rbf,2)))
plt.semilogy(y_pred_quad, soilData.iloc[:, 2:3], 'g+', label='Quadratic SVR: $R^2=$' + str(round(r2_score(soilData.iloc[:, 0:1].values, y_pred_quad),2))+ ', RMSE='+str(round(np.sqrt(mean_squared_error(soilData.iloc[:, 0:1].values, y_pred_quad)),3)) + '$cm^3$/$cm^3$'+', d1='+str(round(d1_quad,2)))
plt.semilogy(y_pred_linear, soilData.iloc[:, 2:3], 'ks', label='Linear SVR: $R^2=$' + str(round(r2_score(soilData.iloc[:, 0:1].values, y_pred_linear),2))+ ', RMSE='+str(round(np.sqrt(mean_squared_error(soilData.iloc[:, 0:1].values, y_pred_linear)),3)) + '$cm^3$/$cm^3$'+', d1='+str(round(d1_linear,2)))

plt.xlabel('Soil Water Content ($cm^3$/$cm^3$)')
plt.ylabel('Suction (cm)')
plt.ylim([0, 100])
plt.legend(loc='lower left')
#plt.legend(loc='upper right')
#ax.legend(ncol=2,handleheight=2.1, labelspacing=0.05, bbox_to_anchor=(0.5, -0.1), loc='upper center')

plt.tight_layout()
plt.savefig('./Output_files/SVR_Wetting2_SWC_C=1000.png', bbox_inches="tight")
plt.savefig('./Output_files/SVR_Wetting2_SWC_C=1000.pdf', bbox_inches="tight")
plt.show()

#'''