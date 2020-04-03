# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:56:15 2020

@author: DELL
"""

import pandas as pd
Amtrak = pd.read_csv("Amtraks.csv") #read csv file
month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] #created the list of months

import numpy as np
p = Amtrak["Month"][0] # p variable starts with 0 index 
p[0:3] # creating as quarterly of month
Amtrak['months']= 0 #creating extra column name as months

for i in range(159):  #159 as per data avability
    p= Amtrak["Month"][i] #i = 159
    Amtrak['months'][i]= p[0:3]  #[0:3] slicing for the name of the months as 'Jan'
    
month_dummies = pd.DataFrame(pd.get_dummies(Amtrak['months']))   # get_dummies is used to convert categorical variable into dummy variable.  String to append DataFrame column names 
Amtrak1 = pd.concat([Amtrak,month_dummies],axis = 1) #amtrak1 = amtrak + month_dummies just the concatination

Amtrak1["t"] = np.arange(1,160)  # creating a column for t
Amtrak1["t_squared"] = Amtrak1["t"]*Amtrak1["t"] # creating a column for t square
Amtrak1.columns

Amtrak1["log_Rider"] = np.log(Amtrak1["Riderships"]) # creating a column for log value of ridership
Amtrak1.rename(columns={"Riderships": 'Riderships'}, inplace=True) # you can rename the column name by using this rename funtion
Amtrak1.Riderships.plot() #plotting the graph

#Amtrak1.log_Rider.plot()

Train = Amtrak1.head(147)
Test = Amtrak1.tail(12)

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13))



####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Riderships~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Riderships'])-np.array(pred_linear))**2))
rmse_linear



##################### Exponential ##############################

Exp = smf.ols('log_Rider~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Riderships'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp




#################### Quadratic ###############################

Quad = smf.ols('Riderships~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Riderships'])-np.array(pred_Quad))**2))
rmse_Quad


################### Additive seasonality ########################

add_sea = smf.ols('Riderships~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Riderships'])-np.array(pred_add_sea))**2))
rmse_add_sea



################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Riderships~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Riderships'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 


################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_Rider~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Riderships'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_Rider~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Riderships'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 



################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse
# so rmse_add_sea_quad has the least value among the models prepared so far


 



# Predicting new values 

predict_data = pd.read_csv("Predict_new.csv")
model_full = smf.ols('Riderships~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Amtrak1).fit()

pred_new  = pd.Series(add_sea_Quad.predict(predict_data))
pred_new
predict_data["forecasted_Ridership"] = pd.Series(pred_new)
