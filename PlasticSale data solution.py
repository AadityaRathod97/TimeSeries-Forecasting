# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 02:03:20 2020

@author: DELL
"""



import pandas as pd
Plastic = pd.read_csv("PlasticSales.csv") #read csv file
month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] #created the list of months

import numpy as np
p = Plastic["Month"][0] # p variable starts with 0 index 
p[0:3] # creating as quarterly of month
Plastic['months']= 0 #creating extra column name as months

for i in range(60):  #159 as per data avability
    p= Plastic["Month"][i] #i = 159
    Plastic['months'][i]= p[0:3]  #[0:3] slicing for the name of the months as 'Jan'
    
month_dummies = pd.DataFrame(pd.get_dummies(Plastic['months']))   # get_dummies is used to convert categorical variable into dummy variable.  String to append DataFrame column names 
Plastic1 = pd.concat([Plastic,month_dummies],axis = 1) #amtrak1 = amtrak + month_dummies just the concatination

Plastic1["t"] = np.arange(1,61)  # creating a column for t
Plastic1["t_squared"] = Plastic1["t"]*Plastic1["t"] # creating a column for t square
Plastic1.columns

Plastic1["log_Sale"] = np.log(Plastic1["Sales"]) # creating a column for log value of ridership
Plastic.rename(columns={"Sales": 'Sales'}, inplace=True) # you can rename the column name by using this rename funtion
Plastic.Sales.plot() #plotting the graph

#Amtrak1.log_Rider.plot()

Train = Plastic1.head(90)
Test = Plastic1.tail(6)

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13))



####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear




##################### Exponential ##############################

Exp = smf.ols('log_Sale~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp




#################### Quadratic ###############################

Quad = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad


################### Additive seasonality ########################

add_sea = smf.ols('Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea



################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Sales~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_Sale~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_Sale~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 



################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse
# so rmse_add_sea_quad has the least value among the models prepared so far


 



# Predicting new values 

predict_data = pd.read_csv("Predict_new.csv")
model_full = smf.ols('Sales~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Plastic1).fit()

pred_new  = pd.Series(add_sea_Quad.predict(predict_data))
pred_new
predict_data["forecasted_Sales"] = pd.Series(pred_new)
