# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 01:29:36 2020

@author: DELL
"""



import pandas as pd
Cocacola = pd.read_csv("CocaCola.csv") #read csv file
quarter = ['Q1','Q2','Q3','Q4'] #created the list of months

import numpy as np
p = Cocacola["Quarter"][0] # p variable starts with 0 index 
p[0:3] # creating as quarterly of month
Cocacola['quarter']= 0 #creating extra column name as months

for i in range(42):  #159 as per data avability
    p= Cocacola["Quarter"][i] #i = 159
    Cocacola['quarter'][i]= p[0:2]  #[0:3] slicing for the name of the months as 'Jan'
    
quater_dummies = pd.DataFrame(pd.get_dummies(Cocacola['quarter']))   # get_dummies is used to convert categorical variable into dummy variable.  String to append DataFrame column names 
Cocacola1 = pd.concat([Cocacola,quater_dummies],axis = 1) #amtrak1 = amtrak + month_dummies just the concatination

Cocacola1["t"] = np.arange(1,43)  # creating a column for t
Cocacola1["t_squared"] = Cocacola1["t"]*Cocacola1["t"] # creating a column for t square
Cocacola1.columns

Cocacola1["log_Sale"] = np.log(Cocacola1["Sales"]) # creating a column for log value of ridership
Cocacola.rename(columns={"Sales": 'Sales'}, inplace=True) # you can rename the column name by using this rename funtion
Cocacola.Sales.plot() #plotting the graph

#Amtrak1.log_Rider.plot()

Train = Cocacola1.head(36)
Test = Cocacola1.tail(9)

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

add_sea = smf.ols('Sales~Q1+Q2+Q3',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1','Q2','Q3']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea



################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Sales~t+t_squared+Q1+Q2+Q3',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1','Q2','Q3','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_Sale~Q1+Q2+Q3',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_Sale~t+Q1+Q2+Q3',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 



################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse
# so rmse_add_sea_quad has the least value among the models prepared so far


 



# Predicting new values 

predict_data = pd.read_csv("Predict_Cocacola.csv")
model_full = smf.ols('Sales~t+t_squared+Q1+Q2+Q3',data=Cocacola1).fit()

pred_new  = pd.Series(add_sea_Quad.predict(predict_data))
pred_new
predict_data["forecasted_Sales"] = pd.Series(pred_new)
