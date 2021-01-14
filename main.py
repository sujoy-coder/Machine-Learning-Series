from regression.linear_model import SimpleRegression
import pandas as pd

# random data-sets
data_set = {
    'sizes' : [43,21,25,42,57,59,26,55,65,31],
    'prices' : [99,65,79,75,87,81,75,86,90,71]
}

# creating pandas dataframe
data = pd.DataFrame(data_set)
print(data)

x = data['sizes']
y = data['prices']

# initialize of model
regs = SimpleRegression()

# train-test spliting of data
x_train,y_train,x_test,y_test = regs.train_test_split(x,y)
print(x_train,y_train,x_test,y_test)

# traning of model
regs.fit(x_train, y_train)
print('Intercept value :',regs.intercept_)
print('Coefficient value :',regs.coef_)

# making prediction 
y_test_predict_value = regs.predict(x_test)
print('Predicted Value :',y_test_predict_value)

# model score/error checking
root_mean_squre_error = regs.scores(actual_value= y_test, predicted_value= y_test_predict_value, error_type= 'rmse')
print('RMSE :',root_mean_squre_error)
