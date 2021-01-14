import numpy as np
import math

class SimpleRegression:
	''' Formula for Simple Regression is : y = mx + c '''

	def __init__(self):
		# 'intercept_' is c in the equation of : y = mx + c
	 	self.intercept_ = None
	 	# 'coef_' is m in the equation of : y = mx + c
	 	self.coef_ = None


	def __prodSum(self, x, y):
		''' It is a private method '''
		try:
			ans = 0
			for i,j in zip(x,y):
				ans += i*j
			return ans
		except Exception as e:
			print(e)


	def __diffSqureSum(self, x1, x2):
		''' It is a private method '''
		try:
			ans = 0
			for i,j in zip(x1,x2):
				ans += math.pow((i-j),2)
			return ans
		except Exception as e:
			print(e)


	def fit(self, x= None, y= None):
		''' x is feature-dataset / feature column : size --> n x 1
			y is target-field / target column / vector : size --> n x 1
		'''
		try:
			# converting x and y to numpy array
			x = np.array(x)
			y = np.array(y)

			# finding the intercept_ value
			a = (y.sum()*self.__prodSum(x,x)) - (x.sum()*self.__prodSum(x,y))
			b = (x.size*self.__prodSum(x,x)) - math.pow(x.sum(),2)
			self.intercept_ = a/b
			self.intercept_ = round(self.intercept_ , 4)

			# finding coef_ value
			c = (x.size*self.__prodSum(x,y)) - (x.sum()*y.sum())
			d = (x.size*self.__prodSum(x,x)) - math.pow(x.sum(),2)
			self.coef_ = c/d
			self.coef_ = round(self.coef_ , 4)

		except Exception as e:
			print(e)


	def predict(self, x_test= None):
		''' this method takes feature inputs in 1D array/list format
			and returns predicted output values
		'''
		if self.intercept_ or self.coef_:
			y_predicted = list()
			try:
				x_test = np.array(x_test)
				for value in x_test:
					result = self.coef_*value + self.intercept_
					y_predicted.append(round(result,2))
				return y_predicted

			except Exception as e:
				print(e)
				return None

		else:
			print('First Train Your model by using \'fit()\' method...')
			return None


	def scores(self, actual_value= None, predicted_value= None, error_type= 'mse'):
		''' actual_value => format list/array,
			predicted_value => format list/array,
			error_type = ['mse'(default) ,'rmse' ,'r2_score' ,'all']	
		'''
		try:
			# calculate scores of model
			score = dict()
			actual_value = np.array(actual_value)
			predicted_value = np.array(predicted_value)
			# getting MeanSquredError / mse 
			score['mse'] = (self.__diffSqureSum(actual_value,predicted_value)) / actual_value.size
			score['mse'] = round(score['mse'],2)
			# getting RootMeanSquredError / rmse 
			score['rmse'] = math.pow(score['mse'],0.5)
			score['rmse'] = round(score['rmse'],2)
			# getting R-SquredError / r2_score 
			ss_res = self.__diffSqureSum(actual_value,predicted_value)
			meanArray_actual_value = np.array([np.mean(actual_value) for _ in range(actual_value.size)])
			ss_total = self.__diffSqureSum(actual_value,meanArray_actual_value)
			score['r2_score'] = 1 - (ss_res/ss_total)
			score['r2_score'] = round(score['r2_score'],2)

			# returning error type as mentioned...
			if error_type == 'mse':
				return score['mse']
			elif error_type == 'rmse':
				return score['rmse']
			elif error_type == 'r2_score':
				return score['r2_score']
			elif error_type == 'all':
				return score
			else:
				print("choose error_type among --> ['mse','rmse','r2_score','all']")
				return None
		except Exception as e:
			print(e)
			return None


	def train_test_split(self, x= None, y= None, test_size= 0.3):
		''' returns --> x_train, y_train, x_test, y_test '''
		try:
			x = np.array(x)
			y = np.array(y)
			test_size = float(test_size)
			no_of_traning_data = int((1 - test_size)*x.size)
			return x[:no_of_traning_data],y[:no_of_traning_data],x[no_of_traning_data:],y[no_of_traning_data:]
		
		except Exception as e:
			print(e)
			return None,None,None,None
		
