import numpy as np
import math

np.seterr(all='ignore')

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
		

class LinearRegression:
    ''' NOTE : For Large dataset Make sure data must be scaled before training 
        to get Good Score of the model'''
    def __init__(self):
        self.__epochs = 2000
        self.__lr = 0.074

    def _forword_propagation(self):
        _z = np.dot(self.__thetas,self.__x)
        return _z
    
    def _compute_cost(self, _z):
        _cost = (1/2*self.__sample_size) * np.sum(np.square(_z - self.__y))
        return _cost

    def _back_propagation(self, _z):
        ''' _d_theta = dJ/dTheta '''
        _dz = (1/self.__sample_size) * (_z - self.__y)
        _d_theta = np.dot(_dz,self.__x.T)
        return _d_theta
    
    def _gradient_decent_update(self, _d_theta):
        self.__thetas = self.__thetas - (self.__lr * _d_theta)

    def _apply(self):
        self.__gradient_decent_data = {'cost':[],'epochs':[]}
        for _epoch in range(1,self.__epochs+1):
            y_hat = self._forword_propagation()
            _cost_value = self._compute_cost(y_hat)
            _dt = self._back_propagation(y_hat)
            self._gradient_decent_update(_dt)
            if (_epoch == 1) or (_epoch % 20 == 0):
                if self.__pgd:
                    self.__gradient_decent_data['cost'].append(_cost_value)
                    self.__gradient_decent_data['epochs'].append(_epoch)
                if self.__display_logs:
                    _mse = np.mean(np.square(y_hat-self.__y))
                    print(f'Cost Value : {_cost_value}  |  Mean Squred Error : {_mse}  |  Epochs : {_epoch}/{self.__epochs}')
					
    def fit(self, x, y, lr= None, epochs= None, display_logs= False, pgd= False):
        try:
            self.__pgd = pgd
            self.__display_logs = display_logs
            self.__x = np.array(x)
            self.__sample_size = self.__x.shape[0]
            _ones = np.ones(self.__sample_size)
            self.__x = np.c_[_ones,self.__x]
            self.__feature_size = self.__x.shape[1]
            self.__x = self.__x.T
            # self.__thetas = np.zeros((1,self.__feature_size))
            self.__thetas = np.random.randn(self.__feature_size).reshape(1,self.__feature_size)
            self.__y = np.array(y).reshape(1,self.__sample_size)
            if lr:
                self.__lr = lr
            if epochs:
                self.__epochs = epochs
            self._apply()
        except Exception as e:
            raise e

    @property
    def intercept_(self):
        try:
            return self.__thetas[0][0]
        except:
            return None

    @property
    def coef_(self):
        try:
            return self.__thetas[0][1:]
        except:
            return None

    def predict(self, x):
        try:
            _x = np.array(x)
            _size = _x.shape[0]
            _ones = np.ones(_size)
            _x = np.c_[_ones,_x]
            _y_pred = np.dot(self.__thetas,_x.T)
            return _y_pred.flatten()
        except Exception as e:
            print(e)

    def score(self, y_true, y_pred, error_type='mse'):
        try:
            error_list = ['mse','rmse','r2_score']
            if error_type in error_list:
                _y_true = np.array(y_true)
                _y_pred = np.array(y_pred)
                if error_type == 'r2_score':
                    _error = 1 - ((np.sum((_y_true-_y_pred)**2)) / (np.sum((_y_true-np.mean(_y_true))**2)))
                    return _error
                elif error_type == 'rmse':
                    _error = (1/len(_y_true)) * np.sum((_y_true-_y_pred)**2)
                    return np.sqrt(_error)
                elif error_type == 'mse':
                    _error = (1/len(_y_true)) * np.sum((_y_true-_y_pred)**2)
                    return _error
                else:
                    pass
            else:
                print(f'Error Invalid Matrix Name... Please choose among : {error_list}')
        except Exception as e:
            print(e)

