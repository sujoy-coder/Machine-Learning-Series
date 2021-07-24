# Machine Learning With Python 

### Implemented Algorithms are :
- ## Simple Regression
<br>
Theoretical Part

![](./assets/simple_regression.png)

How to import this Model

`
from regression.linear_model import SimpleRegression
`

- ## Linear Regression / Multiple Regression 
<br>
Theoretical Part

##### x is independent variables
##### y is dependent variables
##### m is number of samples
##### n is number of features

`x = [[1,2,3,5],
     [5,6,4,7],
     [5,6,1,9],
     [9,7,8,1],
     [2,4,6,8]]     
 ||    dimension -> mxn (5x4)`
<br>
`y = [7,9,8,6]
 ||    dimension -> 1xm (1x5)`

## Steps to Solve :

`Equation : (y_predicted) -> z = w1x1 + w2x2 + w3x3 + w4x4 + b`

1. create constructer of class with parameter:
    weights (dim -> 1xn), bias, learning rate(alpha), epochs

2. forword Propagation:
    first make x = Transpose(x) -> dim : nxm
    find, z = wx + b  -> dim : 1xm

3. calculate the cost function:
    find, J = (1/2*m)sum((z-y)**2)

4. minimize the cost function with back propagation :
    dw = dJ/dw = dJ/dz * dz/dw
    db = dJ/db = dJ/dz * dz/db
    Now, dJ/dz = dz = (1/m)[(z-y)] and dz/dw = x [since z = wx + b]
    Now, dz/db = 1 [since z = wx + b]
    So, dw = dz * (dz/dw) = dz * x.T [to avoid dimenssion issue we have to transpose of x]
    So, db = sum(dz)

5. By gradient decent:
    w = w - alpha * dw
    b = b - alpha * db

6. Repect the process [2-5] for epochs number of times

How to import this Model

`
from regression.linear_model import LinearRegression
`

#### Next Part Commming Soon ...