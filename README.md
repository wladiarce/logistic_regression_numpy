# Logistic regression using numpy
### Learning method: mini-batch gradient descent
----------

## Quick intro to logistic regression

Logistic regression can be used for categorical classification. In our case, we will focus on a model of binary classification.

To do that, a logistic / sigmoid function will be fitted to the data. This function tends to 1 as z -> infinite and to 0 as z -> -infinite, and is as follows:

![](https://latex.codecogs.com/gif.latex?h(z)&space;=&space;\frac{1}{1&plus;\exp{z}})

To obtain a 0 or 1 output, one just simply sets a threshold at 0.5: everything below is zero and everything above is one.

Having a dataset with a series of samples, each of those with their given features, we can express z in matrix form as follows:

![](https://latex.codecogs.com/gif.latex?$z&space;=&space;X\theta$)

![](https://latex.codecogs.com/gif.latex?h(X\theta)&space;=&space;\frac{1}{1&space;&plus;&space;\exp{X\theta}})

Where X is the matrix of [samples] rows and [features] columns, and theta is the vector of the model parameters (weights of our regressor).

The objective will be to obtain that theta vector given a set of X and y([0,1]), that best fits the data. How? Minimizing the cost function of the problem.

The cost function J of a logistic regression problem is the following:

![](https://latex.codecogs.com/gif.latex?J(\theta)&space;=&space;-\frac{1}{n}&space;\sum_{i&space;=&space;1}^{n}&space;(y*log(h(X\theta))&space;&plus;&space;(1-y)*log(1-h(X\theta)))

To find the theta that minimizes the error between the predicted and the real data, an iterative process where that cost function is derived to find its minimum and update the theta parameters to go towards it will be applied. This is called gradient descent:

![](https://latex.codecogs.com/gif.latex?\theta&space;=&space;\theta&space;-&space;\frac{\delta&space;J(\theta)}{\delta\theta})

with

![](https://latex.codecogs.com/gif.latex?\frac{\delta&space;J(\theta)}{\delta\theta}&space;=&space;\frac{1}{n}&space;X^{T}[h(X)&space;-&space;y])

Different ways to update the parameter in function of the gradient can be applied (once for every data entry, once for all the dataset...). In this case, it has been decided to do it in small batches, thus we will be appliying minibatch gradient descent.

--------------------

### Tested with the *Breast Cancer Wisconsin (Diagnostic) Data Set*, included in *sklearn* 
- For each cell nucleus 10 different features are stated (radius, texture, area, compactness...)
- For each feature three values appear: mean, error and worst
- This leads to a 30 dimensional classification problem, into two categories: benign or malignant cell
