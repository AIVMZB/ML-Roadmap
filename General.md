# General topics of Machine Learning

The content in this markdown includes general machine learning topics, 
which can be encountered accross different fields of artifitial inteligence.

## Optimization problem

Optimization is just simply the process of finding set of function arguments to minimize it.
Maximization problem is just an inverse. Machine learning is essencially a kind of optimization problem, 
because the point is to find such weights to make a loss function as small as possible.

### Introduction

The gradient descent is the basis method for optimizing. In summary, it is just sliding down the function until convergence to a local minima.

Materials:
>- [Gradient descent](https://www.youtube.com/watch?v=sDv4f4s2SB8)
>- [Backpropagation](https://youtu.be/IN2XmBhILt4?si=FerkgpZzuK4Rw1Ct)

Tasks:

>1. __Using gradient descent find the minimum of the function__<br>$f(x) = (x - 2) ^ 2 + x ^ 4 - |x|$
>2. __Using gradient descent find the minimum of the function__<br>$f(x, y) = (x - 2) ^ 2 + \frac{(y + 3) ^ 3}{2} + 9.5y^4 - 5$
>3. __Plot the descent move for previous taks__

<hr>
<br>

### Stochastic gradient descent with momentum

Stochastic gradient descent is an modification to the default gradient descent. 
The main difference is that the random batch of data is used for optimization,
 which increasses convergence speed.

Materials
>- [Stochastic gradient descent](https://youtu.be/vMh0zPT0tLI?si=834bfrxgOuG-3WX3)

<hr>
<br>

### Gradient descent with momentum

The method does optimization process remembering the momentum of "falling". 
Imagine a ball sliding down a heel, its momentum makes it slide accross a straight line.

Materials
>- [Video](https://youtu.be/k8fTYJPd3_I?si=_VfVeqVjWiQqdzyO)

Tasks:
> Optimize the functions from previous task using gradient descent with momentum. Also plot the descent move.
<hr>
<br>

### RMS Prop
Materials
>- [Video](https://youtu.be/_e-LFe_igno?si=IKf9O20EKzt38PxZ)

Tasks:
> Optimize the functions from previous task using RMS prop. Also plot the descent move.

###

<hr><br>

### Adam

Materials
>- [Video](https://www.youtube.com/watch?v=JXQT_vxqwIs)

Tasks
>- Optimize the functions from previous tasks using Adam optimization algorithm. Also plot the descent move. 

<br><br><br>


## Metrics


### Precision, Recall, F1 (Classification)

Materials
>- [Wikipedia page](https://en.wikipedia.org/wiki/Precision_and_recall)
>- [Stat Quest Video](https://youtu.be/Kdsp6soqA7o?si=hPuGRkdHlqzSeCJl)
>- [Precision Recall F1](https://youtu.be/8d3JbbSj-I8?si=voA58UvHivPWK_fg)

Tasks:
>- With a given prediction table calculate the precision and recall.
>- Create a confusion matrics for that data
>- Calculate the F1 score

| Predicted | Actual |
|-----------|--------|
| 1         | 1      |
| 0         | 0      |
| 1         | 0      |
| 1         | 1      |
| 0         | 0      |
| 1         | 1      |
| 0         | 0      |
| 1         | 0      |
| 0         | 1      |
| 0         | 0      |
| 1         | 1      |
| 0         | 0      |
| 1         | 0      |
| 0         | 1      |
| 1         | 1      |
| 0         | 0      |
| 0         | 1      |
| 1         | 1      |
| 0         | 0      |
| 1         | 0      |


<hr><br>

### AUC, ROC

Materials
>- [StatQuest](https://youtu.be/4jRBRDbJemM?si=5gKXxgtq_AE_3iZA)

<hr><br>

### R2

Materials
>- [StatQuest](https://www.youtube.com/watch?v=bMccdk8EdGo&t=4s)

Tasks
>- The regression model has predicted the prices for houses. For a given table provide the R2 metric.

| House | Predicted Price               | Actual Price         |
|-------|-------------------------------|----------------------|
| 1     | 200,000                       | 210,000              |
| 2     | 150,000                       | 160,000              |
| 3     | 180,000                       | 170,000              |
| 4     | 220,000                       | 230,000              |
| 5     | 250,000                       | 240,000              |
| 6     | 300,000                       | 310,000              |
| 7     | 260,000                       | 270,000              |
| 8     | 230,000                       | 220,000              |
| 9     | 280,000                       | 290,000              |
| 10    | 190,000                       | 200,000              |


<hr><br><br><br>

## The simplest models

### Linear Regression

The linear regression is method to fit the function $f(x) = wx + b$ to a given data. 
The learning algorithm finds the best options for $w$ and $b$ to minimize a loss function. 
A loss function can be:
- MSE (Mean Squared Error) loss - $\frac{1}{n}\sum_{i=1}^n (\hat{y_i} - y_i) ^ 2$ 
- MAE (Mean Absolute Error) loss - $\frac{1}{n}\sum_{i=1}^n |\hat{y_i} - y_i|$

where $\hat{y_i}$ - $wx + b$ AKA predicted value, $y_i - groudtruth value, $n$ - set amount.

Materials
>- [StatQuest](https://youtu.be/7ArmBVF2dCs?si=lfe0pGuiNWxFPlRU)

Tasks:
>- Using given kaggle [dataset](https://www.kaggle.com/datasets/andonians/random-linear-regression) do linear regression on it. 
>- Split the dataset to train and test.
>- Provide a plot, which visualizes the regressed line comparing to actual data. 
>- Calculate the `R2` metric using test subset. 

