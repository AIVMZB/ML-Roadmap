# General topics of Machine Learning

## Optimization problem


### Introduction

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
Materials
>- [Stochastic gradient descent](https://youtu.be/vMh0zPT0tLI?si=834bfrxgOuG-3WX3)

<hr>
<br>

### Gradient descent with momentum
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

# Linear Regression

Tasks:
>- Using given kaggle [dataset](https://www.kaggle.com/datasets/andonians/random-linear-regression) do linear regression on it. Provide a plot, which visualizes the regressed line comparing to actual data.