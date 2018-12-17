Gradient-Descent-Optimization-Techniques
This is a project that aims to compare and contrast the performance, stability and accuracies of various gradient descent algorithms.

The analysis includes the following algorithms

Batch Gradient Descent
Polyack's Momentum
Nestrov's Accelerated Gradient Descent
RMSProp
ADAM
Method
The dataset used for the analysis is the Fashion-MNIST dataset. The architecture of the network is [784, 500, 100, 10]. The final layer has the softmax activation function, and the other layers use Relu. Batch gradient descent is used. MiniBatch and Stochastic Gradient Descent are not part of this analysis.

Conclusions
From the analysis, it is clearly seen that ADAM performs the best for this particular dataset.
