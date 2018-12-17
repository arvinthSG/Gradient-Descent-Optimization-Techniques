<b>Gradient-Descent-Optimization-Techniques</b> </br>
This is a project that aims to compare and contrast the performance, stability and accuracies of various gradient descent algorithms.</br>

The analysis includes the following algorithms</br>
<ol>
<li>Batch Gradient Descent</li>
<li>Polyack's Momentum</li>
<li>Nestrov's Accelerated Gradient Descent</li>
<li>RMSProp</li>
<li>ADAM</li>
</ol>
</br>
<b>Method</b></br>
The dataset used for the analysis is the Fashion-MNIST dataset. The architecture of the network is [784, 500, 100, 10]. The final layer has the softmax activation function, and the other layers use Relu. Batch gradient descent is used. MiniBatch and Stochastic Gradient Descent are not part of this analysis.</br>

</br>
<b>Conclusions</b></br>
From the analysis, it is clearly seen that ADAM performs the best for this particular dataset.
