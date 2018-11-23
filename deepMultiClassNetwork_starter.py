#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

from load_mnist import mnist

NO_MOMENTUM = "no_momentum"
MOMENTUM = "momentum"
NAG = "NAG"
RMSPROP = "RMSPROP"
ADAM = "ADAM"
GRADIENT_TECHNIQUE = "gradient_technique"


def relu(Z):
    '''
    computes relu activation of Z

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''

    A = np.maximum(0, Z)
    cache = {}
    cache["Z"] = Z
    return A, cache


def relu_der(dA, cache):
    '''
    computes derivative of relu activation

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    dZ[Z < 0] = 0
    return dZ


def linear(Z):
    '''
    computes linear activation of Z
    This function is implemented for completeness

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = Z
    cache = {"Z": Z}
    return A, cache


def linear_der(dA):
    '''
    computes derivative of linear activation
    This function is implemented for completeness

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    return dZ


def softmax_cross_entropy_loss(Z, Y=np.array([])):
    '''
    Computes the softmax activation of the inputs Z
    Estimates the cross entropy loss

    Inputs: 
        Z - numpy.ndarray (n, m)
        Y - numpy.ndarray (1, m) of labels
            when y=[] loss is set to []
    
    Returns:
        A - numpy.ndarray (n, m) of softmax activations
        cache -  a dictionary to store the activations later used to estimate derivatives
        loss - cost of prediction
    '''
    exps = np.exp(Z - np.max(Z, axis=0))
    A = exps / np.sum(exps, axis=0, keepdims=True)
    cache = {}
    cache["A"] = A
    m = Y.shape[1]
    loss = np.sum(Y * np.log(A + 1e-8))  # - Gives div by 0 error

    loss = (-1 / m) * loss
    return A, cache, loss


def softmax_cross_entropy_loss_der(Y, cache):
    '''
    Computes the derivative of softmax activation and cross entropy loss

    Inputs: 
        Y - numpy.ndarray (1, m) of labels
        cache -  a dictionary with cached activations A of size (n,m)

    Returns:
        dZ - numpy.ndarray (n, m) derivative for the previous layer
    '''
    A = cache["A"]
    dZ = A - Y
    return dZ


def initialize_multilayer_weights(net_dims, gradient_method):
    '''
    Initializes the weights of the multilayer network

    Inputs: 
        net_dims - tuple of network dimensions

    Returns:
        dictionary of parameters
    '''

    np.random.seed(0)
    numLayers = len(net_dims)
    parameters = {}
    for l in range(numLayers - 1):
        parameters["W" + str(l + 1)] = np.random.randn(net_dims[l + 1], net_dims[l]) * np.sqrt(2 / net_dims[l + 1])
        parameters["b" + str(l + 1)] = np.random.randn(net_dims[l + 1], 1) * np.sqrt(2 / net_dims[l + 1])
    parameters["layers"] = numLayers - 1
    parameters[GRADIENT_TECHNIQUE] = gradient_method
    if parameters[GRADIENT_TECHNIQUE] == RMSPROP:
        parameters["RMSprop_3"] = 0
        parameters["RMSprop_2"] = 0
        parameters["RMSprop_1"] = 0
    return parameters


def linear_forward(A, W, b):
    '''
    Input A propagates through the layer 
    Z = WA + b is the output of this layer. 

    Inputs: 
        A - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer

    Returns:
        Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        cache - a dictionary containing the inputs A
    '''
    Z = np.dot(W, A) + b
    cache = {}
    cache["A"] = A
    return Z, cache


def layer_forward(A_prev, W, b, activation):
    '''
    Input A_prev propagates through the layer and the activation

    Inputs: 
        A_prev - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer
        activation - is the string that specifies the activation function

    Returns:
        A = g(Z), where Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        g is the activation function
        cache - a dictionary containing the cache from the linear and the nonlinear propagation
        to be used for derivative
    '''
    Z, lin_cache = linear_forward(A_prev, W, b)
    act_cache = None
    A = None
    if activation == "relu":
        A, act_cache = relu(Z)
    elif activation == "linear":
        A, act_cache = linear(Z)

    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache
    return A, cache


def multi_layer_forward(X, parameters):
    '''
    Forward propgation through the layers of the network

    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}
    Returns:
        AL - numpy.ndarray (c,m)  - outputs of the last fully connected layer before softmax
            where c is number of categories and m is number of samples in the batch
        caches - a dictionary of associated caches of parameters and network inputs
    '''
    L = parameters["layers"]
    A = X
    caches = []
    for l in range(1, L):  # since there is no W0 and b0
        A, cache = layer_forward(A, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)

    AL, cache = layer_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "linear")
    caches.append(cache)
    return AL, caches


def linear_backward(dZ, cache, W, b):
    '''
    Backward prpagation through the linear layer

    Inputs:
        dZ - numpy.ndarray (n,m) derivative dL/dz 
        cache - a dictionary containing the inputs A, for the linear layer
            where Z = WA + b,    
            Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''

    A_prev = cache["A"]
    dW = np.dot(dZ, A_prev.T) / A_prev.shape[1]
    db = np.sum(dZ, axis=1, keepdims=True) / A_prev.shape[1]
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


def layer_backward(dA, cache, W, b, activation):
    '''
    Backward propagation through the activation and linear layer

    Inputs:
        dA - numpy.ndarray (n,m) the derivative to the previous layer
        cache - dictionary containing the linear_cache and the activation_cache
        activation - activation of the layer
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)
    
    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "relu":
        dZ = relu_der(dA, act_cache)
    elif activation == "linear":
        dZ = linear_der(dA)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db


def multi_layer_backward(dAL, caches, parameters):
    '''
    Back propgation through the layers of the network (except softmax cross entropy)
    softmax_cross_entropy can be handled separately

    Inputs: 
        dAL - numpy.ndarray (n,m) derivatives from the softmax_cross_entropy layer
        caches - a dictionary of associated caches of parameters and network inputs
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}

    Returns:
        gradients - dictionary of gradient of network parameters 
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
    '''
    L = len(caches)  # with one hidden layer, L = 2
    gradients = {}
    dA = dAL
    activation = "linear"
    for l in reversed(range(1, L + 1)):
        dA, gradients["dW" + str(l)], gradients["db" + str(l)] = \
            layer_backward(dA, caches[l - 1],
                           parameters["W" + str(l)], parameters["b" + str(l)],
                           activation)
        activation = "relu"
    return gradients


def classify(X, Y, parameters):
    '''
    Network prediction for inputs X

    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters 
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
    Returns:
        YPred - numpy.ndarray (1,m) of predictions
    '''
    # Forward propagate X using multi_layer_forward
    # Get predictions using softmax_cross_entropy_loss
    # Estimate the class labels using predictions
    Zl, caches = multi_layer_forward(X, parameters)
    A, cachel, cost = softmax_cross_entropy_loss(Zl, Y)
    YPred = np.argmax(A, axis=0)
    return np.reshape(YPred, (1, YPred.shape[0]))


def simple_gradient_descent(parameters, gradients, epoch, learning_rate, decay_rate=0.01):
    alpha = learning_rate * (1 / (1 + decay_rate * epoch))
    L = parameters["layers"]
    for l in range(L - 1):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - alpha * gradients["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - alpha * gradients["db" + str(l + 1)]

    return parameters, alpha


'''
    Decay rate for RMS prop is going to be 0.9
'''


def rmsprop(parameters, gradients, epoch, learning_rate, beta=0.9, epsilon=1e-8):
    L = parameters["layers"]
    for l in reversed(range(1, L + 1)):
        parameters["RMSprop_" + str(l)] = beta * parameters["RMSprop_" + str(l)] + (1 - beta) * gradients[
            "dW" + str(l)] ** 2
        parameters["W" + str(l)] = parameters["W" + str(l)] - (
                    learning_rate / np.sqrt(parameters["RMSprop_" + str(l)] + epsilon)) * gradients[
                                       "dW" + str(l)]
    return parameters, learning_rate




def update_parameters(parameters, gradients, epoch, learning_rate, decay_rate=0.01):
    '''
    @TODO - Change the comments
    Updates the network parameters with gradient descent

    Inputs:
        parameters - dictionary of network parameters 
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
        gradients - dictionary of gradient of network parameters 
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
        epoch - epoch number
        learning_rate - step size for learning
        decay_rate - rate of decay of step size - not necessary - in case you want to use
    '''

    gradient_method = parameters[GRADIENT_TECHNIQUE]
    if gradient_method == NO_MOMENTUM:
        return simple_gradient_descent(parameters, gradients, epoch, learning_rate, decay_rate)
    elif gradient_method == RMSPROP:
        return rmsprop(parameters, gradients, epoch, learning_rate)

    # =============================================================================
    #    Remove your condition - create a new function and implement
    #     elif gradient_method == MOMENTUM:
    #         g =2
    #     elif gradient_method == NAG:
    #         g=3
    #     elif gradient_method == ADAM:
    #         g=5
    #
    # =============================================================================
    return parameters, 0


def one_hot(Y, num_classes):
    Y_one_hot = np.zeros((num_classes, Y.shape[1]))
    for i in range(Y.shape[1]):
        Y_one_hot[int(Y[0, i]), i] = 1
    return Y_one_hot


def multi_layer_network(X, Y, vX, vY, net_dims, num_iterations=500, learning_rate=0.2, decay_rate=0.01,
                        gradient_method=NO_MOMENTUM):
    '''
    Creates the multilayer network and trains the network

    Inputs:
        X - numpy.ndarray (n,m) of training data
        Y - numpy.ndarray (1,m) of training data labels
        net_dims - tuple of layer dimensions
        num_iterations - num of epochs to train
        learning_rate - step size for gradient descent
    
    Returns:
        costs - list of costs over training
        parameters - dictionary of trained network parameters
    '''

    parameters = initialize_multilayer_weights(net_dims, gradient_method)
    print(f"Training {parameters['layers']} neural network with {gradient_method} at an initial learning rate of {learning_rate}")
    A0 = X

    costs = []
    vCosts = []
    Y_one_hot = one_hot(Y, 10)
    vY_one_hot = one_hot(vY, 10)
    for ii in range(num_iterations):
        # Forward Prop
        ## call to multi_layer_forward to get activations
        ## call to softmax cross entropy loss 
        Zl, caches = multi_layer_forward(A0, parameters)
        Al, cachel, cost = softmax_cross_entropy_loss(Zl, Y_one_hot)

        vZ1, vCaches = multi_layer_forward(vX, parameters)
        vA1, vCache1, vCost = softmax_cross_entropy_loss(vZ1, vY_one_hot)

        # Backward Prop
        ## call to softmax cross entropy loss der
        ## call to multi_layer_backward to get gradients
        ## call to update the parameters
        dZ = softmax_cross_entropy_loss_der(Y_one_hot, cachel)
        grads = multi_layer_backward(dZ, caches, parameters)
        parameters, alpha = update_parameters(parameters, grads, ii, learning_rate, decay_rate)
        if ii % 10 == 0:
            costs.append(cost)
            vCosts.append(vCost)
        if ii % 10 == 0:
            print("Cost at iteration %i is: %.05f, learning rate: %.05f, valid cost : %.05f" % (ii, cost, alpha, vCost))
    return costs, vCosts, parameters


def main():
    '''
    Trains a multilayer network for MNIST digit classification (all 10 digits)
    To create a network with 1 hidden layer of dimensions 800
    Run the progam as:
        python deepMultiClassNetwork_starter.py "[784,800]"
    The network will have the dimensions [784,800,10]
    784 is the input size of digit images (28pix x 28pix = 784)
    10 is the number of digits
 
    To create a network with 2 hidden layers of dimensions 800 and 500
    Run the progam as:
        python deepMultiClassNetwork_starter.py "[784,800,500]"
    The network will have the dimensions [784,800,500,10]
    784 is the input size of digit images (28pix x 28pix = 784)
    10 is the number of digits
    '''
    # =============================================================================
    #     net_dims = ast.literal_eval( sys.argv[1] ) #Get input from command line
    # =============================================================================
    net_dims = [784, 500, 100]
    net_dims.append(10)  # Adding the digits layer with dimensionality = 10
    print("Network dimensions are:" + str(net_dims))

    # getting the subset dataset from MNIST
    train_data, train_label, test_data, test_label, valid_data, valid_label = \
        mnist(noTrSamples=5000, noTsSamples=1000,
              digit_range=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
              noTrPerClass=500, noTsPerClass=100, noVdSamples=1000, noVdPerClass=100)

    num_iterations = 100
    all_costs = []
    all_vcosts = []

    # Add gradient method and its corresponding learning rate to the array.
    gradient_methods = [NO_MOMENTUM, RMSPROP]
    learning_rates = [0.01, 0.001]

    for learning_rate, gradient_method in zip(learning_rates, gradient_methods):
        costs, vCosts, parameters = multi_layer_network(train_data, train_label, valid_data, valid_label, net_dims,
                                                        num_iterations=num_iterations, learning_rate=learning_rate,
                                                        gradient_method=gradient_method)
        all_costs.append(costs)
        all_vcosts.append(costs)
        # compute the accuracy for training set and testing set
        Y_one_hot = one_hot(train_label, 10)
        t_one_hot = one_hot(test_label, 10)

        train_Pred = classify(train_data, Y_one_hot, parameters)
        test_Pred = classify(test_data, t_one_hot, parameters)

        trAcc = np.count_nonzero(train_Pred == train_label) / train_label.shape[1] * 100
        teAcc = np.count_nonzero(test_Pred == test_label) / test_label.shape[1] * 100
        print("Accuracy for training set is {0:0.3f} %".format(trAcc))
        print("Accuracy for testing set is {0:0.3f} %".format(teAcc))
        print("------------------------------------------------------")
        itr = range(len(costs))
        plt.plot(itr, costs, label="Training Error")
        plt.plot(itr, vCosts, label="Validation Error")
        plt.xlabel("Iterations")
        plt.ylabel("Costs")
        plt.legend(loc="upper right")
        plt.show()


if __name__ == "__main__":
    main()
