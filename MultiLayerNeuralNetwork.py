import datetime

import numpy as np

from load_mnist import mnist
import matplotlib.pyplot as plt

NO_MOMENTUM = "No Momentum"
MOMENTUM = "Polyack's Momentum"
NAG = "Accelerated Gradient"
RMSPROP = "RMSProp"
ADAM = "ADAM"


def one_hot(y, num_classes):
    y_one_hot = np.zeros((num_classes, y.shape[1]))
    for i in range(y.shape[1]):
        y_one_hot[int(y[0, i]), i] = 1
    return y_one_hot


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
    cache = {"A": A}
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
    cache = {"A": A}
    return Z, cache


def linear_backward(dZ, cache, W):
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


class MultiLayerNeuralNetwork:
    def __init__(self, dimensions, learning_rate, num_iterations, gradient_method, decay_rate=0.01):
        self.caches = []
        self.validation_costs = []
        self.costs = []
        self.gradients = {"dW": [], "db": []}
        self.parameters = {}
        self.dimensions = dimensions
        self.numLayers = len(self.dimensions) - 1
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.decay_rate = decay_rate
        self.gradient_method = gradient_method
        self.W_values = []
        self.b_values = []
        self.v_values = []
        self.m_values = []
        self.fake_W_values = []
        self.fake_gradients = {"dW": [], "db": []}
        return

    def initialize_params(self):
        np.random.seed(0)
        for l in range(self.numLayers):
            self.W_values.append(
                np.random.randn(self.dimensions[l + 1], self.dimensions[l]) * np.sqrt(2 / self.dimensions[l + 1]))
            self.b_values.append(np.random.randn(self.dimensions[l + 1], 1) * np.sqrt(2 / self.dimensions[l + 1]))

            # V is used by momentum, Adam and RMS Prop
            self.v_values.append(0)
            # M is used by Adam
            self.m_values.append(0)
            # Fakevalues used by NAG
            self.fake_W_values.append(self.W_values[l] - (1e-3 * self.v_values[l]))

    def layer_forward(self, A_prev, W, b, activation):
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

        cache = {"lin_cache": lin_cache, "act_cache": act_cache}
        return A, cache

    def multi_layer_forward(self, X, useFake=False):
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

        A = X
        L = self.numLayers - 1
        caches = []
        if useFake:
            w_values = self.fake_W_values
        else:
            w_values = self.W_values

        for l in range(L):  # since there is no W0 and b0
            A, cache = self.layer_forward(A, w_values[l], self.b_values[l], "relu")
            caches.append(cache)
        AL, cache = self.layer_forward(A, w_values[L], self.b_values[L], "linear")
        caches.append(cache)
        return AL, caches

    def layer_backward(self, dA, cache, W, b, activation):
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
        dA_prev, dW, db = linear_backward(dZ, lin_cache, W)
        return dA_prev, dW, db

    def multi_layer_backward(self, dAL, caches, useFake=False):
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
        dA = dAL
        activation = "linear"
        dw_values = []
        db_values = []
        if useFake:
            w_values = self.fake_W_values
        else:
            w_values = self.W_values

        for l in reversed(range(1, L + 1)):
            dA, dW, db = self.layer_backward(dA, caches[l - 1], w_values[l - 1], self.b_values[l - 1], activation)
            dw_values.append(dW)
            db_values.append(db)
            activation = "relu"

        db_values.reverse()
        dw_values.reverse()
        if useFake:
            self.fake_gradients["dW"] = dw_values
            self.fake_gradients["db"] = db_values
        else:
            self.gradients["dW"] = dw_values
            self.gradients["db"] = db_values

    def batch_gradient_descent(self, epoch):
        alpha = self.learning_rate * (1 / (1 + self.decay_rate * epoch))
        L = self.numLayers
        dw_values = self.gradients["dW"]
        db_values = self.gradients["db"]
        for l in range(L):
            self.W_values[l] = self.W_values[l] - alpha * dw_values[l]
            self.b_values[l] = self.b_values[l] - alpha * db_values[l]
        return alpha

    def polyackmomentum(self, beta=1e-3):
        L = self.numLayers
        dw_values = self.gradients["dW"]
        for l in reversed(range(L)):
            # V (t + 1) = Beta * v(t)  + ( 1- beta ) dw
            # W (t + 1 ) = w(t)  - alpha *  V (t + 1
            self.v_values[l] = (beta * self.v_values[l]) + (1 - beta) * (dw_values[l])
            self.W_values[l] = self.W_values[l] - (self.learning_rate * self.v_values[l])
        return self.learning_rate

    def rmsprop(self, beta=0.9, epsilon=1e-8):
        L = self.numLayers
        dw_values = self.gradients["dW"]
        for l in reversed(range(L)):
            self.v_values[l] = (beta * self.v_values[l]) + (1 - beta) * dw_values[l] ** 2
            self.W_values[l] = self.W_values[l] - (self.learning_rate / np.sqrt(self.v_values[l] + epsilon)) * dw_values[l]
        return self.learning_rate

    def adam(self, beta1=0.9, beta2=0.999, epsilon=1e-7):
        L = self.numLayers
        dw_values = self.gradients["dW"]
        for l in reversed(range(L)):
            self.m_values[l] = (beta1 * self.m_values[l]) + (1 - beta1) * dw_values[l]
            self.v_values[l] = (beta2 * self.v_values[l]) + (1 - beta2) * dw_values[l] ** 2
            v_hat = self.v_values[l] / (1 - beta2)
            m_hat = self.m_values[l] / (1 - beta1)
            self.W_values[l] = self.W_values[l] - (self.learning_rate / (np.sqrt(v_hat) + epsilon)) * m_hat
        return self.learning_rate

    def nag(self, alpha=0.001):
        L = self.numLayers
        fake_dw_values = self.fake_gradients["dW"]
        for l in reversed(range(L)):
            self.v_values[l] = (self.v_values[l] * alpha) + (self.learning_rate * fake_dw_values[l])
            self.W_values[l] = self.W_values[l] - self.v_values[l]
            self.fake_W_values[l] = self.W_values[l] - (alpha * self.v_values[l])

        return self.learning_rate

    def update_parameters(self, epoch):
        gradient_method = self.gradient_method
        if gradient_method == NO_MOMENTUM:
            return self.batch_gradient_descent(epoch)
        elif gradient_method == RMSPROP:
            return self.rmsprop()
        elif gradient_method == MOMENTUM:
            return self.polyackmomentum()
        elif gradient_method == ADAM:
            return self.adam()
        elif gradient_method == NAG:
            return self.nag()
        return 0

    def fit(self, X, Y, vX, vY):
        self.initialize_params()
        print(
            f"Training {self.numLayers} layer neural network with {self.gradient_method} at an initial learning rate of {self.learning_rate}")
        A0 = X
        Y_one_hot = one_hot(Y, 10)
        vY_one_hot = one_hot(vY, 10)

        for ii in range(self.num_iterations):
            Zl, caches = self.multi_layer_forward(A0)
            Al, cachel, cost = softmax_cross_entropy_loss(Zl, Y_one_hot)

            vZ1, vCaches = self.multi_layer_forward(vX)
            vA1, vCache1, vCost = softmax_cross_entropy_loss(vZ1, vY_one_hot)

            dZ = softmax_cross_entropy_loss_der(Y_one_hot, cachel)

            self.multi_layer_backward(dZ, caches)

            # If its NAG, we need to do the same for Fake W's as well.
            if NAG == self.gradient_method:
                fakeZ1, fakeCaches = self.multi_layer_forward(A0, True)
                fakeAl, fakeCachel, fakeCost = softmax_cross_entropy_loss(fakeZ1, Y_one_hot)
                fakedZ = softmax_cross_entropy_loss_der(Y_one_hot, fakeCachel)
                self.multi_layer_backward(fakedZ, fakeCaches, True)

            alpha = self.update_parameters(ii)
            self.costs.append(cost)
            self.validation_costs.append(vCost)
            if ii % 10 == 0:
                print("Cost at iteration %i is: %.05f, learning rate: %.05f, valid cost : %.05f" % (
                    ii, cost, alpha, vCost))

    def predict(self, X, Y):
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
        Zl, caches = self.multi_layer_forward(X)
        A, cachel, cost = softmax_cross_entropy_loss(Zl, Y)
        YPred = np.argmax(A, axis=0)
        return np.reshape(YPred, (1, YPred.shape[0]))


def plot_costs_graph(costs, gradient_methods, filename, learning_rate, validation=False):
    plt.clf()
    plt.figure(figsize=(12, 6))
    for i, cost in enumerate(costs):
        itr = len(cost)
        plt.plot(np.arange(itr), cost, label=gradient_methods[i])

    plt.xlabel("Iterations")
    plt.legend(loc="upper right")
    if validation:
        plt.title(f"Validation Cost vs Iterations at LR: {learning_rate}")
        plt.ylabel("Validation Costs")
    else:
        plt.title(f"Cost vs Iterations at LR: {learning_rate}")
        plt.ylabel("Training Costs")
    plt.savefig(filename)


def plot_bar_graph(values, lables, filename, title, ylabel):
    plt.clf()
    plt.figure(figsize=(12, 6))
    y_pos = [0, 3, 6, 9, 12]
    plt.bar(y_pos, values, align='center', alpha=0.5)
    plt.xticks(y_pos, lables)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)


def main():
    net_dims = [784, 500, 100, 10]
    train_data, train_label, test_data, test_label, valid_data, valid_label = \
        mnist(noTrSamples=5000, noTsSamples=500,
              digit_range=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
              noTrPerClass=500, noTsPerClass=50, noVdSamples=1000, noVdPerClass=100)

    num_iterations_list = [100, 500, 1000]

    # Add gradient method and its corresponding learning rate to the array.
    gradient_methods = [NO_MOMENTUM, MOMENTUM, NAG, RMSPROP, ADAM]
    learning_rates = [0.1, 0.05, 0.01, 0.001]
    for num_iterations in num_iterations_list:
        print(f"Running {num_iterations} Iterations")
        for learning_rate in learning_rates:
            all_costs = []
            all_vcosts = []
            time_per_algo = []
            training_accuracy_per_algo = []
            testing_accuracy_per_algo = []
            for gradient_method in gradient_methods:
                start_time = datetime.datetime.now()
                mm = MultiLayerNeuralNetwork(net_dims, learning_rate, num_iterations, gradient_method)
                mm.fit(train_data, train_label, valid_data, valid_label)
                end_time = datetime.datetime.now()

                # compute the accuracy for training set and testing set
                Y_one_hot = one_hot(train_label, 10)
                t_one_hot = one_hot(test_label, 10)
                train_Pred = mm.predict(train_data, Y_one_hot)
                test_Pred = mm.predict(test_data, t_one_hot)

                trAcc = np.count_nonzero(train_Pred == train_label) / train_label.shape[1] * 100
                teAcc = np.count_nonzero(test_Pred == test_label) / test_label.shape[1] * 100
                print(f"Accuracy for training set is {trAcc:0.03f} %")
                print(f"Accuracy for testing set is {teAcc:0.03f} %")
                print(
                    f"Time for the algo - {gradient_method} for {num_iterations} iterations was {(end_time - start_time).total_seconds()}")
                print("------------------------------------------------------")
                time_per_algo.append((end_time - start_time).total_seconds())
                testing_accuracy_per_algo.append(teAcc)
                training_accuracy_per_algo.append(trAcc)
                all_vcosts.append(mm.validation_costs)
                all_costs.append(mm.costs)

            plot_costs_graph(all_costs, gradient_methods, f"outputs/allcosts-{num_iterations}-{learning_rate}.png",
                             learning_rate, False)
            plot_costs_graph(all_vcosts, gradient_methods,
                             f"outputs/allValidcosts-{num_iterations}-{learning_rate}.png", learning_rate, True)
            print(time_per_algo)
            print(testing_accuracy_per_algo)
            print(training_accuracy_per_algo)

            plot_bar_graph(time_per_algo, gradient_methods,
                           f"outputs/time-{num_iterations}-{learning_rate}.png",
                           f"Plot of Time taken when using various algorithms for learning rate {learning_rate}",
                           "Time")

            plot_bar_graph(testing_accuracy_per_algo, gradient_methods,
                           f"outputs/testing-accuracy-{num_iterations}-{learning_rate}.png",
                           f"Plot of Testing Accuracy at LR: {learning_rate}",
                           "Testing Accuracy")

            plot_bar_graph(training_accuracy_per_algo, gradient_methods,
                           f"outputs/training-accuracy-{num_iterations}-{learning_rate}.png",
                           f"Plot of Training Accuracy at LR: {learning_rate}",
                           "Training Accuracy")


if __name__ == "__main__":
    main()
