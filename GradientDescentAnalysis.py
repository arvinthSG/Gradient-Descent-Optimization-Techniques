import numpy as np
import matplotlib.pyplot as plt
from load_mnist import mnist

np.random.seed(678)


def log(x):
    return 1 / (1 + np.exp(-1 * x))


def d_log(x):
    return log(x) * (1 - log(x))


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    return 1 - np.tanh(x) ** 2


def ReLu(x):
    mask = (x > 0.0) * 1.0
    return x * mask


def d_ReLu(x):
    mask = (x > 0.0) * 1.0
    return mask


def elu(matrix):
    mask = (matrix <= 0) * 1.0
    less_zero = matrix * mask
    safe = (matrix > 0) * 1.0
    greater_zero = matrix * safe
    final = 3.0 * (np.exp(less_zero) - 1) * less_zero
    return greater_zero + final


def d_elu(matrix):
    safe = (matrix > 0) * 1.0
    mask2 = (matrix <= 0) * 1.0
    temp = matrix * mask2
    final = (3.0 * np.exp(temp)) * mask2
    return (matrix * safe) + final


def SGD(training_data, lables, epochs, learning_rate, w1_sgd, w2_sgd, w3_sgd):

    cost_temp_array = []
    total_cost = 0
    for iter in range(epochs):
        for image_index in range(len(training_data)):
            current_image = np.expand_dims(training_data[image_index], axis=0)
            current_image_label = np.expand_dims(lables[image_index], axis=1)

            l1 = current_image.dot(w1_sgd)
            l1A = elu(l1)

            l2 = l1A.dot(w2_sgd)
            l2A = tanh(l2)

            l3 = l2A.dot(w3_sgd)
            l3A = log(l3)

            cost = np.square(l3A - current_image_label).sum() * 0.5
            total_cost = total_cost + cost

            grad_3_part_1 = l3A - current_image_label
            grad_3_part_2 = d_log(l3)
            grad_3_part_3 = l2A
            grad_3 = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

            grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3_sgd.T)
            grad_2_part_2 = d_tanh(l2)
            grad_2_part_3 = l1A
            grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

            grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2_sgd.T)
            grad_1_part_2 = d_elu(l1)
            grad_1_part_3 = current_image
            grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

            w3_sgd = w3_sgd - learning_rate * grad_3
            w2_sgd = w2_sgd - learning_rate * grad_2
            w1_sgd = w1_sgd - learning_rate * grad_1
        if iter % 10 == 0:
            print("a. SGD current Iter: ", iter, " Total Cost: ", total_cost)
        cost_temp_array.append(total_cost)
        total_cost = 0

    return cost_temp_array


def momentum(training_data, lables, epochs, learning_rate, w1_m, w2_m, w3_m):
    v1, v2, v3 = 0, 0, 0
    alpha = 0.001
    total_cost = 0
    cost_temp_array = []
    print('-------------------------')
    for iter in range(epochs):
        for image_index in range(len(training_data)):
            current_image = np.expand_dims(training_data[image_index], axis=0)
            current_image_label = np.expand_dims(lables[image_index], axis=1)

            l1 = current_image.dot(w1_m)
            l1A = elu(l1)

            l2 = l1A.dot(w2_m)
            l2A = tanh(l2)

            l3 = l2A.dot(w3_m)
            l3A = log(l3)

            cost = np.square(l3A - current_image_label).sum() * 0.5
            total_cost = total_cost + cost

            grad_3_part_1 = l3A - current_image_label
            grad_3_part_2 = d_log(l3)
            grad_3_part_3 = l2A
            grad_3 = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

            grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3_m.T)
            grad_2_part_2 = d_tanh(l2)
            grad_2_part_3 = l1A
            grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

            grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2_m.T)
            grad_1_part_2 = d_elu(l1)
            grad_1_part_3 = current_image
            grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

            v3 = v3 * alpha + learning_rate * grad_3
            v2 = v2 * alpha + learning_rate * grad_2
            v1 = v1 * alpha + learning_rate * grad_1

            w3_m = w3_m - v3
            w2_m = w2_m - v2
            w1_m = w1_m - v1
        if iter % 10 == 0:
            print("b. Momentum current Iter: ", iter, " Total Cost: ", total_cost)
        cost_temp_array.append(total_cost)
        total_cost = 0
    return cost_temp_array


def NAG(training_data, lables, epochs, learning_rate, w1_ng, w2_ng, w3_ng):
    # # c. Nesterov accelerated gradient
    v1, v2, v3 = 0, 0, 0
    alpha = 0.001
    total_cost = 0
    cost_temp_array = []
    print('-------------------------')
    for iter in range(epochs):
        for image_index in range(len(training_data)):
            current_image = np.expand_dims(training_data[image_index], axis=0)
            current_image_label = np.expand_dims(lables[image_index], axis=1)

            l1 = current_image.dot(w1_ng)
            l1A = elu(l1)

            l2 = l1A.dot(w2_ng)
            l2A = tanh(l2)

            l3 = l2A.dot(w3_ng)
            l3A = log(l3)

            cost = np.square(l3A - current_image_label).sum() * 0.5
            total_cost = total_cost + cost

            grad_3_part_1 = l3A - current_image_label
            grad_3_part_2 = d_log(l3)
            grad_3_part_3 = l2A
            grad_3 = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

            grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3_ng.T)
            grad_2_part_2 = d_tanh(l2)
            grad_2_part_3 = l1A
            grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

            grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2_ng.T)
            grad_1_part_2 = d_elu(l1)
            grad_1_part_3 = current_image
            grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

            # ------- FAKE GRADIENT --------
            fake_w3_ng = w3_ng - alpha * v3
            fake_w2_ng = w2_ng - alpha * v2
            fake_w1_ng = w1_ng - alpha * v1

            l1 = current_image.dot(fake_w1_ng)
            l1A = elu(l1)

            l2 = l1A.dot(fake_w2_ng)
            l2A = tanh(l2)

            l3 = l2A.dot(fake_w3_ng)
            l3A = log(l3)

            grad_3_part_1 = l3A - current_image_label
            grad_3_part_2 = d_log(l3)
            grad_3_part_3 = l2A
            grad_3_fake = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

            grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(fake_w3_ng.T)
            grad_2_part_2 = d_tanh(l2)
            grad_2_part_3 = l1A
            grad_2_fake = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

            grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(fake_w2_ng.T)
            grad_1_part_2 = d_elu(l1)
            grad_1_part_3 = current_image
            grad_1_fake = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)
            # ------- FAKE GRADIENT --------

            v3 = v3 * alpha + learning_rate * grad_3_fake
            v2 = v2 * alpha + learning_rate * grad_2_fake
            v1 = v1 * alpha + learning_rate * grad_1_fake

            w3_ng = w3_ng - v3
            w2_ng = w2_ng - v2
            w1_ng = w1_ng - v1
        if iter % 10 == 0:
            print("c. Nesterov accelerated gradient current Iter: ", iter, " Total Cost: ", total_cost)
        cost_temp_array.append(total_cost)
        total_cost = 0
    return cost_temp_array


def RMS_Prop(training_data, lables, epochs, learning_rate, w1_RMSprop, w2_RMSprop, w3_RMSprop):
    RMSprop_1, RMSprop_2, RMSprop_3 = 0, 0, 0
    RMSprop_v, RMSprop_e = 0.9, 0.00000001
    total_cost = 0
    cost_temp_array = []

    print('-------------------------')
    for iter in range(epochs):
        for image_index in range(len(training_data)):
            current_image = np.expand_dims(training_data[image_index], axis=0)
            current_image_label = np.expand_dims(lables[image_index], axis=1)

            l1 = current_image.dot(w1_RMSprop)
            l1A = elu(l1)

            l2 = l1A.dot(w2_RMSprop)
            l2A = tanh(l2)

            l3 = l2A.dot(w3_RMSprop)
            l3A = log(l3)

            cost = np.square(l3A - current_image_label).sum() * 0.5
            total_cost = total_cost + cost

            grad_3_part_1 = l3A - current_image_label
            grad_3_part_2 = d_log(l3)
            grad_3_part_3 = l2A
            grad_3 = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

            grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3_RMSprop.T)
            grad_2_part_2 = d_tanh(l2)
            grad_2_part_3 = l1A
            grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

            grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2_RMSprop.T)
            grad_1_part_2 = d_elu(l1)
            grad_1_part_3 = current_image
            grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

            RMSprop_3 = RMSprop_v * RMSprop_3 + (1 - RMSprop_v) * grad_3 ** 2
            RMSprop_2 = RMSprop_v * RMSprop_2 + (1 - RMSprop_v) * grad_2 ** 2
            RMSprop_1 = RMSprop_v * RMSprop_1 + (1 - RMSprop_v) * grad_1 ** 2

            w3_RMSprop = w3_RMSprop - (learning_rate / np.sqrt(RMSprop_3 + RMSprop_e)) * grad_3
            w2_RMSprop = w2_RMSprop - (learning_rate / np.sqrt(RMSprop_2 + RMSprop_e)) * grad_2
            w1_RMSprop = w1_RMSprop - (learning_rate / np.sqrt(RMSprop_1 + RMSprop_e)) * grad_1
        if iter % 10 == 0:
            print("d. RMSprop current Iter: ", iter, " Total Cost: ", total_cost)
        cost_temp_array.append(total_cost)
        total_cost = 0
    return cost_temp_array


def Adam(training_data, lables, epochs, learning_rate, w1_adam, w2_adam, w3_adam):
    # g. Adam
    Adam_m_1, Adam_m_2, Adam_m_3 = 0, 0, 0
    Adam_v_1, Adam_v_2, Adam_v_3 = 0, 0, 0
    Adam_Beta_1, Adam_Beta_2 = 0.9, 0.999
    Adam_e = 0.00000001
    total_cost = 0
    cost_temp_array = []

    print('-------------------------')
    for iter in range(epochs):
        for image_index in range(len(training_data)):
            current_image = np.expand_dims(training_data[image_index], axis=0)
            current_image_label = np.expand_dims(lables[image_index], axis=1)

            l1 = current_image.dot(w1_adam)
            l1A = elu(l1)

            l2 = l1A.dot(w2_adam)
            l2A = tanh(l2)

            l3 = l2A.dot(w3_adam)
            l3A = log(l3)

            cost = np.square(l3A - current_image_label).sum() * 0.5
            total_cost = total_cost + cost

            grad_3_part_1 = l3A - current_image_label
            grad_3_part_2 = d_log(l3)
            grad_3_part_3 = l2A
            grad_3 = grad_3_part_3.T.dot(grad_3_part_1 * grad_3_part_2)

            grad_2_part_1 = (grad_3_part_1 * grad_3_part_2).dot(w3_adam.T)
            grad_2_part_2 = d_tanh(l2)
            grad_2_part_3 = l1A
            grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

            grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2_adam.T)
            grad_1_part_2 = d_elu(l1)
            grad_1_part_3 = current_image
            grad_1 = grad_1_part_3.T.dot(grad_1_part_1 * grad_1_part_2)

            Adam_m_3 = Adam_Beta_1 * Adam_m_3 + (1 - Adam_Beta_1) * grad_3
            Adam_m_2 = Adam_Beta_1 * Adam_m_2 + (1 - Adam_Beta_1) * grad_2
            Adam_m_1 = Adam_Beta_1 * Adam_m_1 + (1 - Adam_Beta_1) * grad_1

            Adam_v_3 = Adam_Beta_2 * Adam_v_3 + (1 - Adam_Beta_2) * grad_3 ** 2
            Adam_v_2 = Adam_Beta_2 * Adam_v_2 + (1 - Adam_Beta_2) * grad_2 ** 2
            Adam_v_1 = Adam_Beta_2 * Adam_v_1 + (1 - Adam_Beta_2) * grad_1 ** 2

            Adam_m_3_hat = Adam_m_3 / (1 - Adam_Beta_1)
            Adam_m_2_hat = Adam_m_2 / (1 - Adam_Beta_1)
            Adam_m_1_hat = Adam_m_1 / (1 - Adam_Beta_1)

            Adam_v_3_hat = Adam_v_3 / (1 - Adam_Beta_2)
            Adam_v_2_hat = Adam_v_2 / (1 - Adam_Beta_2)
            Adam_v_1_hat = Adam_v_1 / (1 - Adam_Beta_2)

            w3_adam = w3_adam - (learning_rate / (np.sqrt(Adam_v_3_hat) + Adam_e)) * Adam_m_3_hat
            w2_adam = w2_adam - (learning_rate / (np.sqrt(Adam_v_2_hat) + Adam_e)) * Adam_m_2_hat
            w1_adam = w1_adam - (learning_rate / (np.sqrt(Adam_v_1_hat) + Adam_e)) * Adam_m_1_hat

        if iter % 10 == 0:
            print("e. Adam current Iter: ", iter, " Total Cost: ", total_cost)
        cost_temp_array.append(total_cost)
        total_cost = 0
    return cost_temp_array


def main():
    # 0. Declare Training Data and Labels
    train_data, train_label, test_data, test_label, valid_data, valid_label = \
            mnist(noTrSamples=5000, noTsSamples=1000,
                  digit_range=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  noTrPerClass=500, noTsPerClass=100, noVdSamples=1000, noVdPerClass=100)

    training_images = train_data.T
    training_lables = train_label.T
   # testing_images = test_data.T
   # testing_lables = test_label.T

    # 2. SAME AMOUNT OF TRAINING
    num_epoch = 10
    learn_rate = 0.0003
    # 1. Declare Weights
    w1 = np.random.randn(784, 500) * 0.2
    w2 = np.random.randn(500, 100) * 0.2
    w3 = np.random.randn(100, 1) * 0.2

    cost_array = []
    cost_temp_array = SGD(training_images, training_lables, num_epoch, learn_rate, w1, w2, w3)
    cost_array.append(cost_temp_array)
    # # ----------------------
    cost_temp_array = momentum(training_images, training_lables, num_epoch, learn_rate,w1, w2, w3)
    cost_array.append(cost_temp_array)
    # # ----------------------

    cost_temp_array = NAG(training_images, training_lables, num_epoch, learn_rate, w1, w2, w3)
    cost_array.append(cost_temp_array)
    # # ----------------------

    cost_temp_array = RMS_Prop(training_images, training_lables, num_epoch, learn_rate, w1, w2, w3)
    cost_array.append(cost_temp_array)
    # ----------------------

    cost_temp_array = Adam(training_images, training_lables, num_epoch, learn_rate, w1, w2, w3)
    cost_array.append(cost_temp_array)
    # ----------------------

    bar_color = ['b', 'g', 'saddlebrown', 'steelblue',
                 'orangered', 'y', 'paleturquoise', 'royalblue',
                 'salmon', 'silver', 'skyblue', 'slateblue', 'peru', 'plum']
    labels_z = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']

    for i in range(len(cost_array)):
        plt.plot(np.arange(num_epoch), cost_array[i], color=bar_color[i], linewidth=3, label=str(labels_z[i]))
    plt.title("Total Cost per Training")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
