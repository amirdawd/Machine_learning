import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


def initiate_vectors(data):
    X = data.iloc[:, 0]
    Y = data.iloc[:, 1]
    x = []
    y = []
    for value in range(0, len(X)):
        x.append(X[value])
        y.append(Y[value])
    return x, y


def normalize(x_values, y_values):
    x_values = [x / max(x_values) for x in x_values]
    y_values = [y / max(y_values) for y in y_values]
    return x_values, y_values


def main():
    print('-----------Initiating....-----------')
    data_eng = pd.read_csv('english.csv')
    data_fr = pd.read_csv('french.csv')
    x_eng, y_eng = initiate_vectors(data_eng)
    x_fr, y_fr = initiate_vectors(data_fr)
    print('-----------Normalizing....-----------')
    x_norm, y_norm = normalize(x_eng, y_eng)
    x_norm_fr, y_norm_fr = normalize(x_fr, y_fr)
    print('-----------Ploting initial values....-----------')
    plt.scatter(x_norm, y_norm, c='b', label='English')
    plt.scatter(x_norm_fr, y_norm_fr, c='r', label='French')
    plt.legend(["English", "French"])
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    print('-----------Calculating weights....-----------')
    plot_gradient(x_norm, y_norm)


def batch_descent(x, y, learning_rate, iteration, b, m):
    n = len(x)
    cost = 0
    cost_prev = 0
    for itr in range(0, iteration):
        cost_prev = cost
        for value in range(0, len(x)):
            x_value = x[value]
            y_value = y[value]
            guess = m * x_value + b
            error = y_value - guess
            p_dm = -(2 / n) * np.sum(x_value * error)
            p_db = -(2 / n) * np.sum(error)
            m = m - learning_rate * p_dm
            b = b - learning_rate * p_db
            cost = calculate_cost(x, y, b, m)
        if cost_prev == cost:
            break
        print("m {}, b {}, cost {}, prev_cost {}, iterations {}".format(b, m, cost, cost_prev, itr))
    return b, m


def calculate_cost(x, y, b, m):
    n = len(x)
    cost = 0
    for value in range(0, n):
        cost = 1 / n * np.sum(y[value] - (m * x[value] + b) ** 2)
    return cost


def stochastic_descent(x, y, learning_rate, iteration, b, m):
    n = len(x)
    cost = 0
    cost_prev = 0
    for itr in range(0, iteration):
        cost_prev = cost
        random_value = random.randint(0, n - 1)
        x_value = x[random_value]
        y_value = y[random_value]
        guess = m * x_value + b
        error = y_value - guess
        p_dm = -(2 / n) * np.sum(x_value * error)
        p_db = -(2 / n) * np.sum(error)
        m = m - learning_rate * p_dm
        b = b - learning_rate * p_db
        cost = calculate_cost(x, y, b, m)
        if cost == cost_prev:
            break
        #print("m {}, b {}, cost {}, prev_cost {}, iterations {}".format(b, m, cost, cost_prev, itr))
    return b, m


def plot_gradient(x_norm, y_norm):
    plt.scatter(x_norm, y_norm)
    b_batch, m_batch = batch_descent(x_norm, y_norm, 0.9, 500, 0, 0)
    b_stoch, m_stoch = stochastic_descent(x_norm, y_norm, 0.9, 10000, 0, 0)
    x = np.linspace(-5, 5, 100)
    line_batch = m_batch * x + b_batch
    line_stoch = m_stoch * x + b_stoch
    plt.plot(x, line_batch, 'r-')
    plt.plot(x, line_stoch, 'b-')
    plt.axis([0, 1, 0, 1])
    plt.legend(["Batch", "Stochastic"])
    plt.show()


if __name__ == '__main__':
    main()
