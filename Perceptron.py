import numpy as np
from gradient_decend import *


class Perceptron(object):
    def __init__(self, input_size, learning_rate, epochs):
        self.W = np.zeros(input_size + 1)
        self.epochs = epochs
        self.learning_rate = learning_rate

    def evalute(self, matrix_row, w, expected_value):
        guess = 0
        right = 0
        calc_sum = matrix_row[0] + w[1] * matrix_row[1] + w[2] * matrix_row[2]
        if calc_sum > 0:
            guess = 1
        else:
            guess = 0
        if guess == expected_value:
            right = 1
        return guess, right

    def predict(self, matrix_row, w, expected_value):
        correct_guess = 0
        prediction = []
        for row in matrix_row:
            guess, correct_guess = self.evaluate(row, w, expected_value)
            correct_guess += correct_guess
            prediction.append(guess)
        return prediction, correct_guess

    def fit(self, X_value, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X_value[i], 0, 1)
                y = self.predict(x)
                e = d[i] - y
                self.W = self.W + self.lr * e * x


def main():
    matrix, lang = load_libsvm('data')
    per = Perceptron(len(lang), 1, 100)
    matrix[:, 1] = normalize(matrix[:, 1])
    matrix[:, 2] = normalize(matrix[:, 2])
    print(matrix)
    print('Still under construction...')


def load_libsvm(file):
    data = open(file).read().strip().split('\n')
    temp = [data[i].split() for i in range(len(data))]
    y = [float(data[0]) for data in temp]
    X = [['0:1'] + obs[1:] for obs in temp]
    X = [list(map(lambda x: float(x.split(':')[1]), data)) for data in X]
    return np.array(X), y


def normalize(matrix):
    x_values = [x / max(matrix) for x in matrix]
    return x_values


if __name__ == '__main__':
    main()
