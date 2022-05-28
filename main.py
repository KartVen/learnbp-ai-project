from csv import reader
from numpy import array, arange
from time import time
from tabulate import tabulate
from sys import argv
from sklearn.model_selection import train_test_split

from network import Network

start_time = time()


def main(*args):
    args = array(args[0]).astype(float)

    train_data, test_data = import_and_form_data("BreastTissue.csv", ';')

    layer_sizes = [9, 22, 16, 6]
    epochs = int(args[0]) if len(args) == 3 else 2000
    eta = args[1] if len(args) == 3 else 0.01
    mini_batch_size = 21
    error_goal = 0.25

    x_eta = eta
    # for x_eta in arange(eta, 0.35, 0.02):
    for s1 in range(1, 20, 1):
        layer_sizes[1] = s1
        for s2 in range(1, 20, 1):
            layer_sizes[2] = s2
            network_bp = Network(layer_sizes)
            network_bp.train(train_data, mini_batch_size, epochs, x_eta, error_goal, test_data=test_data)


def import_data(name, delimiter):
    with open(name, 'r') as file:
        return [line for line in reader(file, delimiter=delimiter)]


def import_and_form_data(name, delimiter):
    data = array(import_data(name, delimiter))
    data = data.astype(float)

    data = normalize_min_max(data.T, 0, 1).T

    train_data, test_data = train_test_split(data, test_size=0.20, random_state=25)

    P, T = train_data[:, 1:], train_data[:, :1]
    P_test, T_test = test_data[:, 1:], test_data[:, :1]
    T_vector = create_vector_target(T * 5, 6)
    T_test_vector = create_vector_target(T * 5, 6)

    train_data = [(array([P[i]]).T, array([T_vector[i]]).T) for i in range(0, len(T))]
    test_data = [(array([P_test[i]]).T, array([T_test_vector[i]]).T) for i in range(0, len(T_test))]

    return train_data, test_data


# Function of normalization data
def normalize_min_max(table, y_min, y_max):
    for row in range(0, len(table)):
        min_value, max_value = min(table[row]), max(table[row])
        table[row] = [(y_max - y_min) * (col - min_value) / (max_value - min_value) + y_min if min_value != max_value else max_value for col in table[row]]
    return table


# Function of restore data before normalization
def denormalize(table, y_min, y_max, min_value, max_value):
    for row in range(0, len(table)):
        table[row] = [(col * min_value - col * max_value - min_value * y_max + max_value * y_min) / (y_min - y_max) for col in table[row]]
    return table


# Function of creating vector of Outputs
def create_vector_target(table, vector_scale):
    table_len = len(table)
    table_scale = vector_scale
    return array([[1 if col == table[row] else 0 for col in range(table_scale)] for row in range(table_len)])


if __name__ == "__main__":
    main(argv[1:])
    print("--> %s ms <--" % ((time() - start_time) * 1000))
