from csv import reader
from numpy import array, random
from time import time
from tabulate import tabulate
from random import seed
import sys

from network import Network

start_time = time()


def main(*args: any):
    args = array(args[0]).astype(float)

    data = import_data('./BreastTissue.csv')
    data = array(data, dtype=object)

    # Change to array and convert str to float
    # data = [[float(col) for col in row] for row in data]
    data = data.astype(float)

    # T = [row[0] for row in data]
    # P = [[row[col] for col in range(1, data_cols)] for row in data]
    T, P = data[:, :1], data[:, 1:]

    P_norm = normalize_min_max(P.T, 0, 1).T
    T_norm = normalize_min_max(T.T, 0, 1).T
    # print(tabulate(P_norm[:, 0:15]))

    layer_size = [len(P[0]), int(0.3 * len(data)), 1]

    data_norm = array([(P_norm[i], T[i]) for i in range(0, len(T))], dtype=object)

    spliter = random.rand(len(data_norm)) <= 0.17
    train_data, test_data = data_norm[~spliter], data_norm[spliter]

    seed(1)
    network = Network(layer_size[0], layer_size[1], layer_size[2])
    network.train(data_norm, int(args[0]), args[1], test_data)  # epochs: 20000 eta:0.01
    network.predict(data_norm)


# Read data from file
def import_data(name: str):
    with open(name, 'r') as file:
        return [line for line in reader(file, delimiter=';')]


# Data normalization "min max"
def normalize_min_max(table: any, y_min: float, y_max: float):
    table_len = len(table)
    for row in range(0, table_len):
        min_value, max_value = min(table[row]), max(table[row])
        table[row] = [
            (y_max - y_min) * (table[row][col] - min_value) / (max_value - min_value) + y_min if min_value != max_value else max_value
            for col in range(0, len(table[row]))
        ]
    return table


if __name__ == '__main__':
    main(sys.argv[1:])
    print("--> %s ms <--" % ((time() - start_time) * 1000))
