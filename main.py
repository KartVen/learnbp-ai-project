from csv import reader
from numpy import array
from time import time
from tabulate import tabulate

from network import Network

start_time = time()


def main():
    data = importData('./BreastTissueLight.csv')
    data = array(data)

    # Change to array and convert str to float
    # data = [[float(col) for col in row] for row in data]
    data = data.astype(float)

    # T = [row[0] for row in data]
    # P = [[row[col] for col in range(1, data_cols)] for row in data]
    T, P = data[:, 0], data[:, 1:]

    P_norm = normalizeMinMax(P.T, -1, 1).T
    # print(tabulate(P_norm[:, 0:15]))

    layer_size = [len(P[0]), 12, 1]

    data_norm = [(P_norm[i], [T[i]]) for i in range(0, len(T))]
    # print(tabulate(data_norm))

    network = Network(layer_size[0], layer_size[1], layer_size[2])
    network.train(data_norm, 1, 0.01)


# Read data from file
def importData(name):
    with open(name, 'r') as file:
        return [line for line in reader(file, delimiter=';')]


# Data normalization "min max"
def normalizeMinMax(table, y_min, y_max):
    table_len = len(table)
    for row in range(0, table_len):
        min_value, max_value = min(table[row]), max(table[row])
        table[row] = [(y_max - y_min) * (col - min_value) / (max_value - min_value) + y_min for col in table[row]]
    return table


if __name__ == '__main__':
    main()
    print("--> %s ms <--" % ((time() - start_time) * 1000))
