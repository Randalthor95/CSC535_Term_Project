import csv
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


def read_adjacent_states(path_):
    states_ = []
    adjacent_states_ = []
    with open(path_, encoding="utf-8") as file:
        for line in file:
            first = True
            temp = []
            for word in line.split():
                if first:
                    states_.append(word)
                    first = False
                else:
                    temp.append(word)
            adjacent_states_.append(temp)
    return states_, adjacent_states_


def make_adjacency_matrix(states_, adjacent_states_):
    adjacency_matrix_ = np.identity(len(states_))

    for i, entry in enumerate(adjacent_states_):
        for state in entry:
            adjacency_matrix_[i, states.index(state)] = 1

    return adjacency_matrix_


path = './states.txt'

states, adjacent_states = read_adjacent_states(path)
print(states)
print(adjacent_states)

adjacency_matrix = make_adjacency_matrix(states, adjacent_states)
print(adjacency_matrix)
