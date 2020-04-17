#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
import networkx as nx

if __name__ == "__main__":
    with open('states.txt') as f:
        adj_list = []
        states_index = {}
        i = 0
        for line in f.readlines():
            items = line.split()
            adj_list.append((items[0], items[1:]))
            states_index[items[0]] = i
            i = i + 1
    g = nx.Graph()
    for state, adj_states in adj_list:
        u = states_index[state]
        for adj_s in adj_states:
            v = states_index[adj_s]
            g.add_edge(u,v)
    nx.write_edgelist(g, './edge_list/edge_index.txt')