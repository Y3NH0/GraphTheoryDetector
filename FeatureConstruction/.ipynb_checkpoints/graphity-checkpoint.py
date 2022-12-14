import networkx as nx
import json
import r2pipe
import os
import sys
import numpy as np

def gpickle2graph(path):
    G = nx.read_gpickle(path)
    return G

def dot2graph(path):
    # with open(path,'r') as file:
    #     data=file.read()
    # G=nx.DiGraph()
    # for lines in data.split('\n'):
    #     try:
    #         if '-' in lines:
    #             funcA=lines[0:lines.find('-')-1].replace('"','')
    #             funcB=lines[lines.find('-')+4:].replace('"','').replace(';','')
    #             G.add_edge(funcA,funcB)
    #     except:
    #         pass
    # return G
    with open(path, 'r') as myfile:
        data = myfile.read()

    label = {}
    G = nx.DiGraph()
    if len(data.split('\n')) < 9:
        return G

    for lines in data.split('\n'):
        tmp = []
        for words in lines.split():
            if words[0] == '"':
                words = words.replace('"', '')
            tmp.append(words)
        try:
            if tmp[1][1] == 'l':
                func = tmp[1][7:]
                func = func.replace('"', '')
                label[tmp[0]] = func
        except:
            pass

    for lines in data.split('\n'):
        tmp = []
        for words in lines.split():
            if words[0] == '"':
                words = words.replace('"', '')
            tmp.append(words)
        try:
            if tmp[1] == '->':
                G.add_edge(label[tmp[0]], label[tmp[2]])
        except:
            pass
    return G


def get_density(G):

    degree = {d[0]: d[1] for d in G.degree(G.nodes())}
    density = (sum(degree.values())/(len(degree)-1)) / len(degree)

    return density


def shortest_path(G):

    # shortest_path=dict(nx.all_pairs_shortest_path(G.to_undirected()))
    # shortest_path_length={}

    # for start in shortest_path:
    #     tmp={}
    #     for target in shortest_path[start]:
    #         if start!=target:
    #             tmp[target]=len(shortest_path[start][target])
    #     shortest_path_length[start]=tmp

    List = []
    for C in (G.subgraph(c).copy() for c in nx.connected_components(G.to_undirected())):
        List.append(nx.average_shortest_path_length(C))
    shortest_path = []
    shortest_path.append(np.mean(List))
    shortest_path.append(np.max(List))
    shortest_path.append(np.min(List))
    shortest_path.append(np.median(List))
    shortest_path.append(np.std(List))

    return shortest_path


def closeness_centrality(G):

    List = list(nx.closeness_centrality(G).values())
    closeness_centrality = []
    closeness_centrality.append(np.mean(List))
    closeness_centrality.append(np.max(List))
    closeness_centrality.append(np.min(List))
    closeness_centrality.append(np.median(List))
    closeness_centrality.append(np.std(List))

    return closeness_centrality


def betweeness_centrality(G):

    List = list(nx.betweenness_centrality(G.to_undirected()).values())
    betweeness_centrality = []
    betweeness_centrality.append(np.mean(List))
    betweeness_centrality.append(np.max(List))
    betweeness_centrality.append(np.min(List))
    betweeness_centrality.append(np.median(List))
    betweeness_centrality.append(np.std(List))

    return betweeness_centrality


def degree_centrality(G):

    List = list(nx.degree_centrality(G).values())
    degree_centrality = []
    degree_centrality.append(np.mean(List))
    degree_centrality.append(np.max(List))
    degree_centrality.append(np.min(List))
    degree_centrality.append(np.median(List))
    degree_centrality.append(np.std(List))

    return degree_centrality


########################### Un-use function ##############################################

def get_degree(G):

    out_degree = {d[0]: d[1] for d in G.out_degree(G.nodes())}
    in_degree = {d[0]: d[1] for d in G.in_degree(G.nodes())}

    w_out_degree = {i: out_degree[i] /
                    sum(out_degree.values()) for i in out_degree}
    w_in_degree = {i: in_degree[i]/sum(in_degree.values()) for i in in_degree}

    oDegree = np.mean([i for i in w_out_degree.values()])
    iDegree = np.mean([i for i in w_in_degree.values()])

    return oDegree, iDegree


def connected_components(G):
    conComponents = list(nx.connected_components(G.to_undirected()))
    return len(conComponents)


def get_max_min_sp(sp_value):

    sp = []
    for i in sp_value:
        sp.extend([i for i in sp_value[i].values()])
    diameter = max(sp)
    radius = min(sp)

    return diameter, radius
