import os
import sys
import json
import random
import numpy as np
import scipy as sp
import torch
import networkx as nx
from torch import nn
from torch.utils.data import Dataset
from collections import Counter

class HypernymGraph(Dataset):
    def __init__(self, path, embeddings, holdout=0.1, cross_holdout=None, random_seed=2023):
        print(f"loading {path}")
        sys.stdout.flush()
        self.dir = os.path.dirname(path)
        self.embeddings = embeddings
        edges = json.loads(open(path).read())
        self.original_graph = nx.DiGraph(edges)
        self.graph = nx.transitive_closure(self.original_graph)
        self.nodes = list(self.graph.nodes)
        self.edges = list(self.graph.edges)
        self.size = self.graph.number_of_nodes() ** 2 - self.graph.number_of_nodes()

        root = [n for n,d in self.graph.in_degree() if d==0]
        if len(root) != 1:
            raise ValueError(f"{path} graph has no root or more than one root")
        root = root[0]
        random.seed(random_seed)
        random.shuffle(self.edges)
        if cross_holdout!=None:
            num_train_edges = int(self.graph.number_of_edges() * (1 - cross_holdout))
        else:
            num_train_edges = int(self.graph.number_of_edges() * (1 - holdout))
        train_edges = self.edges[:num_train_edges]
        self.train_graph = nx.transitive_closure(nx.DiGraph(train_edges))
        self.left_train_right_train = []
        self.left_train_right_test = []
        self.left_test_right_train = []
        self.left_test_right_test = []
        for left, right in self.graph.edges:
            if (left in self.train_graph and right in self.train_graph) and not self.train_graph.has_edge(left, right):
                #depth_diff = nx.shortest_path_length(self.original_graph, source=left, target=right)
                depth_diff = nx.shortest_path_length(self.original_graph, source=root, target=left)
                self.left_train_right_train.append(depth_diff)
            if (left in self.train_graph and right not in self.train_graph):
                #depth_diff = nx.shortest_path_length(self.original_graph, source=left, target=right)
                depth_diff = nx.shortest_path_length(self.original_graph, source=root, target=left)
                self.left_train_right_test.append(depth_diff)
            if (left not in self.train_graph and right in self.train_graph):
                #depth_diff = nx.shortest_path_length(self.original_graph, source=left, target=right)
                depth_diff = nx.shortest_path_length(self.original_graph, source=root, target=left)
                self.left_test_right_train.append(depth_diff)
            if (left not in self.train_graph and right not in self.train_graph):
                #depth_diff = nx.shortest_path_length(self.original_graph, source=left, target=right)
                depth_diff = nx.shortest_path_length(self.original_graph, source=root, target=left)
                self.left_test_right_test.append(depth_diff)
 
        print(f"full graph num of edges: {self.graph.number_of_edges()}; sparsity: {self.graph.number_of_edges() / self.size:.3f}; during training, edge coverage {self.train_graph.number_of_edges() / self.graph.number_of_edges():.3f}, node coverage {self.train_graph.number_of_nodes() / self.graph.number_of_nodes():.3f}")
        print(f"number of testing edges (left known right known): {len(self.left_train_right_train)}, diff: {np.mean(self.left_train_right_train)}")
        print(f"number of testing edges (left known right unknown): {len(self.left_train_right_test)}, diff: {np.mean(self.left_train_right_test)}")
        print(f"number of testing edges (left unknown right known): {len(self.left_test_right_train)}, diff: {np.mean(self.left_test_right_train)}")
        print(f"number of testing edges (left unknown right unknown): {len(self.left_test_right_test)}, diff: {np.mean(self.left_test_right_test)}")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        i, j = idx // (self.graph.number_of_nodes() - 1), idx % (self.graph.number_of_nodes() - 1)
        if j == i: 
            j = i + 1
        left = int(self.nodes[i]) 
        right = int(self.nodes[j]) 
        label = int(self.graph.has_edge(left, right))
        train_label = int(self.train_graph.has_edge(left, right))
        in_train = False
        if self.train_graph.has_node(left) and self.train_graph.has_node(right):
            in_train = True
        return left, right, self.embeddings[left], self.embeddings[right], train_label, label, in_train
