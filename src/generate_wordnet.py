import json
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from collections import deque
import sys
import re
import numpy as np
import networkx as nx

from sentence_transformers import SentenceTransformer

class SentEmbedding:
    def __init__(self):
        self.model = SentenceTransformer('hkunlp/instructor-xl')
        self.instruction1 = "Represent the word: "
        self.instruction2 = "Represent the sentence: "

    def generate(self, texts):
        inputs1 = []
        inputs2 = []
        for word, definition in texts:
            inputs1.append([self.instruction1, word, 0])
            inputs2.append([self.instruction2, f"{word}: {definition}", 0])
        embeddings1 = self.model.encode(inputs1)
        embeddings2 = self.model.encode(inputs2)
        return np.concatenate([embeddings1, embeddings2], -1) # batchsize, 768 x 2

root = "entity.n.01" 

domain_list = ["person.n.01", "animal.n.01", "mammal.n.01", "body_part.n.01", "location.n.01", "commodity.n.01", "disease.n.01"]

# breath first search

root_wn = wn.synset(root)

def bfs(root, syn2info=None):
    # wordnet is a rooted DAG
    if syn2info == None:
        syn2info = {}
    graph = []

    visited = set() # avoid duplicated counting
    buffer_list = deque([root])
    if root not in syn2info:
        syn2info[root] = {"id": len(syn2info), "synset": root, "name": root.lemma_names()[0], "definition": root.definition()}
    visited.add(root)

    while len(buffer_list) > 0:
        parent = buffer_list.popleft()
        for child in parent.hyponyms():
            if child not in syn2info:
                syn2info[child] = {"id": len(syn2info), "synset": child, "name": child.lemma_names()[0], "definition": child.definition()}
            pid = syn2info[parent]["id"]
            cid = syn2info[child]["id"]
            graph.append((pid, cid))
            if child not in visited:
                buffer_list.append(child)
                visited.add(child)

    
    return graph, syn2info

def output(root, graph):
    open(f"data/edges.{root}.json","w").write(json.dumps(graph, indent="\t"))
    graph_nx = nx.DiGraph(graph)
    print(f"Dag rooted with {root}, number of nodes = {graph_nx.number_of_nodes()}, number of edges = {graph_nx.number_of_edges()}, depth of the longest path = {len(nx.dag_longest_path(graph_nx))}")




root_wn = wn.synset(root)
_, syn2info = bfs(root_wn)
id2info = dict([(v["id"], {"synset": f"{v['synset']}", "name": v["name"], "def": re.sub(r'[^\w\s]', '', v["definition"])}) for v in syn2info.values()])
open(f"data/id2info.json","w").write(json.dumps(id2info, indent="\t"))

'''
word2ids = {}
for i, syn in id2info.items():
    name = syn["name"].split("_")
    for token in name:
        if token not in word2ids:
            word2ids[token] = []
        word2ids[token].append(i)

embeds = np.zeros((len(id2info), 300))
counts = np.zeros(len(id2info))
for v in open('glove.840B.300d.txt'):
    v = v.strip().split()
    token, vec = v[0], np.array([float(s) for s in v[-300:]])
    if token in word2ids:
        for i in word2ids[token]:
            embeds[i] += vec
            counts[i] += 1.0
for i in range(len(embeds)):
    if counts[i] > 0:
        embeds[i] = embeds[i] / counts[i]
cover_ratio = np.array(counts > 0, dtype=np.float32).mean()
np.save(f"embedding.300d.npy", embeds)
print(f"{cover_ratio} of nodes found embeddings")
for i in list(np.argwhere(embeds.sum(-1) == 0)):
    print(id2info[int(i)]["name"])
'''
#'''
model = SentEmbedding()
embeds = np.zeros((len(id2info), 768*2))
for i in tqdm(id2info.keys()):
    name = id2info[i]["name"].replace("_", " ")
    definition = id2info[i]["def"]
    embed_ = model.generate([[name, definition]])
    embeds[i] = embed_[0]
np.save(f"data/embedding.instructor.1536d.npy", embeds)
#'''
for r in domain_list:
    r_wn = wn.synset(r)
    g_r, _ = bfs(r_wn, syn2info)
    output(r, g_r)
