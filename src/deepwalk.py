import time
import random
import numpy as np
from gensim.models import Word2Vec


random.seed(2018)
np.random.seed(2018)


def gen_deep_walk_feature(A, number_walks=10, alpha=0, walk_length=80, window=10, workers=2, size=128):#,row, col,
    row,col = A.nonzero()
    edges = np.concatenate((row.reshape(-1, 1), col.reshape(-1, 1)), axis=1).astype(dtype=np.dtype(str))
    print("build adj_mat")
    t1 = time.time()
    G = {}
    for [i, j] in edges:
        if i not in G:
            G[i] = []
        if j not in G:
            G[j] = []
        G[i].append(j)
        G[j].append(i)
    for node in G:
        G[node] = list(sorted(set(G[node])))
        if node in G[node]:
            G[node].remove(node)

    nodes = list(sorted(G.keys()))
    print("len(G.keys()):", len(G.keys()), "\tnode_num:", A.shape[0])
    corpus = [] 
    for cnt in range(number_walks):
        random.shuffle(nodes)
        for idx, node in enumerate(nodes):
            path = [node]  
            while len(path) < walk_length:
                cur = path[-1]  
                if len(G[cur]) > 0:
                    if random.random() >= alpha:
                        path.append(random.choice(G[cur])) 
                    else:
                        path.append(path[0]) 
                else:
                    break
            corpus.append(path)
    t2 = time.time()
    print("cost: {}s".format(t2 - t1))
    print("train...")
    model = Word2Vec(corpus,
                     size=size,  # emb_size
                     window=window,
                     min_count=0,
                     sg=1,  # skip gram
                     hs=1,  # hierarchical softmax
                     workers=workers)
    print("done.., cost: {}s".format(time.time() - t2))
    output = []
    for i in range(A.shape[0]):
        if str(i) in model.wv:  # word2vec
            output.append(model.wv[str(i)])
        else:
            print("{} not trained".format(i))
            output.append(np.zeros(size))
    return output
