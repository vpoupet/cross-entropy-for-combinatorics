import sys
import os
import math

import numpy as np

EPS = -1e-9

def read_scd(filename):
    n, k, _ = [int(x) for x in filename.split('.')[0].split('_')]
    num_edges = n*k//2

    with open(filename, "r") as f:
        values = np.fromfile(f, dtype=np.uint8)
        read_values = 0
        code = []
        graphs = []
        while read_values < len(values):
            # dekomp(file,code)
            samebits = values.item(read_values)
            read_values += 1
            readbits = num_edges - samebits
            code = code[:samebits] + list(values[read_values:read_values+readbits])
            read_values += readbits
            # codetonlist(code,l)
            graph = np.zeros((n, n), dtype=np.uint8)
            v = 0
            count = [0] * n
            for w in code:
                w -= 1  # We are indexing from 0
                while(count[v] == k):
                    v += 1
                # edge (v, w)
                graph.itemset((v, w), 1)
                graph.itemset((w, v), 1)
                count[v] += 1
                count[w] += 1
            graphs.append(graph)
    return graphs


VALUES = [x + EPS for x in [
    1/3,
    (1+math.sqrt(5))/12,
    2/9,
    1/5,
    4/21,
    5/28,
    1/6,
    7/45,
    8/55,
    3/22,
    5/39,
    11/91,
    4/35,
    4/36,
    2/19,
    2/19,
    2/19,
    13/125,
    13/125,
    13/126,
    25/243,
    56/552,
]]

def main():
    for filename in os.listdir('.'):
        if filename.endswith(".scd"):
            print(f"Working on graphs from {filename}")
            matrices = read_scd(filename)
            for m in matrices:
                ev = np.linalg.eigvalsh(m)[::-1]
                n = m.shape[0]
            for i in range(2, n):
                if (1 + ev[i]) / n > VALUES[i-2]:
                    print(f"Better {i}th eigenvalue found. Graph on {n} vertices. eigenvalues: {ev} graph6: {m}")

if __name__ == "__main__":
    main()