from collections import defaultdict
from AFCMiner import FairnessFilter, AttributedConceptsDerivation

def BronKerboschIterative(V_set, adj):
    maximal_cliques = []
    stack = [(set(), set(V_set), set())]

    while stack:
        R, P, X = stack.pop()

        if not P and not X:
            if R:
                maximal_cliques.append(R)
            continue

        if not P:
            continue

        pivot = max(P | X, key=lambda u: len(P & adj[u]))
        candidates = list(P - adj[pivot])

        for v in candidates:
            stack.append((R | {v}, P & adj[v], X & adj[v]))
            P.remove(v)
            X.add(v)

    return maximal_cliques


def BKMiner(V, node_attribute_set, R_input):
    res = []
    V_set = set(V)
    attributes = set(node_attribute_set) - V_set
    
    Matrix = defaultdict(lambda: defaultdict(int))
    adj = defaultdict(set)
    
    # 1. Self-loops for all nodes
    for v in V_set:
        Matrix[v][v] = 1
        
    # 2. Build graph edges and attribute mappings
    for i, j in R_input:
        Matrix[i][j] = 1
        if i in V_set and j in V_set:
            Matrix[j][i] = 1
            adj[i].add(j)
            adj[j].add(i)
            
    # 3. Discover maximal cliques
    maximal_cliques = BronKerboschIterative(V_set, adj)
    print(len(maximal_cliques))
    for clique in maximal_cliques:
        if FairnessFilter(clique, clique, attributes, Matrix):
            res.append(clique)
        else:
            powerset = AttributedConceptsDerivation(clique)
            powerset.sort(key=lambda x: len(x), reverse=True)
            
            cur_maxi = []
            for sub in powerset:
                if not sub:
                    continue
                if any((cur & sub) == sub for cur in cur_maxi):
                    continue
                if FairnessFilter(sub, sub, attributes, Matrix):
                    cur_maxi.append(sub)
                    res.append(sub)
    print(len(res))               
    return res