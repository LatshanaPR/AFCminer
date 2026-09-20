import itertools

# =========================================================================
# 1. OPTIMIZED FCA EQUICONCEPT MINER (Fast In-Close / Bit-Parallel Lattice)
# =========================================================================
def extract_equiconcepts_fast_fca(nodes, edges):
    """
    Fast Bit-Parallel FCA Equiconcept Extraction.
    Uses degeneracy ordering and pivot-guided concept branch-and-bound.
    """
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for i, node in enumerate(nodes)}

    # Build bitmasks for adjacency (no self-loops in neighbors)
    adj_bits = [0] * n
    degree = [0] * n
    for u, v in edges:
        if u in node_to_idx and v in node_to_idx:
            u_i, v_i = node_to_idx[u], node_to_idx[v]
            adj_bits[u_i] |= (1 << v_i)
            adj_bits[v_i] |= (1 << u_i)
            degree[u_i] += 1
            degree[v_i] += 1

    # Degeneracy ordering (process lower degree nodes first for efficient pruning)
    order = sorted(range(n), key=lambda x: degree[x])
    pos_in_order = {node_idx: pos for pos, node_idx in enumerate(order)}

    equiconcepts = []

    def in_close_explore(R_mask, P_mask, X_mask):
        if not P_mask and not X_mask:
            # Equiconcept found: Extent == Intent (Maximal Clique)
            clique = [idx_to_node[i] for i in range(n) if (R_mask >> i) & 1]
            equiconcepts.append(clique)
            return

        # Choose pivot with maximum connections to candidates in P
        PX = P_mask | X_mask
        pivot = max(range(n), key=lambda u: bin(P_mask & adj_bits[u]).count("1") if ((PX >> u) & 1) else -1)
        candidates_mask = P_mask & ~adj_bits[pivot]

        while candidates_mask:
            # Extract lowest bit
            v = (candidates_mask & -candidates_mask).bit_length() - 1
            v_bit = 1 << v
            
            # Recursive closure branch
            in_close_explore(
                R_mask | v_bit,
                P_mask & adj_bits[v],
                X_mask & adj_bits[v]
            )

            # Move candidate from P to X
            P_mask &= ~v_bit
            X_mask |= v_bit
            candidates_mask &= ~v_bit

    # Run top-level search
    P_all = (1 << n) - 1
    in_close_explore(0, P_all, 0)
    return equiconcepts


# =========================================================================
# 2. DIRECT CARTESIAN FAIR SUB-CONCEPT DERIVATION
# =========================================================================
def derive_fair_subcliques_direct(maximal_clique, node_attributes, a_val):
    """
    Directly forms balanced partitions without computing full power sets 2^k.
    Generates all combinations of size m from each attribute bucket simultaneously.
    """
    # Bucket nodes by their attribute values
    buckets = {attr: [] for attr in a_val}
    for node in maximal_clique:
        buckets[node_attributes[node]].append(node)

    # Maximum possible equal count per attribute
    min_bucket_size = min(len(buckets[attr]) for attr in a_val)
    if min_bucket_size == 0:
        return []

    fair_subcliques = []

    # Direct combination generation for each fair size m
    for m in range(1, min_bucket_size + 1):
        # Generate combinations of size m for each attribute bucket
        attr_combinations = [list(itertools.combinations(buckets[attr], m)) for attr in a_val]
        
        # Cartesian product across attribute combinations gives valid fair cliques
        for prod in itertools.product(*attr_combinations):
            combined_clique = tuple(sorted(itertools.chain.from_iterable(prod)))
            fair_subcliques.append(combined_clique)

    return fair_subcliques


# =========================================================================
# 3. HIGH-PERFORMANCE AFCMINER PIPELINE (Algorithm 1)
# =========================================================================
def afc_miner_optimal(nodes, edges, node_attributes, include_afc=True):
    """
    Complete high-performance implementation of AFCMiner.
    """
    a_val = sorted(list(set(node_attributes.values())))
    num_attributes = len(a_val)

    # 1. Fast Bit-Parallel Equiconcept Extraction
    equiconcepts = extract_equiconcepts_fast_fca(nodes, edges)

    zeta_afmc = set()
    zeta_afc = set()

    for clique_nodes in equiconcepts:
        k = len(clique_nodes)
        
        # Check if the maximal equiconcept itself is an AFMC
        is_afmc = False
        if k % num_attributes == 0 and k > 0:
            counts = {attr: 0 for attr in a_val}
            for u in clique_nodes:
                counts[node_attributes[u]] += 1
            if len(set(counts.values())) == 1:
                clique_tuple = tuple(sorted(clique_nodes))
                zeta_afmc.add(clique_tuple)
                zeta_afc.add(clique_tuple)
                is_afmc = True

        # Derive sub-cliques only when the caller requests the complete AFC
        # result. Table IV requests AFMC counts only and skips this step.
        if include_afc:
            derived_fair = derive_fair_subcliques_direct(
                clique_nodes,
                node_attributes,
                a_val,
            )
            for fair_c in derived_fair:
                zeta_afc.add(fair_c)

    return zeta_afmc, zeta_afc