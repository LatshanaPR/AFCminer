import numpy as np

def ConceptBuilder(Matrix, V, node_attribute_set):
    """
    High-Performance NumPy-Accelerated FCA (Close-By-One / Open-Close).
    Matches exact input types (dict-of-dicts Matrix, V, node_attribute_set)
    and exact output type: set of (extent_frozenset, intent_frozenset, attribute_frozenset).
    """
    # 1. Map nodes and attributes to contiguous integer indices
    nodes_list = sorted(list(V))
    N = len(nodes_list)
    node_to_idx = {node: i for i, node in enumerate(nodes_list)}
    idx_to_node = {i: node for i, node in enumerate(nodes_list)}

    attributes_list = sorted(list(set(node_attribute_set) - set(V)))
    num_attrs = len(attributes_list)
    attr_to_idx = {attr: i for i, attr in enumerate(attributes_list)}
    idx_to_attr = {i: attr for i, attr in enumerate(attributes_list)}

    # 2. Build NumPy Adjacency Bitmasks
    # Determine uint size needed based on N (32-bit, 64-bit, or array of uint64 blocks)
    # Using python objects inside numpy or fixed uint64 arrays for bit manipulation:
    neighbors_list = [0] * N
    for u in nodes_list:
        u_idx = node_to_idx[u]
        u_dict = Matrix.get(u, {})
        mask = 0
        for v, val in u_dict.items():
            if val == 1 and v in node_to_idx:
                mask |= (1 << node_to_idx[v])
        neighbors_list[u_idx] = mask

    # Convert node neighbor bitmasks to a NumPy vector (dtype=object for arbitrary bit length > 64)
    NEIGHBORS = np.array(neighbors_list, dtype=object)

    # Build Attribute Bitmask NumPy Vector (each entry represents nodes possessing attribute `a`)
    attr_neighbors_list = [0] * num_attrs
    for u in nodes_list:
        u_idx = node_to_idx[u]
        u_dict = Matrix.get(u, {})
        for a_idx, attr in enumerate(attributes_list):
            if u_dict.get(attr, 0) == 1:
                attr_neighbors_list[a_idx] |= (1 << u_idx)

    ATTR_NEIGHBORS = np.array(attr_neighbors_list, dtype=object)

    ALL_NODES_MASK = (1 << N) - 1

    # Pre-build bitmask pruning lookup filters using NumPy
    # PRUNE_MASKS[j] is a mask with bits 0 to j-1 set to 1: (1 << j) - 1
    PRUNE_MASKS = np.array([(1 << j) - 1 for j in range(N)], dtype=object)

    # Fast NumPy Vectorized Intent Derivation
    def derive_intent_np(extent_mask):
        if not extent_mask:
            return ALL_NODES_MASK
        
        # Identify active node indices in extent via bit shifting
        # Vectorized check: filter neighbors where the bit corresponding to extent is set
        active_indices = [x for x in range(extent_mask.bit_length()) if (extent_mask >> x) & 1]
        if not active_indices:
            return ALL_NODES_MASK
            
        # Vectorized NumPy bitwise AND reduction across active neighbor bitmasks
        return int(np.bitwise_and.reduce(NEIGHBORS[active_indices]))

    # Storage for output bitmask concepts
    raw_concepts = []

    # 3. Recursive Open-Close DFS Traversal
    def open_close_dfs_np(r, extent, intent):
        raw_concepts.append((extent, intent))

        for j in range(r, N):
            # If element j is already in intent, skip branch
            if (intent >> j) & 1:
                continue

            # Compute candidate extent via bitwise AND with neighbor j
            new_extent = extent & NEIGHBORS[j]
            if not new_extent:
                continue

            # Compute Galois Intent Closure
            new_intent = derive_intent_np(new_extent)

            # CANONICAL TEST (Open-Close Optimization):
            # Evaluate newly added bits against PRUNE_MASKS[j]
            added_bits = new_intent & ~intent
            if added_bits & PRUNE_MASKS[j]:
                continue  # Redundant branch pruned!

            open_close_dfs_np(j + 1, new_extent, new_intent)

    # Compute initial concept
    initial_extent = ALL_NODES_MASK
    initial_intent = derive_intent_np(initial_extent)

    # Run CbO/Open-Close Traversal
    open_close_dfs_np(0, initial_extent, initial_intent)

    # 4. Decode Concepts back to Frozensets
    final_concepts = set()

    for extent_mask, intent_mask in raw_concepts:
        # Reconstruct node frozensets
        extent_set = frozenset(
            idx_to_node[i] for i in range(extent_mask.bit_length()) if (extent_mask >> i) & 1
        )
        intent_set = frozenset(
            idx_to_node[i] for i in range(intent_mask.bit_length()) if (intent_mask >> i) & 1
        )

        # Vectorized Attribute Evaluation using NumPy
        attr_set = []
        if extent_mask and num_attrs > 0:
            # Check which attribute masks contain the full extent mask
            # (extent_mask & ATTR_NEIGHBORS) == extent_mask
            valid_attr_mask = (ATTR_NEIGHBORS & extent_mask) == extent_mask
            valid_attr_indices = np.where(valid_attr_mask)[0]
            attr_set = [idx_to_attr[a_idx] for a_idx in valid_attr_indices]

        final_concepts.add((extent_set, intent_set, frozenset(attr_set)))

    return final_concepts