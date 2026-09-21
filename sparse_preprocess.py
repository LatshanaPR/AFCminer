import os
import networkx as nx
import numpy as np
import scipy.io
from collections import Counter, deque

def load_facebook100_data(
    mat_filename="American75.mat",
    folder_name="facebook100",
    max_nodes=250,
    attribute_type="multidim_gender_year",
    granularity=2  # Number of bins for multivalued categories (2V, 3V, 4V)
):
    """
    Loads FB100 network and bins attributes according to paper setups.
    Uses BFS neighborhood sampling to extract sparse subgraphs.
    """
    file_path = os.path.join(folder_name, mat_filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}")

    # Load the .mat file
    mat_data = scipy.io.loadmat(file_path)
    adj_matrix = mat_data["A"]          # Adjacency matrix
    local_info = mat_data["local_info"] # Node attributes

    # Create NetworkX graph G from the adjacency matrix
    G = nx.from_scipy_sparse_array(adj_matrix)

    # 1. Filter out nodes with missing values based on attribute choice
    valid_nodes = set()
    for node in G.nodes():
        if attribute_type == "status" and local_info[node, 0] > 0:
            valid_nodes.add(node)
        elif attribute_type == "gender" and local_info[node, 1] in [1, 2]:
            valid_nodes.add(node)
        elif attribute_type == "major" and local_info[node, 2] > 0:
            valid_nodes.add(node)
        elif attribute_type == "dorm" and local_info[node, 4] > 0:
            valid_nodes.add(node)
        elif attribute_type == "year" and local_info[node, 5] in [2006, 2007, 2008, 2009]:
            valid_nodes.add(node)        
        elif attribute_type.startswith("multidim"):
            if local_info[node, 1] in [1, 2] and local_info[node, 5] in [2006, 2007, 2008, 2009]:
                valid_nodes.add(node) 

    # Create induced valid subgraph to perform accurate BFS search
    valid_subG = G.subgraph(valid_nodes)

    # 2. Extract sparse subgraph using BFS Neighborhood Sampling
    # Find a seed node with a modest, natural degree (between 5 and 15)
    candidate_seeds = [n for n in valid_subG.nodes() if 5 <= valid_subG.degree[n] <= 15]
    
    if candidate_seeds:
        seed = candidate_seeds[0]
    else:
        # Fallback if no node fits the exact degree range
        seed = list(valid_nodes)[0]

    # BFS Traversal to collect exactly max_nodes
    selected_indices = []
    visited = {seed}
    queue = deque([seed])

    while queue and len(selected_indices) < max_nodes:
        curr = queue.popleft()
        selected_indices.append(curr)

        for nbr in valid_subG.neighbors(curr):
            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)
                if len(selected_indices) + len(queue) >= max_nodes:
                    break

    # Fill up to max_nodes if queue empties before reaching target limit
    if len(selected_indices) < max_nodes:
        remaining = list(valid_nodes - set(selected_indices))
        selected_indices.extend(remaining[:max_nodes - len(selected_indices)])

    # Construct final sparse subgraph
    subG = G.subgraph(selected_indices).copy()
    nodes = [f"v{i}" for i in subG.nodes()]
    edges = [(f"v{u}", f"v{v}") for u, v in subG.edges()]

    # 3. Convert raw attribute values
    raw_vals = {}
    for node_idx in subG.nodes():
        node_str = f"v{node_idx}"
        
        if attribute_type == "status":
            val = local_info[node_idx, 0]
            raw_vals[node_str] = "undergrad" if val == 1 else "other_status"

        elif attribute_type == "gender":
            val = local_info[node_idx, 1]
            raw_vals[node_str] = "female" if val == 1 else "male"

        elif attribute_type == "major":
            val = int(local_info[node_idx, 2])
            raw_vals[node_str] = val

        elif attribute_type == "year":
            val = int(local_info[node_idx, 5])
            if granularity == 2:
                raw_vals[node_str] = "junior_senior" if val in [2006, 2007] else "fresh_soph"
            else:
                raw_vals[node_str] = val

        elif attribute_type == "dorm":
            raw_vals[node_str] = int(local_info[node_idx, 4])

        elif attribute_type == "multidim_gender_year":
            gender = "female" if local_info[node_idx, 1] == 1 else "male"
            year = int(local_info[node_idx, 5])
            raw_vals[node_str] = (gender, year)

    # Dynamic frequency binning for Major / Dorm
    if attribute_type in ["major", "dorm"]:
        counts = Counter(raw_vals.values())
        top_k = [item[0] for item in counts.most_common(granularity - 1)]
        
        node_attributes = {}
        for node_str, v in raw_vals.items():
            if v in top_k:
                node_attributes[node_str] = f"{attribute_type.capitalize()}_{v}"
            else:
                node_attributes[node_str] = "Others"
    else:
        node_attributes = raw_vals

    # 4. Build attribute column list
    if attribute_type == "multidim_gender_year":
        years = [2006, 2007, 2008, 2009]
        genders = ["male", "female"]
        attribute_values = [
            (gender, year)
            for year in years
            for gender in genders
        ]
    else:
        attribute_values = list(dict.fromkeys(node_attributes.values()))
    attribute_columns = nodes + attribute_values

    # 5. Combine edges and attributes
    combined_data = edges + list(node_attributes.items())

    return nodes, attribute_columns, combined_data


def split_preprocessed_data(nodes, combined_data):
    """Separate graph edges and node attributes from the combined output."""
    edges = []
    node_attributes = {}

    for record in combined_data:
        first, second = record
        if first in nodes and isinstance(second, str) and second in nodes:
            edges.append((first, second))
        else:
            node_attributes[first] = second

    return edges, node_attributes
