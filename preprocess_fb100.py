import os
import networkx as nx
import numpy as np
import scipy.io
from collections import Counter

def load_facebook100_data(
    mat_filename="American75.mat",
    folder_name="facebook100",
    max_nodes=250,
    attribute_type="gender",
    granularity=2  # Number of bins for multivalued categories (2V, 3V, 4V)
):
    """
    Loads FB100 network and bins attributes according to paper setups.
    
    Attribute Types:      
      - 'status': [Undergrad = 1, Other = 2]
      - 'gender': [Female = 1, Male = 2, Unknown = 0]
      - 'major': Multivalued (Top-K majors vs Others)
      - 'second_major': Multivalued (Top-K second majors vs Others)
      - 'dorm': Multivalued (Top-K dorms vs Others)
      - 'year': [2006, 2007, 2008, 2009]
      - 'high_school': Multivalued (Top-K high schools vs Others)
      - 'multidim_gender_year': 2D Attribute (Gender x Year)
      
    """
    file_path = os.path.join(folder_name, mat_filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}")

    #Using scipy library to load the .mat file
    #Reads a MATLAB (.mat) file and returns a dictionary containing the variables in the file.
    mat_data = scipy.io.loadmat(file_path)

    adj_matrix = mat_data["A"]          # Adjacency matrix of the network
    local_info = mat_data["local_info"] # Node attributes

    # Create a NetworkX graph G from the adjacency matrix
    G = nx.from_scipy_sparse_array(adj_matrix)
    '''
        Adjacency matrix:
                0  1  2
            0   0  1  0
            1   1  0  1
            2   0  1  0
        NetworkX graph G:
            G.nodes() = [0, 1, 2]
            G.edges() = [(0, 1), (1, 2)]
    '''

    # 1. Filter out nodes with completely missing values (0 is missing in FB100)
    valid_nodes = []
    for node in G.nodes():
        # Check validity based on chosen attribute
        if attribute_type == "status" and local_info[node, 0] > 0:
            valid_nodes.append(node)
        elif attribute_type == "gender" and local_info[node, 1] in [1, 2]:
                    valid_nodes.append(node)
        elif attribute_type == "major" and local_info[node, 2] > 0:
            valid_nodes.append(node)
        elif attribute_type == "dorm" and local_info[node, 4] > 0:
                    valid_nodes.append(node)
        elif attribute_type == "year" and local_info[node, 5] in [2006, 2007, 2008, 2009]:
            valid_nodes.append(node)        
        elif attribute_type.startswith("multidim"):
            if local_info[node, 1] in [1, 2] and local_info[node, 5] in [2006, 2007, 2008, 2009]:
                valid_nodes.append(node) 

    # 2. Extract dense subgraph based on node degree (select the top 250 nodes with max degree)
    sub_degree = [(n, G.degree[n]) for n in valid_nodes]
    sorted_nodes = sorted(sub_degree, key=lambda x: x[1], reverse=True)
    selected_indices = [n for n, deg in sorted_nodes[:max_nodes]]


    # Smaller graph with selected node is created
    # Node : [6, 1076, 1547] -> ["v6", "v1076", "v1547"]
    # Edge : [(6, 1076)] -> [("v6", "v1076")]
    subG = G.subgraph(selected_indices).copy()
    nodes = [f"v{i}" for i in subG.nodes()]
    edges = [(f"v{u}", f"v{v}") for u, v in subG.edges()]


    # 3. Binning & Mapping logic (Table III & Table VI in paper)
    raw_vals = {}
    for node_idx in subG.nodes():
        node_str = f"v{node_idx}"
        
        # -------------------------------------------------------------
        # Category 0: Student Status (Undergrad = 1, Other = 2)
        # -------------------------------------------------------------
        if attribute_type == "status":
            val = local_info[node_idx, 0]
            node_attributes[node_str] = "Undergrad" if val == 1 else "Other_Status"

        # -------------------------------------------------------------
        # Category 1: Gender (Female = 1, Male = 2)
        # -------------------------------------------------------------
        elif attribute_type == "gender":
            val = local_info[node_idx, 1]
            raw_vals[node_str] = "Female" if val == 1 else "Male"

        # -------------------------------------------------------------
        # Category 2: Major / Field (Top Major vs Others - 2V / 3V / 4V)
        # -------------------------------------------------------------
        elif attribute_type == "major":
            val = int(local_info[node_idx, 2])
            raw_vals[node_str] = val

        # -------------------------------------------------------------
        # Category 5: Class Year (Upperclassmen vs Underclassmen)
        # -------------------------------------------------------------
        elif attribute_type == "year":
            val = int(local_info[node_idx, 5])
            if granularity == 2:
                # Binary: Fresh/Soph (2008/2009) vs Junior/Senior (2006/2007)
                raw_vals[node_str] = "Junior_Senior" if val in [2006, 2007] else "Fresh_Soph"
            else:
                # 4-Valued: 2006, 2007, 2008, 2009
                raw_vals[node_str] = f"Class_{val}"

        # -------------------------------------------------------------
        # Category 4: Dorm / Residence
        # -------------------------------------------------------------
        elif attribute_type == "dorm":
            raw_vals[node_str] = int(local_info[node_idx, 4])

        # -------------------------------------------------------------
        # Multi-Dimensional: Gender x Class Year (2D fairness)
        # -------------------------------------------------------------
        elif attribute_type == "multidim_gender_year":
            g = "F" if local_info[node_idx, 1] == 1 else "M"
            y = "Upper" if local_info[node_idx, 5] in [2006, 2007] else "Lower"
            raw_vals[node_str] = f"{g}_{y}"  # Forms 4 compound classes: F_Upper, F_Lower, M_Upper, M_Lower

    # Dynamic frequency binning for high-cardinality fields (Major / Dorm)
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

    return nodes, edges, node_attributes