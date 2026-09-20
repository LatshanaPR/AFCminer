from backup_afc import afc_miner

# Optimal afc logic from paper
from optimal_afc import afc_miner_optimal
from preprocess_fb100 import load_facebook100_data

# 1. Preprocess Facebook100 network (American75) and store the nodes, edges, and node attributes
nodes, edges, node_attributes = load_facebook100_data(
    mat_filename="American75.mat",  # One of the campus network in FB100
    folder_name="facebook100",
    max_nodes=250,  # Scale subsets: 250, 500, 750 (like the paper)
    attribute_type="major",
)

#Print the number of nodes and edges in the preprocessed dataset
print(
    f"Successfully Loaded: {len(nodes)} Students, {len(edges)} Friendship"
    " Edges"
)
# Print the unique attribute values for the selected nodes and their counts
unique_attributes = sorted(set(node_attributes.values()))
print(f"Unique attribute values ({len(unique_attributes)}): {unique_attributes}")
print(f"Attribute value counts: { {value: list(node_attributes.values()).count(value) for value in unique_attributes} }\n")

# 2. Run AFCMiner Algorithm
afmc, afc = afc_miner_optimal(nodes, edges, node_attributes)

# 3. View Discovered Absolute Fair Cliques
print(f"Absolute Fair Maximal Cliques Found: {len(afmc)}")
maxi=float('-inf')
mini=float('inf')
for c in list(afmc)[:5]:
    print("  AFMC:", c)
for c in list(afmc):
    maxi=max(maxi,len(c))
    mini=min(mini,len(c))
print("Largest Fair Clique Size:",maxi)
print("Minimum Fair cliqur Size:",mini)
print(f"\nTotal Absolute Fair Cliques (including sub-cliques): {len(afc)}")