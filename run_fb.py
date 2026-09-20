'''
FILE NOT REQUIRED
This file is not required for the experiments. It is a standalone validation run that
loads the first 250 nodes of the American75.mat dataset, 
prints the attribute distribution, and runs the optimal AFCMiner algorithm to find 
absolute fair maximal cliques (AFMCs) and absolute fair cliques (AFCs). 
The results are printed to the console.
'''

from preprocess_fb100 import load_facebook100_data, split_preprocessed_data
from optimal_afc import afc_miner_optimal

# This is a single direct 250-node validation run.
# No nested subset construction here: we want to check the first subset alone.
nodes, _, combined_data = load_facebook100_data(
    mat_filename="American75.mat",
    folder_name="facebook100",
    max_nodes=250,
    attribute_type="gender",
)
edges, node_attributes = split_preprocessed_data(nodes, combined_data)

print(
    f"Successfully Loaded: {len(nodes)} Students, {len(edges)} Friendship"
    " Edges"
)
unique_attributes = sorted(set(node_attributes.values()))
print(f"Unique attribute values ({len(unique_attributes)}): {unique_attributes}")
print(f"Attribute value counts: { {value: list(node_attributes.values()).count(value) for value in unique_attributes} }\n")

afmc, afc = afc_miner_optimal(nodes, edges, node_attributes)

print(f"Absolute Fair Maximal Cliques Found: {len(afmc)}")
maxi = float('-inf')
mini = float('inf')
for c in list(afmc)[:5]:
    print("  AFMC:", c)
for c in list(afmc):
    maxi = max(maxi, len(c))
    mini = min(mini, len(c))
print("Largest Fair Clique Size:", maxi)
print("Minimum Fair cliqur Size:", mini)
print(f"\nTotal Absolute Fair Cliques (including sub-cliques): {len(afc)}")
