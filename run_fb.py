# run_fb_experiment.py
from main import afc_miner
from optimal import afc_miner_optimal
from preprocess_fb100 import load_facebook100_data

# 1. Preprocess Facebook100 network (e.g., Caltech36 or Reed98)
nodes, edges, node_attributes = load_facebook100_data(
    mat_filename="American75.mat",  # Smallest campus network in FB100
    folder_name="facebook100",
    max_nodes=250,  # Scale subsets: 250, 500, 750 (like the paper)
    attribute_type="gender",
)
print(nodes)
print(edges)
print(
    f"Successfully Loaded: {len(nodes)} Students, {len(edges)} Friendship"
    " Edges"
)
print(f"Attribute Sample: {list(node_attributes.items())[:5]}\n")

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