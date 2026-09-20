import itertools
from backup_afc import fairness_filter, attributed_concepts_derivation, build_formal_context

# ========================================================
# BRON-KERBOSCH WITH PIVOTING (Algorithm 4 Baseline)
# ========================================================
def bron_kerbosch_pivot(R, P, X, adj, maximal_cliques):
    """
    Standard recursive Bron-Kerbosch algorithm with pivoting 
    to enumerate all maximal cliques in the graph.
    """
    if not P and not X:
        maximal_cliques.append(R)
        return

    # Choose pivot u in P union X to maximize |P ∩ N(u)|
    u = next(iter(P | X))
    
    # Iterate over candidates not connected to the pivot
    for v in list(P - adj[u]):
        bron_kerbosch_pivot(
            R | {v},
            P & adj[v],
            X & adj[v],
            adj,
            maximal_cliques
        )
        P.remove(v)
        X.add(v)


def bk_afc_miner(nodes, edges, node_attributes):
    """
    Algorithm 4: BK Algorithm for Mining Absolute Fair Cliques
    Serves as the ground-truth baseline to verify AFCMiner.
    """
    zeta_afmc = set()
    zeta_afc = set()

    # Build adjacency dictionary
    adj, node_attributes, a_val = build_formal_context(nodes, edges, node_attributes)
    
    # Adjacency for BK must NOT include self-loops in neighbor lookups
    adj_no_self = {v: adj[v] - {v} for v in nodes}

    # Step 1-10: Find all maximal cliques using Bron-Kerbosch
    maximal_cliques = []
    bron_kerbosch_pivot(set(), set(nodes), set(), adj_no_self, maximal_cliques)

    # Step 12-18: Filter fairness and derive sub-cliques
    for c in maximal_cliques:
        if fairness_filter(c, node_attributes, a_val):
            clique_frozen = tuple(sorted(list(c)))
            zeta_afmc.add(clique_frozen)
            zeta_afc.add(clique_frozen)
        else:
            sub_cliques = attributed_concepts_derivation(c)
            for sub_c in sub_cliques:
                if fairness_filter(sub_c, node_attributes, a_val):
                    zeta_afc.add(tuple(sorted(list(sub_c))))

    return zeta_afmc, zeta_afc

if __name__ == "__main__":
    from preprocess_fb100 import load_facebook100_data

    nodes, edges, node_attributes = load_facebook100_data(
        mat_filename="American75.mat",
        folder_name="facebook100",
        max_nodes=250,
        attribute_type="major"
    )

    # Ground truth from BK
    bk_afmc, bk_afc = bk_afc_miner(nodes, edges, node_attributes)
    
    print(f"--- Bron-Kerbosch (Ground Truth) ---")
    print(f"Maximal Fair Cliques: {len(bk_afmc)}")
    print(f"Total Fair Cliques:   {len(bk_afc)}")
    
    # Check max size found by BK
    if bk_afc:
        max_len = max(len(c) for c in bk_afc)
        print(f"Largest Fair Clique Size: {max_len}")