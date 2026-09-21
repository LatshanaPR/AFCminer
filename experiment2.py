"""
This file will contain the three experiments described in the paper:

1. Binary-valued attribute experiment using gender.
2. Multivalued attribute experiment using year.
3. Multidimensional attribute experiment using gender and year together.

The experiments will use nested Facebook100 subgraphs with these sizes:
250, 500, 750, 1000, 1250, and 1500 nodes.

"""

from collections import Counter
from time import perf_counter
from experiment_utils import SUBSET_SIZES, load_top_1500_graph
from sparse_preprocess import load_facebook100_data, split_preprocessed_data
from AFCMiner import AFCMiner
from bk import BKMiner

# The paper first describes the attribute distribution before running the
# algorithm. We use the same preprocessing function as run_fb.py, but request
# the largest planned input so that the category counts describe 1500 nodes.
TOP_NODE_COUNT = 1500
DATASET_FILE = "American75.mat"
DATASET_FOLDER = "facebook100"


def print_table_ii():
	"""Print friendship statistics for the six nested graph sizes."""
	# This experiment uses the shared graph loader with the gender attribute as
	# its validity filter, but exp2/exp3 can pass "year" or
	# "multidim_gender_year" instead without changing the graph-building logic.
	ordered_nodes, all_edges, _ = load_top_1500_graph(attribute_type="year")

	print("\nTABLE II")
	print("DATASETS I. FRIENDSHIP OF USERS")
	print("================================")
	print(f"\n{'Sub-Dataset':<14} {'Node':>6} {'Edge':>8} {'d_avg':>8} {'d_max':>8}")
	print("-" * 48)

	for index, subset_size in enumerate(SUBSET_SIZES, start=1):
		# Each row uses the first N nodes from the shared top-1500 ordering.
		subset_nodes = set(ordered_nodes[:subset_size])
		subset_edges = [
			(first_node, second_node)
			for first_node, second_node in all_edges
			if first_node in subset_nodes and second_node in subset_nodes
		]

		node_count = len(subset_nodes)
		edge_count = len(subset_edges)
		average_degree = (2 * edge_count) / node_count
		degrees = {node: 0 for node in subset_nodes}
		for first_node, second_node in subset_edges:
			degrees[first_node] += 1
			degrees[second_node] += 1
		maximum_degree = max(degrees.values(), default=0)

		print(
			f"SubSet {index:<7} {node_count:>6} {edge_count:>8} "
			f"{average_degree:>8.2f} {maximum_degree:>8}"
		)

	print("-" * 48)


def run_experiment_1():
    result_rows = []
    runtime_rows = []

    for index, subset_size in enumerate(SUBSET_SIZES, start=1):
        print(f"\nLoading SubSet {index} ({subset_size} nodes)...", flush=True)

        # 1. Load data directly matching the image structure
        nodes, attribute_columns, combined_data = load_facebook100_data(
            mat_filename=DATASET_FILE,
            folder_name=DATASET_FOLDER,
            max_nodes=subset_size,
            attribute_type="gender",
            granularity=2
        )

        # Convert nodes to set or list as required by your signatures
        V = nodes
        node_attribute_set = attribute_columns
        R = combined_data

        # 2. Run Bron-Kerbosch baseline (only on SubSet 1 to prevent overload)
        if index == 1:
            print("  Running BKMiner...", flush=True)
            start_time = perf_counter()
            baseline_afmc = BKMiner(V, node_attribute_set, R)
            baseline_time_ms = (perf_counter() - start_time) * 1000
            baseline_count = len(baseline_afmc)
        

        # 3. Run AFCMiner for all subsets
        print("  Running AFCMiner...", flush=True)
        start_time = perf_counter()
        afcminer_afmc = AFCMiner(V, node_attribute_set, R)
        afcminer_time_ms = (perf_counter() - start_time) * 1000
        afcminer_count = len(afcminer_afmc)

        runtime_rows.append((index, baseline_time_ms, afcminer_time_ms))
        result_rows.append((index, baseline_count, afcminer_count))

    # --- Output Tables ---
    print("\nTABLE IV")
    print("EXPERIMENT-1. DIFFERENT SCALES OF NODES WITH GENDER")
    print("====================================================")
    print(f"{'Dataset':<14} {'Baseline (BK)':>18} {'AFCMiner':>12}")
    print("-" * 48)
    for idx, b_cnt, a_cnt in result_rows:
        print(f"SubSet {idx:<7} {str(b_cnt):>18} {a_cnt:>12}")
    print("-" * 48)

    print("\nTABLE V")
    print("EXPERIMENT-1. RUNNING TIME COMPARISON")
    print("=====================================")
    print(f"{'Dataset':<14} {'BK (ms)':>12} {'AFCMiner (ms)':>18}")
    print("-" * 48)
    for idx, b_time, a_time in runtime_rows:
        b_str = f"{b_time:.2f}" if b_time is not None else "-"
        print(f"SubSet {idx:<7} {b_str:>12} {a_time:>18.2f}")
    print("-" * 48)

def load_attributes_for_table_iii():
	"""Load every supported attribute category for the top-1500 input."""
	# The preprocessing function handles one attribute type per call. Calling it
	# once for each type lets Table III show unused attributes as well.
	attribute_settings = [
		("Status", "status", 2),
		("Gender", "gender", 2),
		("Major", "major", 5),
		("Year", "year", 4),
		("Dorm", "dorm", 5),
	]
	attribute_groups = []

	for dataset_name, attribute_type, granularity in attribute_settings:
		nodes, _, combined_data = load_facebook100_data(
			mat_filename=DATASET_FILE,
			folder_name=DATASET_FOLDER,
			max_nodes=TOP_NODE_COUNT,
			attribute_type=attribute_type,
			granularity=granularity,
		)
		_, node_attributes = split_preprocessed_data(nodes, combined_data)
		attribute_groups.append((dataset_name, nodes, node_attributes))

	return attribute_groups


def print_table_iii():
	"""Print category names, tags, and counts in a Table III-style layout."""
	attribute_groups = load_attributes_for_table_iii()

	print("\nTABLE III")
	print("DATASETS II. CATEGORIES OF DIFFERENT ATTRIBUTES")
	print("===============================================")
	print(f"\n{'Dataset':<12} {'Category':<18} {'Tag':<5} {'Number':>8}")
	print("-" * 48)

	# Tags are assigned in display order. This keeps the output stable while
	# allowing each attribute to have a different number of categories.
	for dataset_name, nodes, node_attributes in attribute_groups:
		counts = Counter(node_attributes.values())
		for tag, (category, count) in enumerate(
			sorted(counts.items()), start=1
		):
			print(
				f"{dataset_name:<16} {category:<18} {tag:<5} "
				f"{count:>8}"
			)
		print("-" * 48)

	print("All groups are based on their own 1500-node preprocessed input.")


if __name__ == "__main__":
	# Table II: friendship statistics for the six nested subsets.
	print_table_ii()

	# Table III: category distribution for the 1500-node graph across
	# different attributes such as status, gender, major, year, and dorm.
	print_table_iii()

	# Table IV: AFMC counts for the gender-based experiment across subsets.
	# Table V: runtime comparison between BK and AFCMiner for the same setup.
	run_experiment_1()
