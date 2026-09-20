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

from bk import bk_afc_miner
from experiment_utils import SUBSET_SIZES, load_top_1500_graph
from optimal_afc import afc_miner_optimal
from preprocess_fb100 import load_facebook100_data, split_preprocessed_data


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
	ordered_nodes, all_edges, _ = load_top_1500_graph(attribute_type="gender")

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


def print_table_iv():
	"""Print AFMC counts and runtimes for the gender experiments."""
	ordered_nodes, all_edges, all_attributes = load_top_1500_graph()
	result_rows = []
	runtime_rows = []

	for index, subset_size in enumerate(SUBSET_SIZES, start=1):
		# Use the same nested node set and induced edges for both algorithms.
		subset_nodes = ordered_nodes[:subset_size]
		subset_node_set = set(subset_nodes)
		subset_edges = [
			(first_node, second_node)
			for first_node, second_node in all_edges
			if first_node in subset_node_set and second_node in subset_node_set
		]
		subset_attributes = {
			node: all_attributes[node]
			for node in subset_nodes
		}
		if index == 1:
			print(
				f"Running SubSet {index} ({subset_size} nodes): "
				"Bron-Kerbosch and AFCMiner...",
				flush=True,
			)
		else:
			print(
				f"Running SubSet {index} ({subset_size} nodes): AFCMiner...",
				flush=True,
			)

		# Run and time Bron-Kerbosch only on SubSet 1 because larger baseline
		# runs overload the laptop.
		if index == 1:
			start_time = perf_counter()
			baseline_afmc, _ = bk_afc_miner(
				subset_nodes,
				subset_edges,
				subset_attributes,
				include_afc=False,
			)
			baseline_time_ms = (perf_counter() - start_time) * 1000
			baseline_count = len(baseline_afmc)
		else:
			baseline_time_ms = None
			baseline_count = "-"

		# Run and time AFCMiner for every subset. Sub-clique generation is
		# disabled because these tables report AFMC results only.
		start_time = perf_counter()
		afcminer_afmc, _ = afc_miner_optimal(
			subset_nodes,
			subset_edges,
			subset_attributes,
			include_afc=False,
		)
		afcminer_time_ms = (perf_counter() - start_time) * 1000

		runtime_rows.append(
			(index, baseline_time_ms, afcminer_time_ms)
		)
		result_rows.append((index, baseline_count, len(afcminer_afmc)))

	print("\nTABLE IV")
	print("EXPERIMENT-1. DIFFERENT SCALES OF NODES WITH GENDER")
	print("====================================================")
	print(f"\n{'Dataset':<14} {'Baseline Methods':>18} {'AFCMiner':>12}")
	print(f"{'':<14} {'AFMC':>18} {'AFMC':>12}")
	print("-" * 48)
	for index, baseline_count, afcminer_count in result_rows:
		print(
			f"SubSet {index:<7} {str(baseline_count):>18} "
			f"{afcminer_count:>12}"
		)
	print("-" * 48)

	print("\nTABLE V")
	print("EXPERIMENT-1. RUNNING TIME COMPARISON")
	print("=====================================")
	print(f"\n{'Dataset':<14} {'BK (ms)':>12} {'AFCMiner (ms)':>18}")
	print("-" * 48)
	for index, baseline_time_ms, afcminer_time_ms in runtime_rows:
		baseline_text = (
			f"{baseline_time_ms:.2f}" if baseline_time_ms is not None else "-"
		)
		print(
			f"SubSet {index:<7} {baseline_text:>12} "
			f"{afcminer_time_ms:>18.2f}"
		)
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
	print_table_iv()
