"""Experiment 2: enumerate all absolute fair cliques using the Year attribute.

The dataset contains four valid class years: 2006, 2007, 2008, and 2009.
This experiment therefore uses the directly supported Year/4V configuration
and follows the paper's Table VI and Table VII output.
"""

# Standard-library modules used for path handling, combination generation,
# temporary storage, and counting category values.
import os
import itertools
import sqlite3
import tempfile
from pathlib import Path
from collections import Counter

# The data path in the existing loader is relative to AFCminer.  Change to the
# script folder so the file works from VS Code, PowerShell, or another folder.
os.chdir(Path(__file__).resolve().parent)

# Reuse the repository's shared subset sizes and graph-loading helper.
from experiment_utils import SUBSET_SIZES, load_top_1500_graph
# Reuse the FCA maximal-clique extraction routine without changing core code.
from optimal_afc import extract_equiconcepts_fast_fca


# Experiment 2 uses Year as a four-valued attribute.  The four categories are
# kept exactly as they appear in the Facebook100 data.
ATTRIBUTE_TYPE = "year"
GRANULARITIES = (4,)


def load_year_graph(granularity=4):
	"""Load the common 1500-node graph and its Year labels."""
	return load_top_1500_graph(
		attribute_type=ATTRIBUTE_TYPE,
		granularity=granularity,
	)


def year_category_values(granularity):
	"""Return the category mapping used for the requested Year granularity."""
	# The helper is retained for compatibility with the paper-style runner.  The
	# current experiment uses only the directly available four Year categories.
	if granularity == 2:
		return {
			2006: "Class_2006_2007",
			2007: "Class_2006_2007",
			2008: "Class_2008_2009",
			2009: "Class_2008_2009",
		}
	if granularity == 3:
		return {
			2006: "Class_2006",
			2007: "Class_2007",
			2008: "Class_2008_2009",
			2009: "Class_2008_2009",
		}
	return {
		2006: "Class_2006",
		2007: "Class_2007",
		2008: "Class_2008",
		2009: "Class_2009",
	}


def print_year_category_table():
	"""Print the Year category counts used by this experiment."""
	# Load the complete 1500-node input because this table describes the dataset,
	# not one of the six progressively sized graph subsets.
	_, _, node_attributes = load_year_graph(4)

	# Print the table title and column headings in the same style as the paper.
	print("\nTABLE VI")
	print("YEAR DATASET")
	print("==============================")
	print(f"{'DataSet':<10} {'Category':<22} {'Tag':<5} {'Number':>8}")
	# Print a horizontal rule to make the table easier to read.
	print("-" * 50)

	# Count the values in the complete 1500-node input, not in each subset.
	for granularity in GRANULARITIES:
		_, _, raw_attributes = load_year_graph(4)
		mapping = year_category_values(granularity)
		categories = Counter(mapping[year] for year in raw_attributes.values())
		for tag, (category, count) in enumerate(categories.items(), start=1):
			# Print one row for each Year category, its display tag, and its count.
			print(f"{f'{granularity}V':<10} {category:<22} {tag:<5} {count:>8}")
		# Separate each granularity section visually.
		print("-" * 50)


def count_all_afcs(nodes, edges, node_attributes):
	"""Count distinct AFCs without keeping every clique in memory.

	The paper's Experiment 2 counts all Absolute Fair Cliques, including
	subcliques derived from maximal cliques.  Storing every generated clique in
	a Python set can exhaust memory, so this function stores distinct clique
	keys in a temporary SQLite database and keeps only the final count in RAM.
	"""
	with tempfile.TemporaryDirectory() as temp_dir:
		database_path = Path(temp_dir) / "afcs.sqlite3"
		connection = sqlite3.connect(database_path)
		try:
			# These settings make the temporary database faster.  The database is
			# disposable, so durability is unnecessary for this calculation.
			connection.execute("PRAGMA journal_mode=OFF")
			connection.execute("PRAGMA synchronous=OFF")
			connection.execute("PRAGMA temp_store=MEMORY")
			connection.execute(
				"CREATE TABLE afcs (clique TEXT PRIMARY KEY)"
			)
			batch = []
			attribute_values = sorted(set(node_attributes.values()))

			# FCA first finds maximal cliques (equiconcepts).  Every balanced
			# subclique of each maximal clique is then an AFC candidate.
			for maximal_clique in extract_equiconcepts_fast_fca(nodes, edges):
				buckets = {attribute: [] for attribute in attribute_values}
				for node in maximal_clique:
					buckets[node_attributes[node]].append(node)
				minimum_bucket_size = min(
					(len(buckets[attribute]) for attribute in attribute_values),
					default=0,
				)

				# A fair clique must select the same number of nodes from every
				# Year category, so only sizes up to the smallest bucket are valid.
				for fair_size in range(1, minimum_bucket_size + 1):
					combinations = [
						itertools.combinations(buckets[attribute], fair_size)
						for attribute in attribute_values
					]
					for product in itertools.product(*combinations):
						# Sort node IDs so the same clique has one canonical key even
						# when it is found through more than one maximal clique.
						clique = tuple(sorted(
							itertools.chain.from_iterable(product)
						))
						batch.append((repr(clique),))
						if len(batch) >= 100000:
							connection.executemany(
								"INSERT OR IGNORE INTO afcs VALUES (?)",
								batch,
							)
							connection.commit()
							batch.clear()

			if batch:
				# Flush the final batch after all maximal cliques are processed.
				connection.executemany(
					"INSERT OR IGNORE INTO afcs VALUES (?)",
					batch,
				)
				connection.commit()

			# Query the number of unique clique keys stored in SQLite.
			count = connection.execute(
				"SELECT COUNT(*) FROM afcs"
			).fetchone()[0]
		finally:
			# Close the database before TemporaryDirectory removes its file.
			connection.close()
		# Return only the count; the temporary clique records are discarded.
		return count


def print_experiment_2_table():
	"""Print all AFC counts for the six nested Year/4V subsets."""
	# Table VII is printed before the expensive work so the user can see that
	# the program is running even while the first subset is being processed.
	# Print the title before mining starts so a long-running calculation is
	# visibly associated with Table VII immediately.
	print("\nTABLE VII", flush=True)
	print("EXPERIMENT-2. YEAR/4V ALL ABSOLUTE FAIR CLIQUES", flush=True)
	print("============================================", flush=True)
	# This experiment has one directly supported Year configuration: four values.
	print(f"{'Dataset':<14} {'4V':>10}", flush=True)
	# Print the header separator before processing the first subset.
	print("-" * 48, flush=True)

	for index, subset_size in enumerate(SUBSET_SIZES, start=1):
		granularity_counts = []
		# The subsets are nested: each larger subset contains the previous one.
		ordered_nodes, all_edges, all_attributes = load_year_graph(4)
		subset_nodes = ordered_nodes[:subset_size]
		subset_node_set = set(subset_nodes)
		subset_edges = [
			(first_node, second_node)
			for first_node, second_node in all_edges
			if first_node in subset_node_set
			and second_node in subset_node_set
		]

		# Run the AFC miner for the selected Year/4V configuration.
		for granularity in GRANULARITIES:
			# Convert the raw Year values into the category labels used by FCA.
			mapping = year_category_values(granularity)
			subset_attributes = {
				node: mapping[all_attributes[node]]
				for node in subset_nodes
			}
			count = count_all_afcs(
				subset_nodes,
				subset_edges,
				subset_attributes,
			)
			granularity_counts.append(count)
			# Show progress as soon as this subset's count is available.
			print(
				f"Completed SubSet {index} {granularity}V: {count}",
				flush=True,
			)
		# Print the final formatted row for this subset after its calculation.
		print(f"SubSet {index:<7} {granularity_counts[0]:>10}", flush=True)

	# Close Table VII after all six nested subsets have been printed.
	print("-" * 48, flush=True)


if __name__ == "__main__":
	# Run the two required tables only when this file is executed directly.
	print_year_category_table()
	print_experiment_2_table()
