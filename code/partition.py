import numpy as np
from NetworkParser import *
from NetworkTopology import *
from solver import *
from util import *
import networkx as nx
import random
import pickle
import sys
sys.setrecursionlimit(100000)

def create_balanced_connected_partitions(graph, node_weights_og, num_partitions, partition_sizes, target, epsilon, elephant_src_cutoff, history):
    # global history
    partitions = [[] for _ in range(num_partitions)]
    # sort node_weights by value and then select the top k nodes
    node_weights = {k: v for k, v in sorted(node_weights_og.items(), key=lambda item: item[1], reverse=True)}
    END_INDEX = elephant_src_cutoff
    elephant_srcs = list(node_weights.keys())[:END_INDEX]  # those that contribute > 1/2 of the total per-slice demand

    candidates = [set() for _ in range(num_partitions)]

    weight_tracker = [0 for _ in range(num_partitions)]

    available = set(graph.nodes)
    i = 0
    for src in elephant_srcs:
        available.remove(src)
        partitions[num_partitions - i - 1].append(src)
        candidates[num_partitions - i - 1].update(set(graph.neighbors(src)))
        weight_tracker[num_partitions - i - 1] += node_weights[src]
        i += 1
    
    while i < num_partitions:
        availabile_list = list(available)
        random.shuffle(availabile_list)
        src = availabile_list[0]
        available.remove(src)
        partitions[num_partitions - i - 1].append(src)
        candidates[num_partitions - i - 1].update(set(graph.neighbors(src)))
        weight_tracker[num_partitions - i - 1] += node_weights[src]
        i += 1
    
    while any(len(partition) < partition_sizes[i] for i, partition in enumerate(partitions)):
        for j, partition in enumerate(partitions):
            # Retry for too little weight
            if len(partition) == partition_sizes[j] and weight_tracker[j] < target * (1 - epsilon):
                return None
            elif len(partition) == partition_sizes[j]:
                continue
            candidates[j] = candidates[j].intersection(available)
            # Retry for new remaining candidates
            if len(candidates[j]) == 0:
                return None
            node = None
            # per-topology tuned heuristics
            if (len(partition) >= partition_sizes[j] - 5 and weight_tracker[j] < target * 0.5) or (len(partition) >= partition_sizes[j] - 3 and weight_tracker[j] < target * (1 - epsilon)):
                # sort candidates by weight and pick the heaviest node that doesn't exceed the target
                sorted_candidates = sorted(candidates[j], key=lambda node: node_weights[node], reverse=True)
                for candidate in sorted_candidates:
                    if weight_tracker[j] + node_weights[candidate] <= target * (1 + epsilon):
                        node = candidate
                        candidates[j].remove(node)
                        break
            else:
                candidates_list = list(candidates[j])
                random.shuffle(candidates_list)
                node = candidates_list[0]
                candidates[j] = set(candidates_list)
            if node is None:
                return None
            available.remove(node)
            tmp = candidates[j].union(set(graph.neighbors(node)))
            candidates[j] = tmp.intersection(available)
            partition.append(node)
            weight_tracker[j] += node_weights[node]
            new_candidates = set(candidates[j])
            for node in candidates[j]:
                if weight_tracker[j] + node_weights[node] > target * (1 + epsilon):
                    new_candidates.remove(node)
            candidates[j] = new_candidates

    # Retry for partitions with too little weight"
    if any(weight < target * (1 - epsilon) for weight in weight_tracker):
        return None
    
    x = {frozenset(partition) for partition in partitions}

    # Returned slicing configuration must be unique
    if x in history:
        return None
    history.append(x)

    return partitions

# set these parameters manually depending on the topology
NUM_SIMULATIONS = 100
EPSILON = 0.3
NUM_PARTITIONS = 5
ELEPHANT_SRC_CUTOFF = 2  
PARTITION_LIST = [[5, 5, 4, 4, 4], [5, 4, 5, 4, 4], [6, 5, 4, 4, 3], [6, 4, 4, 4, 4]]
TOPOLOGY_NAME = "geant"
network = parse_topology_geant(TOPOLOGY_NAME)  # use the function appropriate for the topology -- parse_topology() for ATT and parse_topology_kdl() for KDL
OUTFILE = "<FILENAME>"

parse_demands_avg(network)

node_weights = {}
for demand_id, demand in network.demands.items():
    if demand_id[0] not in node_weights:
        node_weights[demand_id[0]] = 0
    node_weights[demand_id[0]] += demand.amount

TARGET = sum(node_weights.values()) / NUM_PARTITIONS

for node in network.nodes:
    if node not in node_weights:
        node_weights[node] = 0

history = []
all_partitions = []

for i in range(NUM_SIMULATIONS):
    print("Iteration: ", i)
        
    partition_size_idx = 0
    partition_sizes = PARTITION_LIST[partition_size_idx]
    # create sqrt(n) random partitions
    partitions = create_balanced_connected_partitions(network.to_nx(), node_weights, NUM_PARTITIONS, partition_sizes, TARGET, EPSILON, ELEPHANT_SRC_CUTOFF, history)
    while partitions is None:
        partitions = create_balanced_connected_partitions(network.to_nx(), node_weights, NUM_PARTITIONS, partition_sizes, TARGET, EPSILON, ELEPHANT_SRC_CUTOFF, history)
        if partition_size_idx < len(PARTITION_LIST) - 1:
            partition_size_idx += 1
        else:
            partition_size_idx = 0
        partition_sizes = PARTITION_LIST[partition_size_idx]
    all_nodes = set()
    for partition in partitions:
        all_nodes.update(partition)
    if all_nodes != set(network.nodes.keys()):
        print(partitions)
        raise Exception("Partitioning failed")
    assert len(history) == i + 1

    all_partitions.append(partitions)

with open(OUTFILE, "wb") as f:
    pickle.dump(all_partitions, f)

