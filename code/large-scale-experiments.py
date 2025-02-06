import pickle
import random
from NetworkTopology import *
from NetworkParser import *
from solver import *
from util import *
import sys
import copy

sys.setrecursionlimit(100000)

def rank_slices(slices, network):
    node_to_partition_map = {}
    for i, partition in enumerate(slices):
        for node in partition:
            node_to_partition_map[node] = i
    egress_demand_per_slice = {i: 0 for i in range(len(slices))}
    for demand_id, demand in network.demands.items():
        egress_demand_per_slice[node_to_partition_map[demand_id[0]]] += demand.amount
    return np.std(list(egress_demand_per_slice.values()))

TOPOLOGY_NAME = "geant"
DEMAND_SCALE = 1
NUM_TOTAL_TIMESTAMPS = 5332  # the number of demand matrices available for the topology
NUM_SAMPLES = 1000
REG_PARAMS = {'max_flow': 1, 'mcf': 0.0001, 'min_mlu': 0.0001}
SLICES_FILENAME = "<FILENAME>"  # pickle file for pregenerated slices
DEVIATION_DISTRIBUTION_FILENAME = "<FILENAME>"  # pickle file for the empirical demand deviation distribution --- this should be of the form {'DeviationPercent': [], 'pdf': []}, where 'DeviationPercent is a value between 0 and 100 indicating the percent difference between two slice controllers and 'pdf' is the probability of that deviation
OUTPUT_FILENAME = "<FILENAME>"  # pickle file for output
network = parse_topology_geant(TOPOLOGY_NAME) # use the function appropriate for the topology -- parse_topology() for ATT and parse_topology_kdl() for KDL
network_base = copy.deepcopy(network)

with open(SLICES_FILENAME, "rb") as fp:
    slice_results = pickle.load(fp)

parse_demands_avg(network)
parse_tunnels_only_for_demands(network)
tunnel_ids = hash_tunnels(network)

slice_vals = []
for slices in slice_results:
    slice_vals.append(rank_slices(slices, network))
chosen_slice = slice_results['slices'][np.argmin(slice_vals)]

node_to_partition_map = {}
for i, partition in enumerate(chosen_slice):
    for node in partition:
        node_to_partition_map[node] = i

network = copy.deepcopy(network_base)
parse_demands(network)
parse_tunnels_only_for_demands(network)

edge_cnts = {}
for edge in network.edges:
    unique_slices = set()
    for tunnel in network.edges[edge].tunnels:
        parsed_tunnel = tunnel.name().split(":")
        unique_slices.add(node_to_partition_map[parsed_tunnel[0]])
    edge_cnts[edge] = len(unique_slices)

dropout = set()
for e, cnt in edge_cnts.items():
    if cnt <= 1:
        dropout.add(e)

for e, edge in network.edges.items():
    if edge_cnts[e] <= 1: continue
    max_flow = 0
    for tunnel in edge.tunnels:
        parsed_tunnel = tunnel.name().split(":")
        src = parsed_tunnel[0]
        dst = parsed_tunnel[-1]
        max_flow += network.demands[(src, dst)].amount
    if max_flow / edge.capacity < 1:
        dropout.add(e)

DEMAND_SCALE = 1
timestamps = random.sample(list(range(NUM_TOTAL_TIMESTAMPS)), NUM_SAMPLES)

results = {
    'slicing': chosen_slice,
    'timestamps': timestamps,
    'reg_param': REG_PARAMS,
    'demand_scale': DEMAND_SCALE,
    'throughput': {'max_flow': {'simplex': [], 'scratch_capacity': [], 'barrier': [], 'concord': [], 'concord_opt': [], 'blastshield': [], 'oracle': []}, 'mcf': {'simplex': [], 'scratch_capacity': [], 'barrier': [], 'concord': [], 'concord_opt': [], 'blastshield': []}, 'min_mlu': {'simplex': [], 'barrier': [], 'concord': [], 'concord_opt': [], 'blastshield': [], 'oracle': []}},
    'divergence': {'max_flow': {'simplex': [], 'scratch_capacity': [], 'barrier': [], 'concord': [], 'concord_opt': [], 'blastshield': []}, 'mcf': {'simplex': [], 'scratch_capacity': [], 'barrier': [], 'concord': [], 'concord_opt': [], 'blastshield': []}, 'min_mlu': {'simplex': [], 'barrier': [], 'concord': [],'concord_opt': [], 'blastshield': [], 'oracle': []}},
    'edge_allocs': {'max_flow': {'simplex': [], 'scratch_capacity': [], 'barrier': [], 'concord': [], 'concord_opt': [], 'blastshield': [], 'oracle': []}, 'mcf': {'simplex': [], 'scratch_capacity': [], 'barrier': [], 'concord': [], 'concord_opt': [], 'blastshield': []}, 'min_mlu': {'simplex': [], 'barrier': [], 'concord': [], 'concord_opt': [], 'blastshield': [], 'oracle': []}},
    'realized_mlus': {'simplex': [], 'barrier': [], 'concord': [], 'concord_opt': [], 'blastshield': [], 'oracle': []}
}

for ts in timestamps:
    print("Iteration: ", ts)
    network = copy.deepcopy(network_base)
    parse_demands_ts(network, ts, scale=DEMAND_SCALE)
    parse_tunnels_only_for_demands(network)
    tunnel_ids = hash_tunnels(network)
    
    max_flow_weights = {'simplex': [], 'scratch_capacity': [], 'barrier': [], 'concord': [], 'concord_opt': [], 'blastshield': []}
    mcf_weights =  {'simplex': [], 'scratch_capacity': [], 'barrier': [], 'concord': [], 'concord_opt': [], 'blastshield': []}
    min_mlu_weights = {'simplex': [], 'barrier': [], 'concord': [], 'concord_opt': [], 'blastshield': []}
  
    perturbed_networks = []
    for i in range(len(chosen_slice)):
        new_network = copy.deepcopy(network)
        perturb_demands_empirical(new_network, DEVIATION_DISTRIBUTION_FILENAME)
        perturbed_networks.append(new_network)
        max_flow_weights['simplex'].append(solve_vanilla_max_flow(new_network, tunnel_ids))
        max_flow_weights['scratch_capacity'].append(solve_vanilla_max_flow(new_network, tunnel_ids, capacity_mult=0.95))
        max_flow_weights['barrier'].append(solve_vanilla_max_flow(new_network, tunnel_ids, method=2))
        max_flow_weights['concord'].append(solve_concord_max_flow(new_network, tunnel_ids, reg_param=REG_PARAMS['max_flow']))
        max_flow_weights['concord_opt'].append(solve_concord_max_flow(new_network, tunnel_ids, dropout=dropout, reg_param=REG_PARAMS['max_flow']))
        max_flow_weights['blastshield'].append(solve_vanilla_max_flow(new_network, tunnel_ids, blastshield=True))

        mcf_weights['simplex'].append(solve_vanilla_mcf(new_network, tunnel_ids))
        mcf_weights['scratch_capacity'].append(solve_vanilla_mcf(new_network, tunnel_ids, capacity_mult=0.95))
        mcf_weights['barrier'].append(solve_vanilla_mcf(new_network, tunnel_ids, method=2))
        mcf_weights['concord'].append(solve_concord_mcf(new_network, tunnel_ids, reg_param=REG_PARAMS['mcf']))
        mcf_weights['concord_opt'].append(solve_concord_mcf(new_network, tunnel_ids, dropout=dropout, reg_param=REG_PARAMS['mcf']))
        mcf_weights['blastshield'].append(solve_vanilla_mcf(new_network, tunnel_ids, blastshield=True))

        min_mlu_weights['simplex'].append(solve_vanilla_min_mlu(new_network, tunnel_ids))
        min_mlu_weights['barrier'].append(solve_vanilla_min_mlu(new_network, tunnel_ids, method=2))
        min_mlu_weights['concord'].append(solve_concord_min_mlu(new_network, tunnel_ids, dropout=dropout, reg_param=REG_PARAMS['min_mlu']))
        min_mlu_weights['concord_opt'].append(solve_concord_min_mlu(new_network, tunnel_ids, reg_param=REG_PARAMS['min_mlu'], dropout=dropout))
        min_mlu_weights['blastshield'].append(solve_vanilla_min_mlu(new_network, tunnel_ids, blastshield=True))
    
    max_flow_allocs = {'simplex': [], 'scratch_capacity': [], 'barrier': [], 'concord': [], 'concord_opt': []}
    mcf_allocs =  {'simplex': [], 'scratch_capacity': [], 'barrier': [], 'concord': [], 'concord_opt': []}
    min_mlu_allocs = {'simplex': [], 'barrier': [], 'concord': [], 'concord_opt': []}
    for j, new_network in enumerate(perturbed_networks):
        for k, v in max_flow_weights.items():
            if k != 'blastshield':
                max_flow_allocs[k].append(get_flow_allocations_from_weights(v[j], new_network, tunnel_ids))
            
        for k, v in mcf_weights.items():
            if k != 'blastshield':
                mcf_allocs[k].append(get_flow_allocations_from_weights(v[j], new_network, tunnel_ids))
            
        for k, v in min_mlu_weights.items():
            if k != 'blastshield':
                min_mlu_allocs[k].append(get_flow_allocations_from_weights(v[j], new_network, tunnel_ids))
    
    max_flow_slice_allocs = {method: flows_per_slice(allocs, chosen_slice, tunnel_ids) for method, allocs in max_flow_allocs.items()}
    mcf_slice_allocs = {method: flows_per_slice(allocs, chosen_slice, tunnel_ids) for method, allocs in mcf_allocs.items()}
    min_mlu_slice_allocs = {method: flows_per_slice(allocs, chosen_slice, tunnel_ids) for method, allocs in min_mlu_allocs.items()}

    max_flow_throughputs = {method: sum([get_raw_throughput(alloc) for alloc in allocs]) for method, allocs in max_flow_slice_allocs.items()}
    mcf_throughputs = {method: sum([get_raw_throughput(alloc) for alloc in allocs]) for method, allocs in mcf_slice_allocs.items()}
    min_mlu_throughputs = {method: sum([get_raw_throughput(alloc) for alloc in allocs]) for method, allocs in min_mlu_slice_allocs.items()}

    max_flow_divergences = {method: compute_overflows(allocs, network, tunnel_ids) for method, allocs in max_flow_slice_allocs.items()}
    mcf_divergences = {method: compute_overflows(allocs, network, tunnel_ids) for method, allocs in mcf_slice_allocs.items()}
    min_mlu_divergences = {method: compute_overflows(allocs, network, tunnel_ids) for method, allocs in min_mlu_slice_allocs.items()}

    max_flow_edge_allocs = {method: edge_allocations(allocs, network, tunnel_ids) for method, allocs in max_flow_slice_allocs.items()}
    mcf_edge_allocs = {method: edge_allocations(allocs, network, tunnel_ids) for method, allocs in mcf_slice_allocs.items()}
    min_mlu_edge_allocs = {method: edge_allocations(allocs, network, tunnel_ids) for method, allocs in min_mlu_slice_allocs.items()}

    for method in max_flow_throughputs.keys():
        results[f'throughput']['max_flow'][method].append(max_flow_throughputs[method])
        results[f'divergence']['max_flow'][method].append(sum(val for val in max_flow_divergences[method].values()))
        results[f'edge_allocs']['max_flow'][method].append(max_flow_edge_allocs[method])
        
    for method in mcf_throughputs.keys():
        results[f'throughput']['mcf'][method].append(mcf_throughputs[method])
        results[f'divergence']['mcf'][method].append(sum(val for val in mcf_divergences[method].values()))
        results[f'edge_allocs']['mcf'][method].append(mcf_edge_allocs[method])

    for method in min_mlu_throughputs.keys():
        results[f'throughput']['min_mlu'][method].append(min_mlu_throughputs[method])
        results[f'divergence']['min_mlu'][method].append(sum(val for val in min_mlu_divergences[method].values()))
        results[f'edge_allocs']['min_mlu'][method].append(min_mlu_edge_allocs[method])

    bs_max_flow_throughput, bs_max_flow_divergence, bs_max_flow_edge_allocs = slice_routing(max_flow_weights['blastshield'], perturbed_networks, node_to_partition_map, tunnel_ids)
    bs_mcf_throughput, bs_mcf_divergence, bs_mcf_edge_allocs = slice_routing(mcf_weights['blastshield'], perturbed_networks, node_to_partition_map, tunnel_ids)
    bs_min_mlu_throughput, bs_min_mlu_divergence, bs_min_mlu_edge_allocs = slice_routing(min_mlu_weights['blastshield'], perturbed_networks, node_to_partition_map, tunnel_ids)

    results[f'throughput']['max_flow']['blastshield'].append(bs_max_flow_throughput)
    results[f'divergence']['max_flow']['blastshield'].append(sum(val for val in bs_max_flow_divergence.values()))
    results[f'edge_allocs']['max_flow']['blastshield'].append(bs_max_flow_edge_allocs)
    results[f'throughput']['mcf']['blastshield'].append(bs_mcf_throughput)
    results[f'divergence']['mcf']['blastshield'].append(sum(val for val in bs_mcf_divergence.values()))
    results[f'edge_allocs']['mcf']['blastshield'].append(bs_mcf_edge_allocs)
    results[f'throughput']['min_mlu']['blastshield'].append(bs_min_mlu_throughput)
    results[f'divergence']['min_mlu']['blastshield'].append(sum(val for val in bs_min_mlu_divergence.values()))
    results[f'edge_allocs']['min_mlu']['blastshield'].append(bs_min_mlu_edge_allocs)

    for method in min_mlu_throughputs.keys():
        results[f'realized_mlus'][method].append(get_realized_mlu(min_mlu_slice_allocs[method], network, tunnel_ids))
    
    max_link_util = 0
    for e, alloc in bs_min_mlu_edge_allocs.items():
        curr_util = alloc/network.edges[e].capacity
        if curr_util > max_link_util:
            max_link_util = curr_util
    results[f'realized_mlus']['blastshield'].append(max_link_util)
    
    # create the oracle network
    stitch_demands(network, perturbed_networks, chosen_slice)
    oracle_max_flow_weights = solve_vanilla_max_flow(network, tunnel_ids)
    oracle_max_flow_allocs = get_flow_allocations_from_weights(oracle_max_flow_weights, network, tunnel_ids)
    results['throughput']['max_flow']['oracle'].append(get_raw_throughput(oracle_max_flow_allocs))

    oracle_min_mlu_weights = solve_vanilla_min_mlu(network, tunnel_ids)
    oracle_min_mlu_allocs = get_flow_allocations_from_weights(oracle_min_mlu_weights, network, tunnel_ids)
    results['throughput']['min_mlu']['oracle'].append(get_raw_throughput(oracle_min_mlu_allocs))

    results['edge_allocs']['max_flow']['oracle'].append(edge_allocations_oracle(oracle_max_flow_allocs, network, tunnel_ids))
    oracle_min_mlu_edge_allocs = edge_allocations_oracle(oracle_min_mlu_allocs, network, tunnel_ids)
    results['edge_allocs']['min_mlu']['oracle'].append(oracle_min_mlu_edge_allocs)

    oracle_overflow = 0
    max_link_util = 0
    for e, alloc in oracle_min_mlu_edge_allocs.items():
        oracle_overflow += max(0, alloc - network.edges[e].capacity)
        curr_util = alloc/network.edges[e].capacity
        if curr_util > max_link_util:
            max_link_util = curr_util

    results['divergence']['min_mlu']['oracle'].append(oracle_overflow)
    results['realized_mlus']['oracle'].append(max_link_util)

with open(OUTPUT_FILENAME, 'wb') as f:
    pickle.dump(results, f)