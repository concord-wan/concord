import numpy as np
import copy

def hash_tunnels(network):
    hash_map = {}
    count = 0
    for tunnel in network.tunnels:
        hash_map[tunnel] = count
        count += 1
    return hash_map

def invert_tunnel_ids(tunnel_ids):
    inverted = {}
    for k,v in tunnel_ids.items():
        inverted[v] = k
    return inverted

def update_demand_tunnels(network):
    for tunnel in network.tunnels:
        split_tunnel = tunnel.split(":")
        potential_demand = (split_tunnel[0], split_tunnel[-1])
        if potential_demand in network.demands:
            network.demands[potential_demand].add_tunnel(network.tunnels[tunnel])

def flows_per_slice(slice_flow_allocations, slices, tunnel_ids):
    # Assume parallel lists
    assert len(slice_flow_allocations) == len(slices)

    postproc_slice_allocs = []

    inverted_tunnel_ids = invert_tunnel_ids(tunnel_ids)
    for slice, flow_allocation in zip(slices, slice_flow_allocations):
        slice_alloc = {}
        for tunnel_id, flow in flow_allocation.items():
            if tunnel_id == 'mcf' or tunnel_id == 'Z' or tunnel_id.startswith('u') or tunnel_id.startswith('z'): continue
            tunnel = inverted_tunnel_ids[int(tunnel_id)]
            tunnel_parsed = tunnel.split(":")
            if tunnel_parsed[0] in slice:
                slice_alloc[str(tunnel_id)] = flow
        postproc_slice_allocs.append(slice_alloc)
    return postproc_slice_allocs

def stitch_demands(network, perturbed_networks, slices):
    checker = {demand_id: False for demand_id in network.demands}
    for slice, perturbed_network in zip(slices, perturbed_networks):
        assert len(perturbed_network.demands) > 0
        for demand_id, demand in network.demands.items():
            if demand_id[0] in slice:
                if demand_id not in perturbed_network.demands:
                    demand.change_amount(0)
                else:
                    demand.change_amount(perturbed_network.demands[demand_id].amount)
                checker[demand_id] = True
    
    assert all(checker.values())

def get_raw_throughput(flow_allocations):
    return sum(flow_allocations.values())

def get_flow_allocations_from_weights(flow_allocations, network, tunnel_ids):
    inverted_tunnel_ids = invert_tunnel_ids(tunnel_ids)
    new_flow_allocations = {}
    for flow_id, weight in flow_allocations.items():
        if flow_id == 'mcf' or flow_id == 'Z' or flow_id.startswith('u') or flow_id.startswith('z'): continue
        tunnel = inverted_tunnel_ids[int(flow_id)]
        tunnel_split = tunnel.split(":")
        demand_id = (tunnel_split[0], tunnel_split[-1])
        new_flow_allocations[flow_id] = weight * network.demands[demand_id].amount
    return new_flow_allocations

def get_total_demand(network):
    return sum([demand.amount for demand in network.demands.values()])

def edge_allocations(slice_flow_allocations, network, tunnel_ids):
    edge_allocs = {edge: 0 for edge in network.edges}
    for allocs in slice_flow_allocations:
        for edge in network.edges.values():
            for tunnel in edge.tunnels:
                if str(tunnel_ids[tunnel.name()]) in allocs:
                    edge_allocs[edge.e] += allocs[str(tunnel_ids[tunnel.name()])]
    return edge_allocs

def edge_allocations_oracle(flow_allocations, network, tunnel_ids):
    edge_allocs = {edge: 0 for edge in network.edges}
    for edge in network.edges.values():
        for tunnel in edge.tunnels:
            if str(tunnel_ids[tunnel.name()]) in flow_allocations:
                edge_allocs[edge.e] += flow_allocations[str(tunnel_ids[tunnel.name()])]
    return edge_allocs

def get_pass_through_tunnels(network, tunnel_ids):
    node_tunnels = {node: {} for node in network.nodes}
    for demand in network.demands.values():
        for tunnel in demand.tunnels:
            tunnel_split = tunnel.name().split(":")[:-1]
            for node in tunnel_split:
                if (demand.src, demand.dst) not in node_tunnels[node]:
                    node_tunnels[node][(demand.src, demand.dst)] = []
                node_tunnels[node][(demand.src, demand.dst)].append(tunnel.name())
    return node_tunnels

def program_router_flow_weights(slice_weights, networks, node_to_partition_map, tunnel_ids):
    router_flow_weights = {}
    router_tunnels = get_pass_through_tunnels(networks[0], tunnel_ids)
    for node in node_to_partition_map:
        router_flow_weights[node] = {}
    for node, slice_id in node_to_partition_map.items():
        for demand, tunnels in router_tunnels[node].items():
            tmp = {}
            for tunnel in tunnels:
               split_tunnel = tunnel.split(":")
               # find this node in split_tunnel and find the next hop
               node_index = split_tunnel.index(node)
               next_hop = split_tunnel[node_index + 1]
               if next_hop not in tmp:
                   tmp[next_hop] = 0
               tmp[next_hop] += slice_weights[slice_id][str(tunnel_ids[tunnel])]
            if demand[0] != node:
                weight_sum = sum(tmp.values())
                router_flow_weights[node][demand] = {next_hop: tmp[next_hop] / weight_sum for next_hop in tmp}
                assert abs(sum(router_flow_weights[node][demand].values()) - 1) < 0.001
            else:
                router_flow_weights[node][demand] = {next_hop: tmp[next_hop] for next_hop in tmp}
    return router_flow_weights

def simulate_routing(router_flow_weights, demand_id, demand_val, edge_allocs):
    egress_flow = sum(router_flow_weights[demand_id[0]][demand_id].values()) * demand_val
    flow_received = {node: 0 for node in router_flow_weights}
    flow_received[demand_id[0]] = egress_flow
    dfs_stack = [(demand_id[0], set())]  # start at source with an empty set of visited nodes
    while dfs_stack:
        node, visited = dfs_stack.pop()
        if demand_id not in router_flow_weights[node]:
            continue
        feasible_weight = sum([weight for next_hop, weight in router_flow_weights[node][demand_id].items() if next_hop not in visited])
        flow_sent = 0
        for next_hop, weight in router_flow_weights[node][demand_id].items():
            if next_hop not in visited:
                flow_received[next_hop] += (weight / feasible_weight) * flow_received[node]
                edge_allocs[(node, next_hop)] += (weight / feasible_weight) * flow_received[node]
                flow_sent += (weight / feasible_weight) * flow_received[node]
                new_visited = visited.copy()
                new_visited.add(next_hop)
                dfs_stack.append((next_hop, new_visited))
        flow_received[node] = 0
    
    if abs(egress_flow - flow_received[demand_id[1]]) >= 1:
        print(f"Diff between flow sent and recv for demand {demand_id}:", egress_flow - flow_received[demand_id[1]], egress_flow, flow_received[demand_id[1]])
        print(demand_id, flow_received)
    assert abs(egress_flow - flow_received[demand_id[1]]) < 1
    return egress_flow

def slice_routing(slice_weights, networks, node_to_partition_map, tunnel_ids):
    edge_allocs = {edge_id: 0 for edge_id in networks[0].edges}
    router_flow_weights = program_router_flow_weights(slice_weights, networks, node_to_partition_map, tunnel_ids)
    throughput = 0
    for demand_id in networks[0].demands:
        slice_id = node_to_partition_map[demand_id[0]]
        flow = simulate_routing(router_flow_weights, demand_id, networks[slice_id].demands[demand_id].amount, edge_allocs)
        throughput += flow
    overflows = {}
    for edge_id, edge in networks[0].edges.items():
        diff = edge_allocs[edge_id] - edge.capacity
        if diff > 0.001:
            overflows[edge_id] = diff
    return throughput, overflows, edge_allocs
   
def compute_overflows(slice_flow_allocations, network, tunnel_ids):
    edge_allocs = edge_allocations(slice_flow_allocations, network, tunnel_ids)
    overflows = {}
    for edge in network.edges:
        diff = edge_allocs[edge] - network.edges[edge].capacity
        if diff > 0.001:
            overflows[edge] = diff
    return overflows

def get_realized_mlu(slice_flow_allocations, network, tunnel_ids):
    edge_allocs = edge_allocations(slice_flow_allocations, network, tunnel_ids)
    max_edge_util = 0
    for edge in network.edges:
        edge_util = edge_allocs[edge] / network.edges[edge].capacity
        if edge_util > max_edge_util:
            max_edge_util = edge_util
    return max_edge_util

def compute_errors(flow_allocations1, flow_allocations2, tunnel_ids):
    error = 0
    for tunnel_id in tunnel_ids.values():
        error += abs(flow_allocations1[str(tunnel_id)] - flow_allocations2[str(tunnel_id)])
    return error

def scale_demands(network, scale):
    for demand in network.demands.values():
        demand.amount *= scale

def gen_node_partition_map(network, partitions):
    node_partition_map = {}
    for i, partition in enumerate(partitions):
        for node in partition:
            node_partition_map[node] = i
    return node_partition_map

def slice_degrees(network, partitions):
    node_partition_map = gen_node_partition_map(network, partitions)
    
    assert len(node_partition_map) == len(network.nodes)
    in_degrees = 0
    out_degrees = 0

    for edge in network.edges.keys():
        if node_partition_map[edge[0]] == node_partition_map[edge[1]]:
            in_degrees += 1
        else:
            out_degrees += 1
    
    return in_degrees, out_degrees

def dmd_pct_intra_slice(network, partitions):
    node_partition_map = gen_node_partition_map(network, partitions)
    
    assert len(node_partition_map) == len(network.nodes)
    intra_slice = 0
    total = 0

    for demand in network.demands.values():
        if node_partition_map[demand.src] == node_partition_map[demand.dst]:
            intra_slice += demand.amount
        total += demand.amount
    
    return intra_slice / total

def num_transit_slices(network, partitions, weighted=False):
    node_partition_map = gen_node_partition_map(network, partitions)

    all_tunnels = []

    for tunnel in network.tunnels.values():
        partition_set = set()
        tunnel_split = tunnel.name().split(":")
        for node in tunnel_split[1:-1]:
            partition_set.add(node_partition_map[node])
        if weighted:
            all_tunnels.append(len(partition_set) * network.demands[(tunnel_split[0], tunnel_split[-1])].amount)
        else:
            all_tunnels.append(len(partition_set))
    
    return np.mean(all_tunnels)

def parse_dote_tunnels(file_path):
    mapping = []  # Dictionary to store the result
    demand_mapping = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if not line:  # Skip empty lines
                continue
            
            # Split the line into the source-destination and tunnels part
            src_dst, tunnels_str = line.split(':')
            src, dst = map(int, src_dst.split())  # Extract src and dst
            src = str(src + 1)
            dst = str(dst + 1)
            demand_mapping.append((src, dst))
            
            for tunnel in tunnels_str.split(','):
                path = tunnel.split('-')
                mapping.append(":".join([str(int(x) + 1) for x in path]))
    
    return mapping, demand_mapping

def convert_dote_weights(path_weights, dote_tunnel_mapping, network, tunnel_ids):
    converted_weights = {}
    if not type(path_weights) == list:
        path_weights = path_weights.numpy()
    else:
        path_weights = np.array(path_weights)
    for i, weight in enumerate(path_weights):
        tunnel = dote_tunnel_mapping[i]
        split_tunnel = tunnel.split(":")
        demand_id = (split_tunnel[0], split_tunnel[-1])
        if tunnel in tunnel_ids and demand_id in network.demands:
            converted_weights[str(tunnel_ids[tunnel])] = weight
    assert len(converted_weights) > 0
    return converted_weights