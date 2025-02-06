from NetworkTopology import *
import csv
import numpy as np
import pickle
import json

def parse_topology(network_name: str):
    network = Network(network_name)

    with open(f"../data/topologies/{network_name}/edges.txt") as fi:
        reader = csv.reader(fi, delimiter=" ")
        for row_ in reader:
            if row_[0] == 'to_node': continue
            row = [x for x in row_ if x]
            to_node = row[0]
            from_node = row[1]
            capacity = int(row[2])
            network.add_node(to_node, None, None)
            network.add_node(from_node, None, None)
            network.add_edge(from_node, to_node, 200, capacity / 1000)            
    return network

def parse_topology_geant(network_name: str):
    assert network_name == 'geant'
    network = Network(network_name)

    with open(f"../data/topologies/{network_name}/edges.txt") as fi:
        reader = csv.reader(fi, delimiter=",")
        for row_ in reader:
            row = [x for x in row_ if x]
            to_node = str(int(row[0]) + 1)
            from_node = str(int(row[1]) + 1)
            capacity = float(row[2])
            network.add_node(to_node, None, None)
            network.add_node(from_node, None, None)
            network.add_edge(from_node, to_node, 200, capacity / 1e6)       
            network.add_edge(to_node, from_node, 200, capacity / 1e6)     
    return network

def parse_topology_kdl(network_name: str):
    assert network_name == 'kdl' or network_name == "KDL"
    with open("../data/topologies/kdl/kdl.json", 'r') as json_file:
        data = json.load(json_file)

    network = Network(network_name)
    links_seen = set()
    for link in data['links']:
        from_node = str(link['source'])
        to_node = str(link['target'])
        assert (from_node, to_node) not in links_seen
        capacity = link['capacity']
        network.add_node(from_node, None, None)
        network.add_node(to_node, None, None)
        network.add_edge(from_node, to_node, 200, capacity / 1000)
        links_seen.add((from_node, to_node))
    return network

def get_demand_matrix(network):
    network_name = network.name
    num_nodes = len(network.nodes)
    demand_matrix = {}
    with open(f"../data/topologies/{network_name}/demand.txt") as fi:
        reader = csv.reader(fi, delimiter=" ")
        for row_ in reader:
            row = [float(x) for x in row_ if x]
            assert len(row) == num_nodes ** 2
            for idx, dem in enumerate(row):
                from_node = int(idx/num_nodes) + 1
                to_node = (idx % num_nodes) + 1
                assert str(from_node) in network.nodes
                assert str(to_node) in network.nodes
                if from_node == to_node: continue
                if from_node not in demand_matrix:
                    demand_matrix[from_node] = {}
                if to_node not in demand_matrix[from_node]:
                    demand_matrix[from_node][to_node] = []
                demand_matrix[from_node][to_node].append(dem)
    return demand_matrix

def parse_demands(network, scale=1):
    divide_by = 1000.0
    if network.name == 'geant':
        divide_by = 1e6
    demand_matrix = get_demand_matrix(network)
    for from_node in demand_matrix:
        for to_node in demand_matrix[from_node]:
            max_demand = max(demand_matrix[from_node][to_node])
            network.add_demand(str(from_node), str(to_node), max_demand / divide_by, scale)
    if network.tunnels:
        remove_demands_without_tunnels(network)

def parse_demands_avg(network, scale=1):
    divide_by = 1000.0
    if network.name == 'geant':
        divide_by = 1e6
    demand_matrix = get_demand_matrix(network)
    for from_node in demand_matrix:
        for to_node in demand_matrix[from_node]:
            max_demand = np.mean(demand_matrix[from_node][to_node])
            network.add_demand(str(from_node), str(to_node), max_demand / divide_by, scale)
    if network.tunnels:
        remove_demands_without_tunnels(network)

def parse_demands_ts(network, ts, demand_matrix=None, scale=1):
    divide_by = 1000.0
    if network.name == 'geant':
        divide_by = 1e6
    if demand_matrix is None:
        demand_matrix = get_demand_matrix(network)
    for from_node in demand_matrix:
        for to_node in demand_matrix[from_node]:
            demand_val = demand_matrix[from_node][to_node][ts]
            network.add_demand(str(from_node), str(to_node), demand_val / divide_by, scale)
    if network.tunnels:
        remove_demands_without_tunnels(network)

def perturb_demands_empirical_kdl(demand_amounts, scale=1):
    empirical_data = pickle.load(open("../data/demand_deviation_pdf.pkl", "rb"))

    deviations = empirical_data['DeviationPercent']
    probabilities = empirical_data['pdf']
    new_demand_amounts = {}
    for demand_pair, demand in demand_amounts.items():
        true_demand = demand * scale
        prob = np.random.choice(deviations, p=probabilities) / 100.0
        perturbed_demand = true_demand - (prob * true_demand)
        if perturbed_demand < 1e-4:
            perturbed_demand = 0
        new_demand_amounts[demand_pair] = perturbed_demand
    
    return new_demand_amounts

def perturb_demands_empirical_max(network, distribution_filename, scale=1):
    demand_matrix = get_demand_matrix(network)
    empirical_data = pickle.load(open(distribution_filename, "rb"))

    deviations = empirical_data['DeviationPercent']
    probabilities = empirical_data['pdf']
    for from_node in demand_matrix:
        for to_node in demand_matrix[from_node]:
            max_demand = max(demand_matrix[from_node][to_node]) * scale
            perturbed_demand = max_demand - ((np.random.choice(deviations, p=probabilities) / 100.0) * max_demand)
            assert perturbed_demand >= 0
            demand = network.demands[(str(from_node), str(to_node))]
            demand.change_amount(perturbed_demand / 1000.0)
    
def perturb_demands_empirical(network, distribution_filename):
    empirical_data = pickle.load(open(distribution_filename, "rb"))

    deviations = empirical_data['DeviationPercent']
    probabilities = empirical_data['pdf']
    for demand in network.demands.values():
        perturbed_demand = demand.amount - ((np.random.choice(deviations, p=probabilities) / 100.0) * demand.amount)
        # assert perturbed_demand >= 0
        if perturbed_demand < 1:
            perturbed_demand = 0
        demand.change_amount(perturbed_demand)

def parse_tunnels_only_for_demands(network):
    covered_demands = 0
    for demand in network.demands.values():
        try:
            paths = network.k_shortest_paths(demand.src, demand.dst, 4)
            for path in paths:
                tunnel = network.add_tunnel(path)
            covered_demands += 1
        except Exception as e:
            print("No path from", demand.src, demand.dst)
            continue
    print(f"Covered {covered_demands * 100/len(network.demands)}% of demands")

def parse_tunnels_only_for_demands_edge_disjoint(network):
    covered_demands = 0
    for demand in network.demands.values():
        try:
            paths = network.k_shortest_edge_disjoint_paths(demand.src, demand.dst, 4)
            for path in paths:
                tunnel = network.add_tunnel(path)
            covered_demands += 1
        except Exception as e:
            print("No path from", demand.src, demand.dst)
            continue
    print(f"Covered {covered_demands * 100/len(network.demands)}% of demands")

def parse_tunnels(network):
    # Parse tunnels
    num_pairwise = 0
    for node1 in network.nodes:
        for node2 in network.nodes:
            if node1 == node2: continue
            paths = network.k_shortest_paths(node1, node2, 4)
            for path in paths:
                tunnel = network.add_tunnel(path)
            num_pairwise += 1
            if num_pairwise % 1000 == 0:
                print(f"{num_pairwise} complete")
    if network.demands:
        remove_demands_without_tunnels(network)

def parse_tunnels_edge_disjoint(network):
    # Parse tunnels
    for node1 in network.nodes:
        for node2 in network.nodes:
            if node1 == node2: continue
            paths = network.k_shortest_edge_disjoint_paths(node1, node2, 4)
            for path in paths:
                tunnel = network.add_tunnel(path)
    if network.demands:
        remove_demands_without_tunnels(network)

def remove_demands_without_tunnels(network):
    removable_demands = [p for p, d in network.demands.items() if not d.tunnels]
    assert len(removable_demands) == 0
    for demand_pair in removable_demands:
        del network.demands[demand_pair]