from gurobipy import *

# low divergence graph partitioning
def graph_partitioning_solver(network, flow_allocations, tunnel_ids, num_partitions, partition_size):
    for simulation in flow_allocations:
        assert len(simulation) == num_partitions

    # create model
    model = Model("graph_partitioning")

    # create variables
    indicator_vars = {}
    for node in network.nodes:
        indicator_vars[node] = {}
        for i in range(num_partitions):
            indicator_vars[node][i] = model.addVar(vtype=GRB.BINARY, name=f"{node}_{i}")
    
    losses = {}
    for edge in network.edges:
        losses[edge] = model.addVar(vtype=GRB.CONTINUOUS, name=f"{edge}_loss")
    
    tunnel_flows = {}
    for h in range(len(flow_allocations)):
        tunnel_flows[h] = {}
        for tunnel_id in tunnel_ids.values():
            tunnel_flows[h][tunnel_id] = model.addVar(vtype=GRB.CONTINUOUS, name=f"{h}_{tunnel_id}_flow")

    # add constraints

    # ensure each node is assigned to exactly one partition
    for node in network.nodes:
        model.addConstr(quicksum(indicator_vars[node][i] for i in range(num_partitions)) == 1)
    
    # ensure each partition contains exactly partition_size nodes
    for i in range(num_partitions):
        model.addConstr(quicksum(indicator_vars[node][i] for node in network.nodes) == partition_size)
    
    # ensure that the flow on each tunnel is equal to the sum of the flows allocated to that tunnel in that simulation
    for h, allocations in enumerate(flow_allocations):
        for tunnel_name, tunnel_id in tunnel_ids.items():
            source_node = tunnel_name.split(":")[0]
            val = 0
            for slice_id, slice_allocation in enumerate(allocations):
                val += slice_allocation[str(tunnel_id)] * indicator_vars[source_node][slice_id]
            model.addConstr(tunnel_flows[h][tunnel_id] == val)
    
    # ensure that the loss on each edge is equal to the sum of the flows on the tunnels that traverse that edge over all simlations
    for e, edge in network.edges.items():
        edge_loss_slacks = []
        for h in range(len(flow_allocations)):
            edge_alloc = 0
            for tunnel in edge.tunnels:
                edge_alloc += tunnel_flows[h][tunnel_ids[tunnel.name()]]
            slack = model.addVar(vtype=GRB.CONTINUOUS, name=f"{e}_{h}_slack")
            model.addConstr(slack >= edge_alloc - edge.capacity)
            edge_loss_slacks.append(slack)
        model.addConstr(losses[e] == quicksum(edge_loss_slacks))
    
    quadratic_indicator_vars = {}
    for i in range(num_partitions):
        quadratic_indicator_vars[i] = {}
        for node1 in network.nodes:
            for node2 in network.nodes:
                if node1 == node2: continue
                if (node2, node1) in quadratic_indicator_vars[i]:
                    quadratic_indicator_vars[i][(node1, node2)] = quadratic_indicator_vars[i][(node2, node1)]
                else:
                    quadratic_indicator_vars[i][(node1, node2)] = model.addVar(vtype=GRB.CONTINUOUS, name=f"{node}_{node}")
                    model.addConstr(quadratic_indicator_vars[i][(node1, node2)] == indicator_vars[node1][i] * indicator_vars[node2][i])
    
    phantom_flows = {edge: {} for edge in network.edges}
    for src in network.nodes:
        for dst in network.nodes:
            if src == dst: continue
            for edge in network.edges:
                phantom_flows[edge][(src, dst)] = model.addVar(vtype=GRB.CONTINUOUS, lb = 0, name=f"{src}_{dst}_{edge}_phantom_flow")
        for dst in network.nodes:
            if src == dst: continue
            model.addConstr(quicksum(phantom_flows[edge][(src, dst)] for edge in network.nodes[src].outgoing_edges) ==  quicksum(quadratic_indicator_vars[i][(src, dst)] for i in range(num_partitions)))
            model.addConstr(quicksum(phantom_flows[edge][(src, dst)] for edge in network.nodes[src].incoming_edges) ==  0)

    for dst in network.nodes:
        for src in network.nodes:
            if src == dst: continue
            model.addConstr(quicksum(phantom_flows[edge][(src, dst)] for edge in network.nodes[dst].incoming_edges) == quicksum(quadratic_indicator_vars[i][(src, dst)] for i in range(num_partitions)))
            model.addConstr(quicksum(phantom_flows[edge][(src, dst)] for edge in network.nodes[dst].outgoing_edges) ==  0)

    for edge in network.edges:
        for src in network.nodes:
            for dst in network.nodes:
                if src == dst: continue
                model.addConstr(phantom_flows[edge][(src, dst)] <= quicksum(quadratic_indicator_vars[i][(edge[0], edge[1])] for i in range(num_partitions)))

    # flow conservation constraint
    for node in network.nodes:
        for src in network.nodes:
            for dst in network.nodes:
                if src == dst: continue
                if node == src: continue 
                if node == dst: continue
                incoming_flows = [phantom_flows[edge][(src, dst)] for edge in network.nodes[node].incoming_edges]
                outgoing_flows = [phantom_flows[edge][(src, dst)] for edge in network.nodes[node].outgoing_edges]
                model.addConstr(quicksum(incoming_flows) - quicksum(outgoing_flows) == 0)
        
    # set objective
    model.setObjective(quicksum(losses[edge] for edge in network.edges), GRB.MINIMIZE)

    # optimize
    model.optimize()

    # extract results
    partition_assignments = {}
    for node in network.nodes:
        for i in range(num_partitions):
            if indicator_vars[node][i].x == 1:
                partition_assignments[node] = i
                break

    return partition_assignments, {v.VarName : v.X for v in model.getVars()}

# low blast radius graph partitioning
def balanced_demand_partitioning(network, source_demands, num_partitions, partition_sizes, fixed_nodes):
    assert len(partition_sizes) == num_partitions

    # create model
    model = Model("demand_balanced_graph_partitioning")

    # create variables
    indicator_vars = {}
    for node in network.nodes:
        indicator_vars[node] = {}
        for i in range(num_partitions):
            if node not in fixed_nodes:
                indicator_vars[node][i] = model.addVar(vtype=GRB.BINARY, name=f"{node}_{i}")
            else:
                if i == fixed_nodes[node]:
                    indicator_vars[node][i] = 1
                else:
                    indicator_vars[node][i] = 0

    # ensure each node is assigned to exactly one partition
    for node in network.nodes:
        if node not in fixed_nodes:
            model.addConstr(quicksum(indicator_vars[node][i] for i in range(num_partitions)) == 1)
    
    # ensure each partition contains exactly partition_size nodes
    for i in range(num_partitions):
        model.addConstr(quicksum(indicator_vars[node][i] for node in network.nodes) == partition_sizes[i])
    
    slice_egress_flow = {}
    for i in range(num_partitions):
        slice_egress_flow[i] = model.addVar(vtype=GRB.CONTINUOUS, name=f"{i}_egress_flow")
    
    for i in range(num_partitions):
        model.addConstr(slice_egress_flow[i] == quicksum(source_demands[node] * indicator_vars[node][i] for node in network.nodes))
    
    model.update()
    diffs = {}
    for i in range(num_partitions):
        for j in range(num_partitions):
            if i == j: continue
            diffs[(i, j)] = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name=f"{i}_{j}_diff")
            model.addConstr(diffs[(i, j)] == slice_egress_flow[i] - slice_egress_flow[j])
    
    model.update()
    real_diffs = {}
    for (i, j) in diffs:
        real_diffs[(i,j)] = model.addVar(vtype=GRB.CONTINUOUS, name=f"{i}_{j}_real_diff")
        model.addConstr(real_diffs[(i, j)] == abs_(diffs[(i, j)]))
   
    quadratic_indicator_vars = {}
    for i in range(num_partitions):
        quadratic_indicator_vars[i] = {}
        for node1 in network.nodes:
            for node2 in network.nodes:
                if node1 == node2: continue
                if (node2, node1) in quadratic_indicator_vars[i]:
                    quadratic_indicator_vars[i][(node1, node2)] = quadratic_indicator_vars[i][(node2, node1)]
                else:
                    quadratic_indicator_vars[i][(node1, node2)] = model.addVar(vtype=GRB.CONTINUOUS, name=f"{node1}_{node2}")
                    model.addConstr(quadratic_indicator_vars[i][(node1, node2)] == indicator_vars[node1][i] * indicator_vars[node2][i])
    
    phantom_flows = {edge: {} for edge in network.edges}
    for src, i in fixed_nodes.items():
        for dst in network.nodes:
            if src == dst: continue
            if dst in fixed_nodes: continue
            for edge in network.edges:
                phantom_flows[edge][(src, dst)] = model.addVar(vtype=GRB.CONTINUOUS, lb = 0, name=f"{src}_{dst}_{edge}_phantom_flow")
        for dst in network.nodes:
            if src == dst: continue
            if dst in fixed_nodes: continue
            model.addConstr(quicksum(phantom_flows[edge][(src, dst)] for edge in network.nodes[src].outgoing_edges) ==  quicksum(quadratic_indicator_vars[i][(src, dst)] for i in range(num_partitions)))
            model.addConstr(quicksum(phantom_flows[edge][(src, dst)] for edge in network.nodes[src].incoming_edges) ==  0)
            
    for dst in network.nodes:
        for src, i in fixed_nodes.items():
            if src == dst: continue
            if dst in fixed_nodes: continue
            model.addConstr(quicksum(phantom_flows[edge][(src, dst)] for edge in network.nodes[dst].incoming_edges) == quicksum(quadratic_indicator_vars[i][(src, dst)] for i in range(num_partitions)))
            model.addConstr(quicksum(phantom_flows[edge][(src, dst)] for edge in network.nodes[dst].outgoing_edges) ==  0)
    
    for edge in network.edges:
        for src, i in fixed_nodes.items():
            for dst in network.nodes:
                if src == dst: continue
                if dst in fixed_nodes: continue
                model.addConstr(phantom_flows[edge][(src, dst)] <= quicksum(quadratic_indicator_vars[i][(edge[0], edge[1])] for i in range(num_partitions)))

    # flow conservation constraint
    for node in network.nodes:
        for src in fixed_nodes:
            for dst in network.nodes:
                if src == dst: continue
                if node == src: continue 
                if node == dst: continue
                if dst in fixed_nodes: continue
                incoming_flows = [phantom_flows[edge][(src, dst)] for edge in network.nodes[node].incoming_edges]
                outgoing_flows = [phantom_flows[edge][(src, dst)] for edge in network.nodes[node].outgoing_edges]
                model.addConstr(quicksum(incoming_flows) - quicksum(outgoing_flows) == 0)
    # set objective
    model.setObjective(quicksum(real_diffs[(i, j)] for (i, j) in real_diffs), GRB.MINIMIZE)

    # optimize
    model.optimize()

    model.update()

    # extract results
    partition_assignments = {}
    for node in network.nodes:
        for i in range(num_partitions):
            if type(indicator_vars[node][i]) == int:
                if indicator_vars[node][i] == 1:
                    partition_assignments[node] = i
                    break
            else:
                if indicator_vars[node][i].x == 1:
                    partition_assignments[node] = i
                    break

    return partition_assignments, {v.VarName : v.X for v in model.getVars()}