from gurobipy import *
from util import *

def get_demand_loss_objective_weights(flows, network):
    objective = 0
    for demand_id, flows_over_tunnels in flows.items():
        objective += network.demands[demand_id].amount
        for flow in flows_over_tunnels.values():
            objective -= (network.demands[demand_id].amount * flow)
    return objective

def edge_capacity_map(network, capacity_mult=1):
    edge_capacities = {}
    for e, edge in network.edges.items():
        edge_capacities[e] = capacity_mult * int(edge.capacity)
    return edge_capacities

def init_flow_vars(model, network, tunnel_ids, blastshield=False):
    flows = {}
    lower_bound = 0
    if blastshield:
        lower_bound = 1e-4
    for demand_id, demand in network.demands.items():
        tunnel_flows = {}
        for tunnel in demand.tunnels:
            tunnel_flows[tunnel_ids[tunnel.name()]] = model.addVar(lb = lower_bound, name = f"{tunnel_ids[tunnel.name()]}")
        flows[demand_id] = tunnel_flows
    
    return flows

def demand_constraint_weights(model, flows, network, tunnel_ids):
    for demand_id, demand in network.demands.items():
        flow_on_tunnels = sum([flows[demand_id][tunnel_ids[tunnel.name()]] for tunnel in demand.tunnels])
        model.addConstr(flow_on_tunnels <= 1)

def demand_constraint_equals_weights(model, flows, network, tunnel_ids):
    for demand_id, demand in network.demands.items():
        flow_on_tunnels = sum([flows[demand_id][tunnel_ids[tunnel.name()]] for tunnel in demand.tunnels])
        model.addConstr(flow_on_tunnels == 1)

def demand_constraint_mcf(model, flows, mcf, network, tunnel_ids):
    for demand_id, demand in network.demands.items():
        flow_on_tunnels = sum([flows[demand_id][tunnel_ids[tunnel.name()]] for tunnel in demand.tunnels])
        model.addConstr(flow_on_tunnels == mcf * demand.amount)
    
def demand_constraint_mcf_weights(model, flows, mcf, network, tunnel_ids):
    for demand_id, demand in network.demands.items():
        flow_on_tunnels = sum([flows[demand_id][tunnel_ids[tunnel.name()]] for tunnel in demand.tunnels])
        model.addConstr(flow_on_tunnels == mcf)

def capacity_constraint_weights(model, flows, network, edge_capacities, tunnel_ids):
    for edge in network.edges.values():
        flow_on_tunnels = 0
        for tunnel in edge.tunnels:
            for demand_id, tunnel_list in flows.items():
                if tunnel_ids[tunnel.name()] in tunnel_list:
                    flow_on_tunnels += (flows[demand_id][tunnel_ids[tunnel.name()]] * network.demands[demand_id].amount)
        assert flow_on_tunnels is not None
        assert edge.capacity is not None
        model.addConstr(flow_on_tunnels <= edge_capacities[(edge.e[0], edge.e[1])])

def solve_vanilla_mcf(network, tunnel_ids, blastshield=False, method=-1, capacity_mult=1):
    model = Model("vanilla_mcf")

    model.setParam("Method", method)

    edge_capacities = edge_capacity_map(network, capacity_mult)

    mcf = model.addVar(lb = 0, ub = 1, name = "mcf")

    flows = init_flow_vars(model, network, tunnel_ids, blastshield=blastshield)

    # demand constraints
    demand_constraint_mcf_weights(model, flows, mcf, network, tunnel_ids)
    
    # capacity constraints
    capacity_constraint_weights(model, flows, network, edge_capacities, tunnel_ids)

    model.setObjective(mcf, GRB.MAXIMIZE)
    model.optimize()
    model.update()
    return {v.VarName : v.X for v in model.getVars()}

def solve_concord_mcf(network, tunnel_ids, dropout=set(), reg_param=1, blastshield=False):
    model = Model("concord_mcf")

    edge_capacities = edge_capacity_map(network)

    mcf = model.addVar(lb = 0, ub = 1, name = "mcf")

    flows = init_flow_vars(model, network, tunnel_ids, blastshield=blastshield)

    objective = mcf

    for e, edge in network.edges.items():
        if e in dropout: continue
        flow_on_tunnels = 0
        u = model.addVar(lb = 0, name = f"u{edge.e[0]}-{edge.e[1]}")
        for tunnel in edge.tunnels:
            for demand_id, tunnel_list in flows.items():
                if tunnel_ids[tunnel.name()] in tunnel_list:
                    flow_on_tunnels += (flows[demand_id][tunnel_ids[tunnel.name()]] * network.demands[demand_id].amount)
        model.addConstr(u == flow_on_tunnels/edge.capacity)
        objective -= (reg_param * u * u)

    # demand constraints
    demand_constraint_mcf_weights(model, flows, mcf, network, tunnel_ids)
    
    # capacity constraints
    capacity_constraint_weights(model, flows, network, edge_capacities, tunnel_ids)

    model.setObjective(objective, GRB.MAXIMIZE)
    model.optimize()
    model.update()
    return {v.VarName : v.X for v in model.getVars()}

def solve_vanilla_max_flow(network, tunnel_ids, blastshield=False, method=-1, capacity_mult=1):
    model = Model("vanilla_mt")
    if method >= 0:
        model.setParam("Method", method)

    edge_capacities = edge_capacity_map(network, capacity_mult)

    flows = init_flow_vars(model, network, tunnel_ids, blastshield=blastshield)

    # demand constraints
    demand_constraint_weights(model, flows, network, tunnel_ids)
    
    capacity_constraint_weights(model, flows, network, edge_capacities, tunnel_ids)

    objective = get_demand_loss_objective_weights(flows, network)
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()
    model.update()
    return {v.VarName : v.X for v in model.getVars()}


def solve_concord_max_flow(network, tunnel_ids, dropout=set(), reg_param=1, blastshield=False):
    model = Model("concord_mt")

    edge_capacities = edge_capacity_map(network)

    flows = init_flow_vars(model, network, tunnel_ids, blastshield=blastshield)

    # demand constraints
    demand_constraint_weights(model, flows, network, tunnel_ids)
    
    # capacity constraints
    capacity_constraint_weights(model, flows, network, edge_capacities, tunnel_ids)

    objective = get_demand_loss_objective_weights(flows, network)
    for e, edge in network.edges.items():
        if e in dropout: continue
        flow_on_tunnels = 0
        u = model.addVar(lb = 0, name = f"u{edge.e[0]}-{edge.e[1]}")
        for tunnel in edge.tunnels:
            for demand_id, tunnel_list in flows.items():
                if tunnel_ids[tunnel.name()] in tunnel_list:
                    flow_on_tunnels += (flows[demand_id][tunnel_ids[tunnel.name()]] * network.demands[demand_id].amount)
        model.addConstr(u == flow_on_tunnels/edge.capacity)
        objective += (reg_param * u * u)
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()
    model.update()
    return {v.VarName : v.X for v in model.getVars()}

def solve_vanilla_min_mlu(network, tunnel_ids, blastshield=False, method=-1):
    model = Model("vanilla_min_mlu")
    if method > 0:
        model.setParam("Method", method)

    flows = init_flow_vars(model, network, tunnel_ids, blastshield=blastshield)

    # demand constraints
    demand_constraint_equals_weights(model, flows, network, tunnel_ids)

    z = model.addVar(lb = 0, name = "Z")

    for edge in network.edges.values():
        flow_on_tunnels = 0
        u = model.addVar(lb = 0, name = f"u{edge.e[0]}-{edge.e[1]}")
        for tunnel in edge.tunnels:
            for demand_id, tunnel_list in flows.items():
                if tunnel_ids[tunnel.name()] in tunnel_list:
                    flow_on_tunnels += (flows[demand_id][tunnel_ids[tunnel.name()]] * network.demands[demand_id].amount)
        model.addConstr(u == flow_on_tunnels/edge.capacity)
        model.addConstr(u <= z)
    model.setObjective(z, GRB.MINIMIZE)
    model.optimize()
    model.update()
    return {v.VarName : v.X for v in model.getVars()}

def solve_concord_min_mlu(network, tunnel_ids, dropout=set(), reg_param=1, blastshield=False):
    model = Model("concord_min_mlu")

    flows = init_flow_vars(model, network, tunnel_ids, blastshield=blastshield)

    demand_constraint_equals_weights(model, flows, network, tunnel_ids)

    z = model.addVar(lb = 0, name = "Z")

    objective = z
    for e, edge in network.edges.items():
        flow_on_tunnels = 0
        u = model.addVar(lb = 0, name = f"u{edge.e[0]}-{edge.e[1]}")
        for tunnel in edge.tunnels:
            for demand_id, tunnel_list in flows.items():
                if tunnel_ids[tunnel.name()] in tunnel_list:
                    weight = flows[demand_id][tunnel_ids[tunnel.name()]]
                    flow_on_tunnels += (weight * network.demands[demand_id].amount)
        model.addConstr(u == flow_on_tunnels/edge.capacity)
        model.addConstr(u <= z)
        if e not in dropout:
            objective += (reg_param * u * u)
    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()
    model.update()
    return {v.VarName : v.X for v in model.getVars()}
