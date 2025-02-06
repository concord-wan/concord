from itertools import islice
import sys

class Node:
    def __init__(self, mkt):
        self.mkt = mkt
        self.latitude = None
        self.longitude = None
        self.devices = []
        self.regions = []
        self.incoming_edges = {}
        self.outgoing_edges = {}
        
    def update(self, device=None, region=None, latitude=None, longitude=None):
        if device and device not in self.devices:
            self.devices.append(device)
        if region and region not in self.regions:
            self.regions.append(region)
        if latitude:
            self.latitude = latitude
        if longitude:
            self.longitude = longitude
    
    def add_incoming_edge(self, e, edge):
        self.incoming_edges[e] = edge
    
    def add_outgoing_edge(self, e, edge):
        self.outgoing_edges[e] = edge
            
class Edge:
    #
    # An Edge contains a Graph edge object.
    # and additional attributes.
    # tunnels   - List of tunnels that he edge is part of
    # x_e_t     - Traffic allocation on e for tunnel t
    #
    def __init__(self, e, unity, capacity):
        self.e = e
        self.is_shortcut = False
        self.unity = unity
        self.capacity = capacity
        self.distance = None
        self.tunnels = []
        self.shortcuts = []
        self.x_e_t = {}

    def __repr__(self):
        return f"{self.e}"

    def add_tunnel(self, t):
        assert self.e in [edge.e for edge in t.path]
        if all(t.pathstr != x.pathstr for x in self.tunnels):
            self.tunnels.append(t)

    def add_shortcut(self, s):
        assert self.e in [edge.e for edge in s.path]
        if all(s.pathstr != x.pathstr for x in self.shortcuts):
            self.shortcuts.append(s)
    
    def increment_capacity(self, capacity_increment):
        self.capacity += capacity_increment

    def add_distance(self, distance):
        self.distance = distance
        
    def init_x_e_vars(self, model):
        for idx in range(len(self.tunnels)):
            tunnel = self.tunnels[idx]
            var = model.Variable()
            model.Assert(var >= 0)
            self.x_e_t[tunnel] = var
        return model            
    
class Demand:
    def __init__(self, src, dst, amount):
        self.src = src
        self.dst = dst
        self.amount = amount
        self.tunnels = []
        self.b_d = None

    def __repr__(self):
        return f"({self.src}:{self.dst})"

    def add_tunnel(self, t):
        assert t.pathstr.split(':')[0] == self.src
        assert t.pathstr.split(':')[-1] == self.dst
        if t.pathstr not in [x.pathstr for x in self.tunnels]:
            self.tunnels.append(t)
    
    def change_amount(self, new_amount):
        self.amount = new_amount
            
    def init_b_d(self, model):
        self.b_d = model.Variable()
        model.Assert(self.b_d >= 0)

class Shortcut:
    def __init__(self, path, pathstr, unity, distance):
        self.path = path
        self.pathstr = pathstr
        assert unity > 0
        self.unity = unity
        self.distance = distance
        self.src = path[0].e[0]
        self.src = path[-1].e[1]
        self.w_s = 0
        self.y_s = {}
        # List of tunnels that the shortcut is in
        self.tunnels = []
        for e in path:
            e.add_shortcut(self)

    def name(self):
        return self.pathstr

    def __repr__(self):
        return self.name()
    
    def add_tunnel(self, t):
        assert self.pathstr in t.pathstr
        if all(t.pathstr != x.pathstr for x in self.tunnels):
            self.tunnels.append(t)        

    def init_wavelength_vars(self, model, var=None):
        if not var:
            self.w_s = model.Variable(type="Int")
            model.Assert(self.w_s >= 0)
        else:
            self.w_s = var
        return model
    
    def init_y_s_vars(self, model):
        for idx in range(len(self.tunnels)):
            tunnel = self.tunnels[idx]
            self.y_s[tunnel] = model.Variable()
            model.Assert(self.y_s[tunnel] >= 0)
        return model
        
class Tunnel:
    def __init__(self, path, pathstr):
        # path here is a list of edges
        self.path = path
        self.pathstr = pathstr
        self.weight = 0
        self.shortcuts = []        
        self.v_flow = None    # Solver variable for flow
        for e in path:
            e.add_tunnel(self)
        
    def name(self):
        return self.pathstr

    def __repr__(self):
        return self.name()
    
    def add_weight(self, weight):
        self.weight = weight

    def add_shortcut(self, s):
        self.shortcuts.append(s)
    
    def init_flow_var(self, model):
        self.v_flow = model.addVar(lb = 0, name = self.name())
        # return model

class Network:
    def __init__(self, name):
        self.name = name
        self.nodes = {}
        self.edges = {}
        self.tunnels = {}
        self.demands = {}
        # shortcuts only used for shoofly
        self.shortcuts = {}
        self.graph = None
        
    def add_node(self, mkt, region=None, device=None):
        assert isinstance(mkt, str)
        if mkt in self.nodes:
            node = self.nodes[mkt]
        else:
            node = Node(mkt)
            self.nodes[mkt] = node
        node.update(device=device, region=region)
        return node

    def add_edge(self, mktA, mktB, unity=None, capacity=None):
        assert isinstance(mktA, str)
        assert isinstance(mktB, str)
        self.add_node(mktA)
        self.add_node(mktB)
        if mktA == mktB: return None
        
        if (mktA, mktB) in self.edges:
            edge = self.edges[(mktA, mktB)]
            edge.increment_capacity(capacity)
        else:
            edge = Edge((mktA, mktB), unity, capacity)
            self.edges[(mktA, mktB)] = edge
            self.nodes[mktA].add_outgoing_edge((mktA, mktB), edge)
            self.nodes[mktB].add_incoming_edge((mktA, mktB), edge)
            
        return edge

    def remove_zero_capacity_edges(self):
        edges_to_rm = []
        for edge in self.edges:
            if self.edges[edge].capacity == 0:
                edges_to_rm.append(edge)
        for edge in edges_to_rm:
            self.edges.pop(edge)
                
    def add_demand(self, src, dst, amount, scale=1):
        assert isinstance(src, str)
        assert isinstance(dst, str)
        self.add_node(src)
        self.add_node(dst)
        
        if (src, dst) not in self.demands:
            self.demands[(src, dst)] = Demand(src, dst, amount*scale)

        return self.demands[(src, dst)]

    def add_tunnel(self, tunnel):
        assert isinstance(tunnel, list)
        assert isinstance(tunnel[0], str)
        tunnel_str = ":".join(tunnel)
        if tunnel_str in self.tunnels: return
        
        tunnel_start = tunnel[0]
        tunnel_end = tunnel[-1]
        tunnel_edge_list = []
        for src, dst in zip(tunnel, tunnel[1:]):
            # nodeA = self.add_node(src)
            # nodeB = self.add_node(dst)
            assert (src, dst) in self.edges
            edge = self.edges[(src, dst)]
            tunnel_edge_list.append(edge)

        tunnel_obj = Tunnel(tunnel_edge_list, tunnel_str)
        self.tunnels[tunnel_str] = tunnel_obj        
        if (tunnel_start, tunnel_end) in self.demands:
            demand = self.demands[(tunnel_start, tunnel_end)]
            demand.add_tunnel(tunnel_obj)

    def add_shortcut(self, shortcut, unity, distance):
        assert isinstance(shortcut, list)
        assert isinstance(shortcut[0], str)
        if unity == 0: return
        shortcut_str = ":".join(shortcut)
        # Shortcut is identified by the string of form region1:region2:..:regionN
        if shortcut_str in self.shortcuts:
            return
        shortcut_edge_list = []
        for src, dst in zip(shortcut, shortcut[1:]):
            nodeA = self.add_node(src)
            nodeB = self.add_node(dst)
            assert (src, dst) in self.edges
            edge = self.edges[(src, dst)]
            shortcut_edge_list.append(edge)
            
        shortcut_obj = Shortcut(shortcut_edge_list, shortcut_str, unity, distance)
        self.shortcuts[shortcut_str] = shortcut_obj
        for tunnel_str in self.tunnels:
            if shortcut_str in tunnel_str:
                tunnel_obj = self.tunnels[tunnel_str]
                tunnel_obj.add_shortcut(shortcut_obj)
                shortcut_obj.add_tunnel(tunnel_obj)

        assert shortcut_obj 
        return shortcut_obj
    
    def to_nx(self):
        import networkx
        graph = networkx.DiGraph()
        for n in self.nodes.keys():
            graph.add_node(n)
        # Putting 100 km distance for all edges as of now, fix later.
        for (s,t), obj in self.edges.items():
            graph.add_edge(s, t, weight=obj.capacity)
        return graph
    
    def graph_partitions_kl(self, G):
        import networkx
        undirected = G.to_undirected()
        return networkx.community.kernighan_lin_bisection(undirected, weight='weight')
    
    def modularity_communities(self, G, num_partitions):
        import networkx
        undirected = G.to_undirected()
        return networkx.community.greedy_modularity_communities(undirected, weight='weight', cutoff=num_partitions, best_n=num_partitions)

    def draw(self, labels, color_map=None, seed=None):
        import matplotlib.pyplot as plt
        import networkx as nx
        G = self.to_nx()
        if seed is not None:
            pos = nx.spring_layout(G,seed=seed)
        else:
            pos = nx.spring_layout(G) #, weight=1,
                               # pos={'0':(0,0), '1':(0,1), '2':(1,1), '3':(1,0)}, 
                               # fixed=['0', '1', '2', '3'])
        plt.figure(figsize=(25,20))
        options = {
            'width': 1,
            'arrowstyle': '-|>',
            'arrowsize': 12
        }
        if color_map == None:
            nx.draw(G, pos, edge_color = 'black', linewidths = 3,
                # connectionstyle='arc3, rad = 0.1',
                node_size = 1000, node_color = 'pink',
                alpha = 0.9, with_labels = True, **options)
        else:
            nx.draw(G, pos, edge_color = 'black', linewidths = 3,
                # connectionstyle='arc3, rad = 0.1',
                node_size = 1000, node_color = color_map,
                alpha = 0.9, with_labels = True, **options)
        nx.draw_networkx_edge_labels(G, pos, font_size=10,
                                     label_pos=0.3,
                                     edge_labels=labels)
        ax = plt.gca()
        ax.collections[0].set_edgecolor("#000000")
        plt.axis('off')
        plt.show()

    def k_shortest_paths(self, source, target, k):
        import networkx as nx
        G = self.to_nx()
        return list(islice(nx.shortest_simple_paths(G, source, target), k))

    def k_shortest_edge_disjoint_paths(self, source, target, k):
        import networkx as nx
        G = self.to_nx()
        possibilities = list(nx.edge_disjoint_paths(G, source, target))
        possibilities.sort(key=len)
        return list(islice(possibilities, k))