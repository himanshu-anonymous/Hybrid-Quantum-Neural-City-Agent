import networkx as nx
import random

class NetworkTopology:
    """A real-world graph-based map with DYNAMIC traffic."""
    def __init__(self):
        self.G = nx.Graph()
        self._build_network()
        
        self.start_node = "Warehouse_A"
        self.start_pos = self.start_node
        self.end_node = "Delivery_Hub"
        
        # Initial Danger Zones
        self.danger_zones = ["Intersection_4", "Bridge_East"]

    def _build_network(self):
        edges = [
            ("Warehouse_A", "Intersection_1", 2),
            ("Warehouse_A", "Intersection_2", 5),
            ("Intersection_1", "Intersection_3", 3),
            ("Intersection_2", "Intersection_3", 1),
            ("Intersection_2", "Intersection_4", 10),
            ("Intersection_3", "Bridge_East", 4),
            ("Intersection_4", "Bridge_East", 2),
            ("Bridge_East", "Delivery_Hub", 3)
        ]
        self.G.add_weighted_edges_from(edges)

    def update_traffic(self):
        """NEW: Randomly moves danger zones to simulate dynamic traffic."""
        all_nodes = list(self.G.nodes())
        all_nodes.remove(self.start_node)
        all_nodes.remove(self.end_node)
        # Randomly select 2 nodes to be 'congested' this episode
        self.danger_zones = random.sample(all_nodes, 2)

    def get_valid_moves(self, current_node):
        return list(self.G.neighbors(current_node))

    def is_end(self, node):
        return node == self.end_node