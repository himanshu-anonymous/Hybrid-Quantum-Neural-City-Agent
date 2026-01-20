import matplotlib.pyplot as plt
import networkx as nx

class CityVisualizer:
    def __init__(self, topology):
        self.topology = topology
        self.pos = nx.spring_layout(topology.G, seed=42)
        plt.ion()
        self.fig, (self.ax_map, self.ax_graph) = plt.subplots(1, 2, figsize=(15, 6))
        self.reward_history = []

    def update(self, path, episode, reward):
        self.reward_history.append(reward)
        self.ax_map.clear()
        nx.draw_networkx_nodes(self.topology.G, self.pos, node_color='lightgray', node_size=600, ax=self.ax_map)
        nx.draw_networkx_labels(self.topology.G, self.pos, font_size=8, ax=self.ax_map)
        nx.draw_networkx_edges(self.topology.G, self.pos, edge_color='gray', style='dotted', ax=self.ax_map)
        
    
        nx.draw_networkx_nodes(self.topology.G, self.pos, nodelist=[self.topology.start_node], node_color='green', ax=self.ax_map)
        nx.draw_networkx_nodes(self.topology.G, self.pos, nodelist=[self.topology.end_node], node_color='gold', ax=self.ax_map)
        nx.draw_networkx_nodes(self.topology.G, self.pos, nodelist=self.topology.danger_zones, node_color='red', ax=self.ax_map)
        
        if len(path) > 1:
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(self.topology.G, self.pos, edgelist=path_edges, edge_color='blue', width=3, ax=self.ax_map)
        
        self.ax_map.set_title(f"Episode {episode}: Pathfinding")
        self.ax_graph.clear()
        self.ax_graph.plot(self.reward_history, color='blue', linewidth=1)
        self.ax_graph.set_title("Learning Progress (Total Reward per Episode)")
        self.ax_graph.set_xlabel("Episode")
        self.ax_graph.set_ylabel("Total Reward")
        
        # Add a horizontal line at 0 for reference
        self.ax_graph.axhline(0, color='black', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.pause(0.05)
