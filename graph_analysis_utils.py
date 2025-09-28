import networkx as nx
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

class GraphAnalyzer:
    def __init__(self, G, group_fn):
        """
        Parameters:
        - G: networkx.Graph
        - group_fn: function mapping a node to a group label (e.g., d value)
        """
        self.G = G
        self.group_fn = group_fn

    def find_mis_excluding_single_group(self):
        """Return all MISs excluding those where all nodes belong to the same group."""
        complement_G = nx.complement(self.G)
        all_mis = list(nx.find_cliques(complement_G))
        filtered_mis = [sorted(mis) for mis in all_mis if len(set(self.group_fn(node) for node in mis)) > 1]
        return filtered_mis

    def print_mis_excluding_single_group(self):
        mis_list = self.find_mis_excluding_single_group()
        print(f"Total MISs (excluding single-group ones): {len(mis_list)}\n")
        for i, mis in enumerate(mis_list, 1):
            print(f"{i}: {mis}")

    def save_vertex_mis_incidence(self, filename="vertex_mis_incidence.csv"):
        mis_list = self.find_mis_excluding_single_group()
        nodes = sorted(self.G.nodes(), key=lambda node: (self.group_fn(node), node))
        incidence = []
        for mis in mis_list:
            row = [1 if node in mis else 0 for node in nodes]
            incidence.append(row)
        df = pd.DataFrame(incidence, columns=nodes)
        df.to_csv(filename, index=False)
        return df

    def is_perfect(self):
        """Check whether G is perfect by testing for odd holes and antiholes."""
        def is_induced_cycle(H):
            n = H.number_of_nodes()
            m = H.number_of_edges()
            return (
                n >= 4 and
                nx.is_connected(H) and
                m == n and
                all(deg == 2 for node, deg in H.degree())
            )

        def has_odd_hole(H):
            for k in range(5, len(H.nodes) + 1, 2):
                for nodes in combinations(H.nodes, k):
                    subH = H.subgraph(nodes)
                    if is_induced_cycle(subH):
                        return True
            return False

        if has_odd_hole(self.G):
            return False
        if has_odd_hole(nx.complement(self.G)):
            return False
        return True

    def enumerate_cliques_and_check(self, violate_pairwise_fn):
        """
        Enumerate all maximal cliques and check if any pair in a clique violates
        a user-defined pairwise condition. Returns list of (clique, satisfies_condition).

        Parameters:
        - violate_pairwise_fn: function taking two nodes and returning True if the condition is violated.
        """
        cliques = list(nx.find_cliques(self.G))
        result = []
        for idx, clique in enumerate(cliques):
            violated = any(
                violate_pairwise_fn(u, v)
                for u, v in combinations(clique, 2)
            )
            result.append({
                'index': idx + 1,
                'clique': sorted(clique),
                'satisfies_condition': not violated
            })
        return result

    def plot_grouped_on_circle(self, title="Graph", node_size=1000, figsize=(6, 6), cmap_name="Set2", group_gap=np.pi/15):
        """
        Plot the graph on a circle with equal angular spacing between groups and color by group_fn.
        """
        groups = {}
        for node in self.G.nodes:
            key = self.group_fn(node)
            groups.setdefault(key, []).append(node)

        group_keys = list(groups.keys())
        group_keys.sort()
        K = len(group_keys)
        J = max(len(groups[g]) for g in group_keys)
        N = sum(len(groups[g]) for g in group_keys)

        total_gap = K * group_gap
        angle_available = 2 * np.pi - total_gap
        angle_between_points = angle_available / N

        angles = []
        for g in range(K):
            start = g * (J * angle_between_points + group_gap)
            for j in range(len(groups[group_keys[g]])):
                angles.append(start + j * angle_between_points)

        # Calculate positions on the circle with a rotation offset
        angle_offset = -1.15 * np.pi / 2
        x = np.cos(np.array(angles) + angle_offset)
        y = np.sin(np.array(angles) + angle_offset)
        pos = {node: (x[i], y[i]) for i, node in enumerate([n for g in group_keys for n in groups[g]])}

        # Assign colors
        cmap = cm.get_cmap(cmap_name, K)
        group_color = {group_keys[i]: cmap(i) for i in range(K)}
        node_colors = [group_color[self.group_fn(node)] for node in self.G.nodes]

        # Plot
        plt.figure(figsize=figsize)
        nx.draw(self.G, pos, with_labels=True, node_color=node_colors,
                node_size=node_size, edge_color='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()

    @staticmethod
    def build_graph_pairwise(nodes, violate_pairwise_fn, group_fn):
        """
        Build a graph where an edge is added between nodes u and v
        unless violate_pairwise_fn(u, v) is True or group_fn(u) == group_fn(v).

        Parameters:
        - nodes: list of nodes
        - violate_pairwise_fn: function taking two nodes and returning True if the edge should not exist
        - group_fn: function mapping a node to its group (e.g., based on 'd')
        """
        G = nx.Graph()
        G.add_nodes_from(nodes)
        for i, u in enumerate(nodes):
            for v in nodes[i+1:]:
                if group_fn(u) == group_fn(v):
                    continue
                if not violate_pairwise_fn(u, v):
                    G.add_edge(u, v)
        return G
