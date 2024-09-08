import random
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Graph:
    def __init__(self, nodes: dict[dict] = {}, edges:dict[dict] = {}, start:int = 0, finish:int = 0):
        self.nodes = nodes
        self.edges = edges
        self.start = start
        self.finish = finish
        self.reset()

    def reset(self):
        self.started = False
        self.finished = False
        self.fringe = {}
        self.visited_nodes = {}
        self.predecessor = {}
        self.path = []
        self.view_node_id = False

    def generate(self):
        self.reset()
        self.nodes = {}
        self.edges = {}
        num_random_nodes = random.randint(4, 6)
        num_organized_nodes = random.randint(2, 4)
        num_nodes = num_random_nodes + num_organized_nodes
        canvas_size = num_nodes**2
        for i in range(num_random_nodes):
            node_exists = True
            while node_exists:
                x = random.randint(0, canvas_size)
                y = random.randint(0, canvas_size)
                node_exists = False
                for coord in self.nodes.values():
                    distance = ((coord['x'] - x)**2 + (coord['y'] - y)**2)**0.5
                    if distance < 3 * num_nodes:
                        node_exists = True
                        break
            self.nodes[i] = {'x': x, 'y': y}
            self.edges[i] = {}
        for i in range(num_random_nodes, num_nodes):
            node_exists = True
            retry_threshold = 100
            while node_exists:
                retry_threshold -= 1
                if retry_threshold <= 0:
                    self.generate()
                    return
                node_A = random.randint(0, num_random_nodes - 1)
                node_B = random.randint(0, num_random_nodes - 1)
                node_exists = False
                for coord in self.nodes.values():
                    distance = ((coord['x'] - self.nodes[node_A]['x'])**2 + (coord['y'] - self.nodes[node_B]['y'])**2)**0.5
                    if distance < 2 * num_nodes:
                        node_exists = True
                        break
            self.nodes[i] = {'x': self.nodes[node_A]['x'], 'y': self.nodes[node_B]['y']}
            self.edges[i] = {}
        for i in range(num_nodes):
            for j in range(num_nodes):
                self.edges[i][j] = 0
        for i in range(num_nodes):
            option_nodes = {}
            degree = random.randint(1, 4)
            for j in range(num_nodes):
                if i != j:
                    if self.edges[i][j] == 0:
                        option_nodes[j] = round(((self.nodes[i]['x'] - self.nodes[j]['x'])**2 + (self.nodes[i]['y'] - self.nodes[j]['y'])**2)**0.5, 2)
                    else:
                        degree -= 1
            for j in range(degree):
                node_to_connect = min(option_nodes, key=option_nodes.get)
                self.edges[i][node_to_connect] = option_nodes[node_to_connect]
                self.edges[node_to_connect][i] = option_nodes[node_to_connect]
                option_nodes.pop(node_to_connect)
        non_adjacent_pairs = [(i, j) for i, row in self.edges.items() for j, val in row.items() if val == 0 and i != j]
        if not self.is_connected() or not non_adjacent_pairs:
            self.generate()
        else:
            self.start, self.finish = random.choice(non_adjacent_pairs)

    def is_connected(self):
        G = nx.Graph()
        for node, pos in self.nodes.items():
            G.add_node(node, pos=(pos['x'], pos['y']))
        for i, row in self.edges.items():
            for j, val in row.items():
                if val != 0:
                    G.add_edge(i, j)
        return nx.is_connected(G)

    def visualize(self):
        G = nx.Graph()
        for node, pos in self.nodes.items():
            G.add_node(node, pos=(pos['x'], pos['y']))
        for i, row in self.edges.items():
            for j, val in row.items():
                if val != 0:
                    G.add_edge(i, j, weight=val)
        node_labels = {}
        node_colors = []
        border_colors = []
        for node in G.nodes():
            if node in self.visited_nodes:
                node_labels[node] = self.visited_nodes[node]
                node_colors.append("lightblue")
                border_colors.append("lightblue")
            elif node in self.fringe:
                node_labels[node] = self.fringe[node]
                node_colors.append("white")
                border_colors.append("lightblue")
            else:
                node_labels[node] = ""
                node_colors.append("white")
                border_colors.append("darkgrey")
        if self.view_node_id:
            for node in G.nodes():
                node_labels[node] = node
        if self.start in self.visited_nodes.keys():
            node_colors[self.start] = "lightgreen"
            border_colors[self.start] = "lightgreen"
        else:
            node_colors[self.start] = "white"
            border_colors[self.start] = "lightgreen"
        if self.finish in self.visited_nodes.keys():
            node_colors[self.finish] = "lightcoral"
            border_colors[self.finish] = "lightcoral"
        else:
            node_colors[self.finish] = "white"
            border_colors[self.finish] = "lightcoral"
        edge_colors = []
        edge_width = []
        for (u, v) in G.edges():
            if self.finished and ([u, v] in self.path or [v, u] in self.path):
                edge_colors.append('lightgreen')
                edge_width.append(4.0)
            elif (u, v) in self.predecessor.items() or (v, u) in self.predecessor.items():
                edge_colors.append('lightblue')
                edge_width.append(2.0)
            else:
                edge_colors.append('darkgrey')
                edge_width.append(1.0)
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=1000,
                node_color=node_colors, edgecolors=border_colors, edge_color=edge_colors, width=edge_width, font_size=10)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    def search_control(self, algorithm:str, heuristic:str, node_to_expand):
        if not self.finished:
            manual = False
            if algorithm == "Greedy":
                evaluator = self.greedy
            elif algorithm == "Dijkstra":
                evaluator = self.dijkstra
            elif algorithm == "A*":
                evaluator = self.A_star
            else:
                manual = True
                evaluator = self.dijkstra
            if heuristic == "Euclidean":
                heuristic = self.heuristic_euclidean
            elif heuristic == "Manhattan":
                heuristic = self.heuristic_manhattan
            if not self.started:
                self.visited_nodes[self.start] = 0
                self.fringe[self.start] = evaluator(self.start, self.start, heuristic)
                self.predecessor[self.start] = self.start
                self.started = True
            if not manual:
                node_to_expand = min(self.fringe, key=self.fringe.get)
            else:
                if node_to_expand not in self.fringe.keys():
                    return
            self.visited_nodes[node_to_expand] = round(self.visited_nodes[self.predecessor[node_to_expand]] + self.edges[self.predecessor[node_to_expand]][node_to_expand], 2)
            for node in self.nodes.keys():
                if not node in self.visited_nodes.keys() and self.edges[node_to_expand][node] > 0:
                    if not node in self.fringe.keys() or self.fringe[node] > evaluator(node_to_expand, node, heuristic):
                        self.fringe[node] = evaluator(node_to_expand, node, heuristic)
                        self.predecessor[node] = node_to_expand
            self.fringe.pop(node_to_expand)
            if node_to_expand == self.finish:
                self.finished = True
                current_node = self.finish
                while current_node != self.start:
                    self.path.append([self.predecessor[current_node], current_node])
                    current_node = self.predecessor[current_node]

    def greedy(self, node_to_expand:int, node:int, heuristic):
        return round(heuristic(node) + self.edges[node_to_expand][node], 2)

    def dijkstra(self, node_to_expand:int, node:int, heuristic):
        return round(self.visited_nodes[node_to_expand] + self.edges[node_to_expand][node], 2)

    def A_star(self, node_to_expand:int, node:int, heuristic):
        return round(self.visited_nodes[node_to_expand] + self.edges[node_to_expand][node] + heuristic(node), 2)

    def heuristic_euclidean(self, node:int):
        return ((self.nodes[node]['x'] - self.nodes[self.finish]['x'])**2 + (self.nodes[node]['y'] - self.nodes[self.finish]['y'])**2)**0.5

    def heuristic_manhattan(self, node:int):
        return abs(self.nodes[node]['x'] - self.nodes[self.finish]['x']) + abs(self.nodes[node]['y'] - self.nodes[self.finish]['y'])

def compute_edges(edge_map:dict[list], nodes:dict[dict]):
    edges = {}
    for i in nodes.keys():
        for j in nodes.keys():
            if not i in edges:
                edges[i] = {}
            if not j in edges:
                edges[j] = {}
            if j in edge_map[i] and i != j:
                edges[i][j] = round(((nodes[i]['x'] - nodes[j]['x'])**2 + (nodes[i]['y'] - nodes[j]['y'])**2)**0.5, 2)
                edges[j][i] = round(((nodes[i]['x'] - nodes[j]['x'])**2 + (nodes[i]['y'] - nodes[j]['y'])**2)**0.5, 2)
            else:
                edges[i][j] = 0
                edges[j][i] = 0
    return edges

class GUI:
    def __init__(self, master):
        self.master = master
        master.title("Shortest Path Algorithm Demonstration")

        self.graph = Graph()
        self.current_graph = "Tree"
        self.selected_algorithm = tk.StringVar(value="Dijkstra")
        self.selected_heuristic = tk.StringVar(value="Euclidean")
        self.manual_expand = tk.StringVar(value="")
        self.start_node = tk.StringVar(value="")
        self.finish_node = tk.StringVar(value="")

        self.create_widgets()
        self.select_graph()

    def create_widgets(self):
        self.fig, self.ax = plt.subplots(figsize=(7, 7), dpi=50)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        control_frame = ttk.Frame(self.master)
        control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10)

        graph_frame = ttk.LabelFrame(control_frame, text="Graph Selection")
        graph_frame.pack(fill=tk.X, padx=5, pady=5)

        graph_options = ["Tree", "Admissibility", "Comparison", "Random", "Extra"]
        self.graph_combobox = ttk.Combobox(graph_frame, values=graph_options, state="readonly", width=10)
        self.graph_combobox.current(0)
        self.graph_combobox.grid(row=0, column=0, padx=5, sticky=tk.W)

        ttk.Button(graph_frame, text="Set", command=self.select_graph, width=5).grid(row=0, column=1, padx=5, sticky=tk.E)

        ttk.Label(graph_frame, text="Start Node").grid(row=1, column=0, padx=5, sticky=tk.W)
        ttk.Entry(graph_frame, textvariable=self.start_node, width=10).grid(row=1, column=1, padx=5, sticky=tk.E)

        ttk.Label(graph_frame, text="Finish Node").grid(row=2, column=0, padx=5, sticky=tk.W)
        ttk.Entry(graph_frame, textvariable=self.finish_node, width=10).grid(row=2, column=1, padx=5, sticky=tk.E)

        ttk.Button(graph_frame, text="Print Data", command=self.print_graph, width=10).grid(row=3, column=0, padx=5, sticky=tk.W)
        ttk.Button(graph_frame, text="Change Ends", command=self.change_ends, width=10).grid(row=3, column=1, padx=5, sticky=tk.E)

        search_frame = ttk.LabelFrame(control_frame, text="Search Control")
        search_frame.pack(fill=tk.X, padx=5, pady=5)
        search_frame.columnconfigure(0, weight=1)
        search_frame.columnconfigure(1, weight=1)

        ttk.Label(search_frame, text="Algorithm").grid(row=0, column=0, sticky=tk.W, padx=5)
        algorithm_options = ["Dijkstra", "Greedy", "A*", "Manual"]
        ttk.Combobox(search_frame, textvariable=self.selected_algorithm, values=algorithm_options, state="readonly", width=10).grid(row=0, column=1, padx=5, sticky=tk.E)

        ttk.Label(search_frame, text="Heuristic").grid(row=1, column=0, sticky=tk.W, padx=5)
        heuristic_options = ["Euclidean", "Manhattan"]
        ttk.Combobox(search_frame, textvariable=self.selected_heuristic, values=heuristic_options, state="readonly", width=10).grid(row=1, column=1, padx=5, sticky=tk.E)

        ttk.Label(search_frame, text="Node to Expand").grid(row=2, column=0, padx=5, sticky=tk.W)
        ttk.Entry(search_frame, textvariable=self.manual_expand, width=10).grid(row=2, column=1, padx=5, sticky=tk.E)

        ttk.Button(search_frame, text="Next Step", command=self.next_step, width=10).grid(row=3, column=0, sticky=tk.W, padx=5)
        ttk.Button(search_frame, text="Reset", command=self.reset, width=10).grid(row=3, column=1, padx=5, sticky=tk.E)

        view_frame = ttk.LabelFrame(control_frame, text="View Data")
        view_frame.pack(fill=tk.X, padx=5, pady=5)
        view_frame.columnconfigure(0, weight=1)
        view_frame.columnconfigure(1, weight=1)

        ttk.Button(view_frame, text="Node ID", command=self.view_node_id, width=10).grid(row=0, column=0, padx=5, sticky=tk.W)
        ttk.Button(view_frame, text="Evaluation", command=self.view_fringe_data, width=10).grid(row=0, column=1, padx=5, sticky=tk.E)

    def select_graph(self):
        self.current_graph = self.graph_combobox.get()
        if self.current_graph == "Tree":
            self.graph = tree
        elif self.current_graph == "Admissibility":
            self.graph = admissibility
        elif self.current_graph == "Comparison":
            self.graph = comparison
        elif self.current_graph == "Extra":
            self.graph = extra
        else:
            self.graph = Graph()
            self.graph.generate()
        self.update_graph_display()

    def next_step(self):
        algorithm = self.selected_algorithm.get()
        heuristic = self.selected_heuristic.get()
        try: node_to_expand = int(self.manual_expand.get())
        except: node_to_expand = 0
        self.graph.search_control(algorithm, heuristic, node_to_expand)
        self.update_graph_display()

    def reset(self):
        self.graph.reset()
        self.update_graph_display()

    def view_node_id(self):
        self.graph.view_node_id = True
        self.update_graph_display()

    def view_fringe_data(self):
        self.graph.view_node_id = False
        self.update_graph_display()

    def print_graph(self):
        print(self.graph.nodes, self.graph.edges, self.graph.start, self.graph.finish)

    def change_ends(self):
        try:
            start = int(self.start_node.get())
            finish = int(self.finish_node.get())
            assert start in self.graph.nodes
            assert finish in self.graph.nodes
            self.graph.start = start
            self.graph.finish = finish
            self.update_graph_display()
        except:
            pass

    def update_graph_display(self):
        self.ax.clear()
        self.graph.visualize()
        self.canvas.draw()

if __name__ == "__main__":
    tree = Graph(
        {0: {'x': 36, 'y': 26}, 1: {'x': 79, 'y': 5}, 2: {'x': 51, 'y': 2}, 3: {'x': 6, 'y': 5}, 4: {'x': 11, 'y': 78}, 5: {'x': 49, 'y': 71}, 6: {'x': 79, 'y': 71}, 7: {'x': 79, 'y': 26}, 8: {'x': 11, 'y': 26}},
        {0: {0: 0, 1: 47.85, 2: 0, 3: 36.62, 4: 57.7, 5: 46.84, 6: 62.24, 7: 0, 8: 25.0}, 1: {0: 47.85, 1: 0, 2: 28.16, 3: 0, 4: 0, 5: 0, 6: 0, 7: 21.0, 8: 0}, 2: {0: 0, 1: 28.16, 2: 0, 3: 45.1, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}, 3: {0: 36.62, 1: 0, 2: 45.1, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 21.59}, 4: {0: 57.7, 1: 0, 2: 0, 3: 0, 4: 0, 5: 38.64, 6: 68.36, 7: 0, 8: 52.0}, 5: {0: 46.84, 1: 0, 2: 0, 3: 0, 4: 38.64, 5: 0, 6: 30.0, 7: 54.08, 8: 0}, 6: {0: 62.24, 1: 0, 2: 0, 3: 0, 4: 68.36, 5: 30.0, 6: 0, 7: 45.0, 8: 0}, 7: {0: 0, 1: 21.0, 2: 0, 3: 0, 4: 0, 5: 54.08, 6: 45.0, 7: 0, 8: 0}, 8: {0: 25.0, 1: 0, 2: 0, 3: 21.59, 4: 52.0, 5: 0, 6: 0, 7: 0, 8: 0}},
        3,
        6
    )
    comparison = Graph(
        {0: {'x': 23, 'y': 24}, 1: {'x': 19, 'y': 10}, 2: {'x': 31, 'y': 34}, 3: {'x': 1, 'y': 23}, 4: {'x': 1, 'y': 10}, 5: {'x': 31, 'y': 10}},
        {0: {0: 0, 1: 14.56, 2: 12.81, 3: 22.02, 4: 0, 5: 16.12}, 1: {0: 14.56, 1: 0, 2: 26.83, 3: 22.2, 4: 18.0, 5: 12.0}, 2: {0: 12.81, 1: 26.83, 2: 0, 3: 31.95, 4: 0, 5: 24.0}, 3: {0: 22.02, 1: 22.2, 2: 31.95, 3: 0, 4: 13.0, 5: 0}, 4: {0: 0, 1: 18.0, 2: 0, 3: 13.0, 4: 0, 5: 0}, 5: {0: 16.12, 1: 12.0, 2: 24.0, 3: 0, 4: 0, 5: 0}},
        2,
        4
    )
    admissibility = Graph(
        {0: {'x': 73, 'y': 67}, 1: {'x': 17, 'y': 3}, 2: {'x': 99, 'y': 38}, 3: {'x': 40, 'y': 11}, 4: {'x': 73, 'y': 11}, 5: {'x': 17, 'y': 67}, 6: {'x': 17, 'y': 38}, 7: {'x': 40, 'y': 38}, 8: {'x': 99, 'y': 3}, 9: {'x': 99, 'y': 67}, 10: {'x': 73, 'y': 38}},
        {0: {0: 0, 1: 0, 2: 38.95, 3: 0, 4: 0, 5: 56.0, 6: 0, 7: 43.93, 8: 0, 9: 26.0, 10: 29.0}, 1: {0: 0, 1: 0, 2: 0, 3: 24.35, 4: 0, 5: 0, 6: 35.0, 7: 0, 8: 0, 9: 0, 10: 0}, 2: {0: 38.95, 1: 0, 2: 0, 3: 0, 4: 37.48, 5: 0, 6: 0, 7: 0, 8: 35.0, 9: 29.0, 10: 26.0}, 3: {0: 0, 1: 24.35, 2: 0, 3: 0, 4: 33.0, 5: 60.54, 6: 0, 7: 27.0, 8: 59.54, 9: 0, 10: 0}, 4: {0: 0, 1: 0, 2: 37.48, 3: 33.0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 27.2, 9: 0, 10: 27.0}, 5: {0: 56.0, 1: 0, 2: 0, 3: 60.54, 4: 0, 5: 0, 6: 29.0, 7: 37.01, 8: 0, 9: 0, 10: 0}, 6: {0: 0, 1: 35.0, 2: 0, 3: 0, 4: 0, 5: 29.0, 6: 0, 7: 23.0, 8: 0, 9: 0, 10: 0}, 7: {0: 43.93, 1: 0, 2: 0, 3: 27.0, 4: 0, 5: 37.01, 6: 23.0, 7: 0, 8: 0, 9: 0, 10: 0}, 8: {0: 0, 1: 0, 2: 35.0, 3: 59.54, 4: 27.2, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 43.6}, 9: {0: 26.0, 1: 0, 2: 29.0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}, 10: {0: 29.0, 1: 0, 2: 26.0, 3: 0, 4: 27.0, 5: 0, 6: 0, 7: 0, 8: 43.6, 9: 0, 10: 0}},
        1,
        9
    )
    extra = Graph(
        {0: {'x': 52, 'y': 62}, 1: {'x': 76, 'y': 26}, 2: {'x': 96, 'y': 85}, 3: {'x': 19, 'y': 13}, 4: {'x': 16, 'y': 68}, 5: {'x': 8, 'y': 97}, 6: {'x': 52, 'y': 13}, 7: {'x': 96, 'y': 62}, 8: {'x': 96, 'y': 26}, 9: {'x': 76, 'y': 62}},
        {0: {0: 0, 1: 43.27, 2: 49.65, 3: 0, 4: 36.5, 5: 56.22, 6: 49.0, 7: 0, 8: 0, 9: 24.0}, 1: {0: 43.27, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 27.29, 7: 41.18, 8: 0, 9: 0}, 2: {0: 49.65, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 23.0, 8: 59.0, 9: 30.48}, 3: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 33.0, 7: 0, 8: 0, 9: 0}, 4: {0: 36.5, 1: 0, 2: 0, 3: 0, 4: 0, 5: 30.08, 6: 0, 7: 0, 8: 0, 9: 0}, 5: {0: 56.22, 1: 0, 2: 0, 3: 0, 4: 30.08, 5: 0, 6: 0, 7: 0, 8: 0, 9: 76.48}, 6: {0: 49.0, 1: 27.29, 2: 0, 3: 33.0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 45.88, 9: 0}, 7: {0: 0, 1: 41.18, 2: 23.0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 36.0, 9: 20.0}, 8: {0: 0, 1: 0, 2: 59.0, 3: 0, 4: 0, 5: 0, 6: 45.88, 7: 36.0, 8: 0, 9: 0}, 9: {0: 24.0, 1: 0, 2: 30.48, 3: 0, 4: 0, 5: 76.48, 6: 0, 7: 20.0, 8: 0, 9: 0}},
        5,
        8
    )
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()