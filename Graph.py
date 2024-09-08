import random
import networkx as nx

class Graph:
    def __init__(self, nodes: dict[dict] = {}, edges:dict[dict] = {}, start:int = 0, finish:int = 0):
        self.nodes = nodes
        self.edges = edges
        self.start = start
        self.finish = finish
        self.view_node_id = False
        self.reset()

    def reset(self):
        self.started = False
        self.finished = False
        self.fringe = {}
        self.visited_nodes = {}
        self.predecessor = {}
        self.path = []
        self.node_to_expand = self.start
        self.algorithm = "Dijkstra"
        self.heuristic = "Euclidean"

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
        self.node_to_expand = self.start

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
                node_colors.append("#a8d6ff")
                border_colors.append("#a8d6ff")
            elif node in self.fringe:
                node_labels[node] = self.fringe[node]
                node_colors.append("white")
                border_colors.append("#a8d6ff")
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
                edge_colors.append('#a8d6ff')
                edge_width.append(2.0)
            else:
                edge_colors.append('darkgrey')
                edge_width.append(1.0)
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=1000,
                node_color=node_colors, edgecolors=border_colors, edge_color=edge_colors, width=edge_width, font_size=10)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    def to_dict(self):
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'start': self.start,
            'finish': self.finish,
            'visited_nodes': self.visited_nodes,
            'fringe': self.fringe,
            'path': self.path,
            'predecessor': self.predecessor,
            'started': self.started,
            'finished': self.finished,
            'view_node_id': self.view_node_id,
            'node_to_expand': self.node_to_expand,
            'algorithm': self.algorithm,
            'heuristic': self.heuristic
        }

    def from_dict(self, data):
        self.nodes = data['nodes']
        self.edges = data['edges']
        self.start = data['start']
        self.finish = data['finish']
        self.visited_nodes = data['visited_nodes']
        self.fringe = data['fringe']
        self.path = data['path']
        self.predecessor = data['predecessor']
        self.started = data['started']
        self.finished = data['finished']
        self.view_node_id = data['view_node_id']
        self.node_to_expand = data['node_to_expand']
        self.algorithm = data['algorithm']
        self.heuristic = data['heuristic']

    def search_control(self):
        if self.algorithm == "Greedy":
            evaluator = self.greedy
        elif self.algorithm == "Dijkstra":
            evaluator = self.dijkstra
        elif self.algorithm == "A*":
            evaluator = self.A_star
        else:
            return
        if self.heuristic == "Euclidean":
            heuristic = self.heuristic_euclidean
        elif self.heuristic == "Manhattan":
            heuristic = self.heuristic_manhattan
        else:
            return
        if not self.started:
            self.visited_nodes[self.start] = 0
            self.fringe[self.start] = evaluator(self.start, heuristic)
            self.predecessor[self.start] = self.start
            self.started = True
            self.node_to_expand = min(self.fringe, key=self.fringe.get)
            return
        if self.started and not self.finished:
            self.visited_nodes[self.node_to_expand] = round(self.visited_nodes[self.predecessor[self.node_to_expand]] + self.edges[self.predecessor[self.node_to_expand]][self.node_to_expand], 2)
            for node in self.nodes.keys():
                if not node in self.visited_nodes.keys() and self.edges[self.node_to_expand][node] > 0:
                    if not node in self.fringe.keys() or self.fringe[node] > evaluator(node, heuristic):
                        self.fringe[node] = evaluator(node, heuristic)
                        self.predecessor[node] = self.node_to_expand
            self.fringe.pop(self.node_to_expand)
            if self.node_to_expand == self.finish:
                self.finished = True
                current_node = self.finish
                while current_node != self.start:
                    self.path.append([self.predecessor[current_node], current_node])
                    current_node = self.predecessor[current_node]
                return
            self.node_to_expand = min(self.fringe, key=self.fringe.get)
            return

    def greedy(self, node:int, heuristic):
        return round(heuristic(node) + self.edges[self.node_to_expand][node], 2)

    def dijkstra(self, node:int, heuristic):
        return round(self.visited_nodes[self.node_to_expand] + self.edges[self.node_to_expand][node], 2)

    def A_star(self, node:int, heuristic):
        return round(self.visited_nodes[self.node_to_expand] + self.edges[self.node_to_expand][node] + heuristic(node), 2)

    def heuristic_euclidean(self, node:int):
        return ((self.nodes[node]['x'] - self.nodes[self.finish]['x'])**2 + (self.nodes[node]['y'] - self.nodes[self.finish]['y'])**2)**0.5

    def heuristic_manhattan(self, node:int):
        return abs(self.nodes[node]['x'] - self.nodes[self.finish]['x']) + abs(self.nodes[node]['y'] - self.nodes[self.finish]['y'])

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