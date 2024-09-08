from flask import Flask, render_template, jsonify, request
import Graph
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from base64 import b64encode
import copy

app = Flask(__name__)

graph = Graph.Graph()

def convert_keys(data):
    if isinstance(data, dict):
        return {int(k) if k.isdigit() else k: convert_keys(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_keys(item) for item in data]
    else:
        return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/setGraph', methods=['POST'])
def set_graph():
    global graph
    data = json.loads(request.data)
    data = convert_keys(data)
    graph_select = data['graphSelect']
    if graph_select == 'Random':
        graph = Graph.Graph()
        graph.generate()
    elif graph_select == 'Tree':
        graph = copy.deepcopy(Graph.tree)
    elif graph_select == 'Comparison':
        graph = copy.deepcopy(Graph.comparison)
    elif graph_select == 'Admissibility':
        graph = copy.deepcopy(Graph.admissibility)
    elif graph_select == 'Extra':
        graph = copy.deepcopy(Graph.extra)
    return jsonify(graph.to_dict())

@app.route('/updateGraph', methods=['POST'])
def update_graph():
    global graph
    data = json.loads(request.data)
    data = convert_keys(data)
    graph.from_dict(data)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    graph.visualize()
    plt.tight_layout()
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_base64 = b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    return jsonify(image=img_base64)

@app.route('/changeView', methods=['POST'])
def change_view():
    global graph
    data = json.loads(request.data)
    data = convert_keys(data)
    graph.from_dict(data)
    return jsonify(graph.to_dict())

@app.route('/nextStep', methods=['POST'])
def next_step():
    global graph
    data = json.loads(request.data)
    data = convert_keys(data)
    graph.from_dict(data)
    graph.search_control()
    return jsonify(graph.to_dict())

@app.route('/resetGraph', methods=['POST'])
def reset_graph():
    global graph
    data = json.loads(request.data)
    data = convert_keys(data)
    graph.from_dict(data)
    graph.reset()
    return jsonify(graph.to_dict())

@app.route('/changeEnds', methods=['POST'])
def change_ends():
    global graph
    data = json.loads(request.data)
    data = convert_keys(data)
    graph.from_dict(data)
    graph.reset()
    return jsonify(graph.to_dict())

if __name__ == '__main__':
    app.run(debug=True)
