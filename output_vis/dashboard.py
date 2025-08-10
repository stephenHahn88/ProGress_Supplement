#!/usr/bin/env python3
import os
import sys
import networkx as nx
import panel as pn
import param
from pyvis.network import Network

pn.extension()

def parse_single_graph(lines):
    try:
        N = int(lines[0].split("=")[1])
    except Exception as e:
        print("Error reading the number of nodes:", e)
        sys.exit(1)
    try:
        x_index = lines.index("X:") + 1
        e_index = lines.index("E:")
    except ValueError as e:
        print("Error: File format must include 'X:' and 'E:' markers.", e)
        sys.exit(1)
    
    x_values = []
    for line in lines[x_index:e_index]:
        x_values.extend(line.split())
    if len(x_values) != N:
        print(f"Error: Expected {N} node values, found {len(x_values)}")
        sys.exit(1)
    node_values = [int(val) for val in x_values]
    
    edge_numbers = []
    for line in lines[e_index+1:]:
        edge_numbers.extend(line.split())
    if len(edge_numbers) != N * N:
        print(f"Error: Expected {N * N} edge values, found {len(edge_numbers)}")
        sys.exit(1)
    matrix = []
    for i in range(N):
        row = [int(edge_numbers[i * N + j]) for j in range(N)]
        matrix.append(row)
    
    return N, node_values, matrix

def parse_graphs_file(filename):
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    graphs_data = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("N="):
            current_chunk = []
            while i < len(lines) and (not (lines[i].startswith("N=") and current_chunk)):
                current_chunk.append(lines[i])
                i += 1
            graphs_data.append(parse_single_graph(current_chunk))
        else:
            i += 1
    return graphs_data

def build_graph(N, node_values, matrix):
    ENCODED_SCALE_DEGREES = [
        'A2', 'm2', 'P8', 'A6',
        'm7', 'M2', 'm6', 'M7',
        'm3', 'M3', 'P5', 'd7',
        'P4', 'M6', 'A4', 'd5'
    ]
    G = nx.DiGraph()
    for i, val in enumerate(node_values):
        label = "Rest" if val == 51 else ENCODED_SCALE_DEGREES[val]
        G.add_node(i, label=label)
    for i in range(N):
        for j in range(N):
            w = matrix[i][j]
            if w != 0:
                G.add_edge(i, j, weight=w)
    return G

def generate_network_html(graph_index, threshold, graphs_data):
    """
    Build a filtered graph, then generate a COMPLETE HTML string from PyVis
    using `generate_html(notebook=False)`. 
    """
    N, node_values, matrix = graphs_data[graph_index]
    G = build_graph(N, node_values, matrix)
    
    # Filter graph
    H = nx.DiGraph()
    for node, data in G.nodes(data=True):
        H.add_node(node, **data)
    for u, v, data in G.edges(data=True):
        if data['weight'] >= threshold:
            H.add_edge(u, v, **data)
    
    # Create the PyVis network. Use remote CDN resources for best compatibility.
    net = Network(height="600px", width="100%", directed=True, cdn_resources='in_line')

    net.from_nx(H)
    
    # Instead of net.show(...), generate the full HTML string:
    html_content = net.generate_html(notebook=False)
    return html_content

class Dashboard(param.Parameterized):
    graph_index = param.Integer(default=0, bounds=(0, 0))
    threshold = param.Integer(default=0, bounds=(0, 30))
    
    @pn.depends('graph_index', 'threshold')
    def view_network(self):
        html_content = generate_network_html(self.graph_index, self.threshold, graphs_data)
        # Save the PyVis graph to a separate HTML file.
        temp_file = "pyvis_graph.html"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        # Embed that file in an iframe.
        iframe_html = f'<iframe src="{temp_file}" width="100%" height="600px" frameBorder="0"></iframe>'
        return pn.pane.HTML(iframe_html, sizing_mode="stretch_both")

if __name__ == "__main__":
    filename = "generated_samples1.txt"  # Adjust as needed
    graphs_data = parse_graphs_file(filename)
    num_graphs = len(graphs_data)
    if num_graphs == 0:
        print("No graphs found in the file.")
        sys.exit(1)
    
    dashboard = Dashboard()
    dashboard.param.graph_index.bounds = (0, num_graphs - 1)
    
    # Layout: left = param widgets, right = PyVis HTML
    dashboard_panel = pn.Row(
        pn.Param(dashboard, parameters=['graph_index', 'threshold'], widgets={
            'graph_index': pn.widgets.IntSlider, 
            'threshold': pn.widgets.IntSlider
        }),
        dashboard.view_network
    )
    
    # Save as a standalone HTML with embedded widgets and remote JS resources
    output_html = "dashboard.html"
    pn.serve(dashboard_panel, port=5006, show=True)
    print("Dashboard saved as", output_html)
