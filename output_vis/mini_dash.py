#!/usr/bin/env python3
import os
import sys
import networkx as nx
import panel as pn
import param
from pyvis.network import Network

pn.extension()

def minimal_pyvis_html():
    """Create a tiny PyVis network with two nodes and one edge, return HTML string."""
    net = Network(
        height="400px",
        width="50%",
        directed=True,
        cdn_resources='in_line'  # <--- embed all JS in the HTML
    )
    net.add_node(0, label="Hello")
    net.add_node(1, label="World")
    net.add_edge(0, 1)
    
    # Instead of net.show(), we do:
    html = net.generate_html(notebook=False)
    return html

class MinimalDashboard(param.Parameterized):
    @pn.depends()
    def view_network(self):
        return pn.pane.HTML(minimal_pyvis_html(), sizing_mode="stretch_both")

if __name__ == "__main__":
    dash = MinimalDashboard()
    layout = pn.Column("## Minimal Test", dash.view_network)
    layout.save("minimal_test.html", embed=True)
    print("Saved minimal_test.html")
