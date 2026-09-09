import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from neuprint import Client, fetch_neurons, fetch_adjacencies

# Connect
client = Client(
    "neuprint.janelia.org",
    dataset="male-cns:v1.0",
    token=os.environ["NEUPRINT_APPLICATION_CREDENTIALS"],
)

# Start with this known neuron type
NEURON_TYPE = "DNge104"

# Find matching neurons
neurons, roi_info = fetch_neurons(NEURON_TYPE)

print("\n=== Neurons ===")
print(
    neurons[
        ["bodyId", "type", "instance", "pre", "post"]
    ].to_string(index=False)
)

# Find downstream connections
neuron_info, connections = fetch_adjacencies(
    sources=NEURON_TYPE,
    targets=None,
    min_total_weight=10,
)

# Connections can appear once per ROI, so combine them
edges = (
    connections
    .groupby(["bodyId_pre", "bodyId_post"], as_index=False)
    ["weight"]
    .sum()
    .sort_values("weight", ascending=False)
)

print("\n=== Strongest downstream connections ===")
print(edges.head(20).to_string(index=False))

# Make graph manageable
edges = edges.head(30)

G = nx.DiGraph()

for _, edge in edges.iterrows():
    G.add_edge(
        int(edge["bodyId_pre"]),
        int(edge["bodyId_post"]),
        weight=int(edge["weight"]),
    )

# Draw
plt.figure(figsize=(14, 10))

pos = nx.spring_layout(G, seed=42)

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=700,
)

nx.draw_networkx_edges(
    G,
    pos,
    arrows=True,
    arrowsize=18,
)

nx.draw_networkx_labels(
    G,
    pos,
    font_size=7,
)

edge_labels = {
    (a, b): d["weight"]
    for a, b, d in G.edges(data=True)
}

nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    font_size=6,
)

plt.title(f"Male CNS connectivity — {NEURON_TYPE}")
plt.axis("off")
plt.tight_layout()
plt.show()
