import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph for AI applications in water treatment
G = nx.DiGraph()

# Central node
G.add_node("AI Technology")

# Define application areas and corresponding methods
application_areas = {
    "Water Quality Monitoring": ["Sensor Data Analysis", "Real-time Prediction"],
    "Contaminant Removal": ["Image Recognition", "Process Control"],
    "Process Optimization": ["Energy Optimization", "Control System"],
    "Sludge Management": ["Sludge Prediction", "Treatment Optimization"],
    "Smart Maintenance": ["Fault Prediction", "Maintenance Optimization"]
}

# Add edges from the central node to application areas and from areas to methods
for area, methods in application_areas.items():
    G.add_edge("AI Technology", area)
    for method in methods:
        G.add_edge(area, method)

# Layout for clear visualization
pos = nx.spring_layout(G, seed=42)

# Drawing the graph with clear labels
plt.figure(figsize=(14, 10))
nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="skyblue", edgecolors="black")
nx.draw_networkx_edges(G, pos, arrows=True, edge_color="grey")
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

# Title
plt.title("Applications of AI in Water Treatment", fontsize=16, fontweight='bold')
plt.axis("off")
plt.show()
