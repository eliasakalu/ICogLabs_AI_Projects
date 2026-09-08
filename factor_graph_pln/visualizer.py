import matplotlib.pyplot as plt
import networkx as nx

def visualize_factor_graph(graph, title="Factor Graph Visualization"):
    plt.figure(figsize=(16, 10))
    
    # Use different shapes for candidates and factors.
    var_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "variable"]
    factor_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "factor"]

    # Keep the layout repeatable and leave space between nodes.
    pos = nx.spring_layout(graph, k=1.2, iterations=100, seed=42)

    nx.draw_networkx_nodes(
        graph, pos, 
        nodelist=var_nodes, 
        node_color="#2b5c8f", 
        node_size=700, 
        node_shape="o", 
        label="Variable Nodes (Candidates)"
    )

    nx.draw_networkx_nodes(
        graph, pos, 
        nodelist=factor_nodes, 
        node_color="#e05252", 
        node_size=500, 
        node_shape="s", 
        label="Factor Nodes (Correlation Links)"
    )

    nx.draw_networkx_edges(
        graph, pos, 
        width=1.5, 
        alpha=0.6, 
        edge_color="#777777"
    )

    labels = {n: n for n in graph.nodes()}
    nx.draw_networkx_labels(
        graph, pos, 
        labels=labels, 
        font_size=6, 
        font_color="white", 
        font_weight="bold"
    )

    # Show the correlation beside each link.
    edge_labels = {
        (u, v): f"{graph.nodes[v]['correlation']:.2f}" 
        if graph.nodes[v].get("node_type") == "factor" else ""
        for u, v in graph.edges()
    }
    nx.draw_networkx_edge_labels(
        graph, pos, 
        edge_labels=edge_labels, 
        font_size=7, 
        font_color="#333333",
        label_pos=0.3
    )

    plt.title(title, fontsize=14, pad=15)
    plt.legend(scatterpoints=1, loc="upper right", frameon=True)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
