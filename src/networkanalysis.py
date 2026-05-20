import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def main():
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)

    comments_df = pd.read_csv("data/processed/comments_with_sentiment.csv")

    reply_df = comments_df[
        (comments_df["is_reply"] == True) &
        (comments_df["reply_to_author"].notna())
    ]

    G = nx.DiGraph()

    for _, row in reply_df.iterrows():
        source = row["author_name"]
        target = row["reply_to_author"]

        if source != target:
            if G.has_edge(source, target):
                G[source][target]["weight"] += 1
            else:
                G.add_edge(source, target, weight=1)

    print("Network nodes:", G.number_of_nodes())
    print("Network edges:", G.number_of_edges())

    if G.number_of_nodes() == 0:
        print("No reply network was created. More reply data is needed.")
        return

    degree_centrality = nx.degree_centrality(G)
    pagerank = nx.pagerank(G, weight="weight")
    betweenness = nx.betweenness_centrality(G)

    centrality_df = pd.DataFrame({
        "user": list(G.nodes()),
        "degree_centrality": [degree_centrality.get(node, 0) for node in G.nodes()],
        "pagerank": [pagerank.get(node, 0) for node in G.nodes()],
        "betweenness": [betweenness.get(node, 0) for node in G.nodes()]
    })

    centrality_df = centrality_df.sort_values(by="pagerank", ascending=False)
    centrality_df.to_csv("outputs/tables/network_centrality_results.csv", index=False)

    network_summary = pd.DataFrame([{
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "is_directed": True,
        "is_weighted": True
    }])

    network_summary.to_csv("outputs/tables/network_summary.csv", index=False)

    top_users = list(centrality_df.head(100)["user"])
    G_sample = G.subgraph(top_users)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G_sample, seed=42)

    nx.draw_networkx_nodes(G_sample, pos, node_size=60)
    nx.draw_networkx_edges(G_sample, pos, alpha=0.3, arrows=True)
    nx.draw_networkx_labels(G_sample, pos, font_size=6)

    plt.title("YouTube Reply Network: Samsung vs iPhone")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/figures/user_reply_network.png", dpi=300)
    plt.close()

    print("Saved network analysis results.")
    print(network_summary)


if __name__ == "__main__":
    main()