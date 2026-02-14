import os
import pickle
import random
import warnings
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUMBER_OF_NODES = 150
NUM_GRAPHS_PER_MODEL = 50
NUM_CORES = 32
TEST_SIZE = 10
TRAIN_SIZE = 140

# Create directories
os.makedirs("graphs/ER", exist_ok=True)
os.makedirs("graphs/WS", exist_ok=True)
os.makedirs("graphs/BA", exist_ok=True)
os.makedirs("plots", exist_ok=True)


# ============================================================================
# SECTION 1: Graph Generation
# ============================================================================


def generate_er_graph(graph_id):
    """Generate Erdős–Rényi graph with random p."""
    p = np.random.uniform(0.02, 0.15)
    G = nx.erdos_renyi_graph(NUMBER_OF_NODES, p, seed=42 + graph_id)
    filename = f"graphs/ER/graph_{graph_id:02d}.gpickle"
    with open(filename, "wb") as f:
        pickle.dump(G, f)
    return {"model": "ER", "graph_id": graph_id, "p": p, "G": G}


def generate_ws_graph(graph_id):
    """Generate Watts–Strogatz graph with random k and beta."""
    k = np.random.choice([4, 6, 8, 10])
    beta = np.random.uniform(0.05, 0.4)
    G = nx.watts_strogatz_graph(NUMBER_OF_NODES, k, beta, seed=42 + graph_id)
    filename = f"graphs/WS/graph_{graph_id:02d}.gpickle"
    with open(filename, "wb") as f:
        pickle.dump(G, f)
    return {"model": "WS", "graph_id": graph_id, "k": k, "beta": beta, "G": G}


def generate_ba_graph(graph_id):
    """Generate Barabási–Albert graph with random m."""
    m = np.random.choice([2, 3, 4, 5])
    G = nx.barabasi_albert_graph(NUMBER_OF_NODES, m, seed=42 + graph_id)
    filename = f"graphs/BA/graph_{graph_id:02d}.gpickle"
    with open(filename, "wb") as f:
        pickle.dump(G, f)
    return {"model": "BA", "graph_id": graph_id, "m": m, "G": G}


def generate_all_graphs():
    """Generate all graphs using multiprocessing."""
    print("Generating graphs...")

    # Generate ER graphs
    with Pool(NUM_CORES) as pool:
        er_results = list(
            tqdm(
                pool.imap(generate_er_graph, range(NUM_GRAPHS_PER_MODEL)),
                total=NUM_GRAPHS_PER_MODEL,
                desc="ER graphs",
            )
        )

    # Generate WS graphs
    with Pool(NUM_CORES) as pool:
        ws_results = list(
            tqdm(
                pool.imap(generate_ws_graph, range(NUM_GRAPHS_PER_MODEL)),
                total=NUM_GRAPHS_PER_MODEL,
                desc="WS graphs",
            )
        )

    # Generate BA graphs
    with Pool(NUM_CORES) as pool:
        ba_results = list(
            tqdm(
                pool.imap(generate_ba_graph, range(NUM_GRAPHS_PER_MODEL)),
                total=NUM_GRAPHS_PER_MODEL,
                desc="BA graphs",
            )
        )

    all_graphs = er_results + ws_results + ba_results
    print(f"Generated {len(all_graphs)} graphs total.")
    return all_graphs


# ============================================================================
# SECTION 2 & 3: Feature Extraction
# ============================================================================


def get_largest_connected_component(G):
    """Get largest connected component of graph."""
    if nx.is_connected(G):
        return G
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def freeman_degree_centralization(G):
    """Compute Freeman degree centralization."""
    if G.number_of_nodes() == 0:
        return 0.0
    degrees = [d for n, d in G.degree()]
    max_deg = max(degrees) if degrees else 0
    if max_deg == 0:
        return 0.0
    n = G.number_of_nodes()
    theoretical_max = (n - 1) * (n - 2) if n > 1 else 0
    if theoretical_max == 0:
        return 0.0
    C = sum(max_deg - d for d in degrees) / theoretical_max
    return C


def degree_entropy(G):
    """Compute Shannon entropy of degree distribution."""
    if G.number_of_nodes() == 0:
        return 0.0
    degrees = [d for n, d in G.degree()]
    if not degrees:
        return 0.0
    degree_counts = {}
    for d in degrees:
        degree_counts[d] = degree_counts.get(d, 0) + 1
    counts = list(degree_counts.values())
    if sum(counts) == 0:
        return 0.0
    probs = np.array(counts) / sum(counts)
    probs = probs[probs > 0]  # Remove zeros
    return entropy(probs, base=2)


def extract_features(G, label):
    """Extract all features from a graph."""
    # Ensure undirected
    G_undir = G.to_undirected() if G.is_directed() else G.copy()

    # Basic stats
    nodes = G_undir.number_of_nodes()
    edges = G_undir.number_of_edges()
    density = nx.density(G_undir)

    # Degree statistics
    degrees = [d for n, d in G_undir.degree()]
    avg_degree = np.mean(degrees) if degrees else 0.0
    degree_variance = np.var(degrees) if degrees else 0.0
    max_degree = max(degrees) if degrees else 0

    # Clustering
    global_clustering = nx.transitivity(G_undir)
    clustering_dict = nx.clustering(G_undir)
    avg_clustering = np.mean(list(clustering_dict.values())) if clustering_dict else 0.0

    # Centralization
    freeman_cent = freeman_degree_centralization(G_undir)

    # Centrality measures
    betweenness = nx.betweenness_centrality(G_undir)
    avg_betweenness = np.mean(list(betweenness.values())) if betweenness else 0.0

    closeness = nx.closeness_centrality(G_undir)
    avg_closeness = np.mean(list(closeness.values())) if closeness else 0.0

    pagerank = nx.pagerank(G_undir)
    avg_pagerank = np.mean(list(pagerank.values())) if pagerank else 0.0

    # Distance metrics on largest connected component
    G_lcc = get_largest_connected_component(G_undir)
    if G_lcc.number_of_nodes() > 1:
        try:
            radius = nx.radius(G_lcc)
            diameter = nx.diameter(G_lcc)
        except:
            radius = 0
            diameter = 0
    else:
        radius = 0
        diameter = 0

    # Entropy
    deg_entropy = degree_entropy(G_undir)

    # Extra features
    assortativity = nx.degree_assortativity_coefficient(G_undir)

    if G_lcc.number_of_nodes() > 1:
        try:
            avg_shortest_path = nx.average_shortest_path_length(G_lcc)
        except:
            avg_shortest_path = 0.0
    else:
        avg_shortest_path = 0.0

    transitivity = nx.transitivity(G_undir)

    try:
        edge_connectivity = nx.edge_connectivity(G_undir)
    except:
        edge_connectivity = 0

    try:
        node_connectivity = nx.node_connectivity(G_undir)
    except:
        node_connectivity = 0

    # Spectral radius (largest eigenvalue of adjacency matrix)
    try:
        adj_matrix = nx.adjacency_matrix(G_undir).todense()
        eigenvalues = np.linalg.eigvals(adj_matrix)
        spectral_radius = np.max(np.real(eigenvalues))
    except:
        spectral_radius = 0.0

    features = {
        "nodes": nodes,
        "edges": edges,
        "density": density,
        "radius": radius,
        "diameter": diameter,
        "avg_degree": avg_degree,
        "degree_variance": degree_variance,
        "max_degree": max_degree,
        "global_clustering": global_clustering,
        "avg_clustering": avg_clustering,
        "freeman_centralization": freeman_cent,
        "avg_betweenness": avg_betweenness,
        "avg_closeness": avg_closeness,
        "avg_pagerank": avg_pagerank,
        "degree_entropy": deg_entropy,
        # Extra features
        "assortativity": assortativity,
        "avg_shortest_path": avg_shortest_path,
        "transitivity": transitivity,
        "edge_connectivity": edge_connectivity,
        "node_connectivity": node_connectivity,
        "spectral_radius": spectral_radius,
        "label": label,
    }

    return features


def extract_features_worker(args):
    """Worker function for parallel feature extraction."""
    graph_info, label = args
    G = graph_info["G"]
    graph_id = graph_info["graph_id"]
    try:
        features = extract_features(G, label)
        features["graph_id"] = graph_id
        return features
    except Exception as e:
        print(f"Error processing graph {graph_id}: {e}")
        return None


def extract_all_features(all_graphs):
    """Extract features from all graphs in parallel."""
    print("Extracting features...")

    # Prepare arguments
    args_list = []
    for graph_info in all_graphs:
        label = graph_info["model"]
        args_list.append((graph_info, label))

    # Extract features in parallel
    with Pool(NUM_CORES) as pool:
        results = list(
            tqdm(
                pool.imap(extract_features_worker, args_list),
                total=len(args_list),
                desc="Feature extraction",
            )
        )

    # Filter out None results
    results = [r for r in results if r is not None]

    # Create DataFrame
    df = pd.DataFrame(results)

    # Reorder columns
    feature_cols = [col for col in df.columns if col not in ["label", "graph_id"]]
    df = df[["graph_id"] + feature_cols + ["label"]]

    print(f"Extracted features from {len(df)} graphs.")
    return df


# ============================================================================
# SECTION 4 & 5: Data Preprocessing and Train/Test Split
# ============================================================================


def preprocess_and_split(df):
    """Preprocess data and create train/test split."""
    print("Preprocessing data...")

    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in ["label", "graph_id"]]
    X = df[feature_cols].copy()
    y = df["label"].copy()

    # Save raw dataset
    df_raw = df.copy()
    df_raw.to_csv("graph_features.csv", index=False)
    print("Saved graph_features.csv")

    # Train/test split (10 test, 140 train)
    # Use stratified split if possible
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            train_size=TRAIN_SIZE,
            stratify=y,
            random_state=42,
        )
    except:
        # If stratification fails, use random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, train_size=TRAIN_SIZE, random_state=42
        )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create scaled DataFrames
    df_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    df_train_scaled["label"] = y_train.values
    df_train_scaled["graph_id"] = df.iloc[X_train.index]["graph_id"].values

    df_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
    df_test_scaled["label"] = y_test.values
    df_test_scaled["graph_id"] = df.iloc[X_test.index]["graph_id"].values

    # Save train/test sets (raw)
    df_train = df.iloc[X_train.index].copy()
    df_test = df.iloc[X_test.index].copy()

    df_train.to_csv("train.csv", index=False)
    df_test.to_csv("test.csv", index=False)
    print("Saved train.csv and test.csv")

    # Save scaled full dataset
    X_scaled = scaler.transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    df_scaled["label"] = y.values
    df_scaled["graph_id"] = df["graph_id"].values
    df_scaled.to_csv("graph_features_scaled.csv", index=False)
    print("Saved graph_features_scaled.csv")

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        feature_cols,
        scaler,
        df_train,
        df_test,
    )


# ============================================================================
# SECTION 6 & 7: kNN Classification with Multiple Distance Metrics
# ============================================================================


def evaluate_knn(X_train, X_test, y_train, y_test, k=3, metric="euclidean"):
    """Train and evaluate kNN classifier."""
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "metric": metric,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "model": knn,
        "y_pred": y_pred,
    }


def run_knn_evaluation(X_train, X_test, y_train, y_test):
    """Run kNN with multiple distance metrics."""
    print("Running kNN classification...")

    metrics = ["euclidean", "manhattan", "cosine"]
    results = []

    for metric in metrics:
        result = evaluate_knn(X_train, X_test, y_train, y_test, k=3, metric=metric)
        results.append(result)
        print(
            f"kNN ({metric}): Accuracy = {result['accuracy']:.4f}, F1 = {result['f1_score']:.4f}"
        )

    return results


# ============================================================================
# SECTION 8: PCA Analysis
# ============================================================================


def perform_pca(X_train_scaled, feature_cols, n_components=3):
    """Perform PCA analysis."""
    print("Performing PCA analysis...")

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Get explained variance
    explained_variance = pca.explained_variance_ratio_

    # Get top contributing features per component
    components = pca.components_
    top_features_per_component = []

    for i, component in enumerate(components):
        # Get absolute loadings
        abs_loadings = np.abs(component)
        top_indices = np.argsort(abs_loadings)[::-1][:5]  # Top 5
        top_features = [
            (feature_cols[idx], component[idx], abs_loadings[idx])
            for idx in top_indices
        ]
        top_features_per_component.append(top_features)

    # Save PCA results
    pca_results = {
        "explained_variance_ratio": explained_variance,
        "top_features": top_features_per_component,
        "pca_model": pca,
    }

    # Create DataFrame for saving
    pca_df = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(n_components)],
            "explained_variance_ratio": explained_variance,
            "cumulative_variance": np.cumsum(explained_variance),
        }
    )
    pca_df.to_csv("pca_results.csv", index=False)
    print("Saved pca_results.csv")

    return pca_results, X_train_pca


# ============================================================================
# SECTION 9: t-SNE Visualization
# ============================================================================


def perform_tsne(X_scaled, y, perplexity=30):
    """Perform t-SNE dimensionality reduction."""
    print("Performing t-SNE...")
    try:
        from sklearn.manifold import TSNE

        tsne = TSNE(
            n_components=2, perplexity=perplexity, random_state=42, max_iter=1000
        )
        X_tsne = tsne.fit_transform(X_scaled)
        return X_tsne
    except ImportError:
        print("sklearn.manifold.TSNE not available, skipping t-SNE")
        return None


# ============================================================================
# SECTION 11: Additional ML Models
# ============================================================================


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple ML models."""
    print("Training additional ML models...")

    models = {
        "kNN": KNeighborsClassifier(n_neighbors=3, metric="euclidean"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="rbf", random_state=42, probability=True),
    }

    results = []

    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Evaluate train
        train_accuracy = accuracy_score(y_train, y_pred_train)
        train_f1 = f1_score(y_train, y_pred_train, average="weighted", zero_division=0)

        # Evaluate test
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)

        results.append(
            {
                "model": name,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "train_f1": train_f1,
                "test_f1": test_f1,
                "model_obj": model,
                "y_pred_test": y_pred_test,
            }
        )

        print(
            f"{name}: Train Acc = {train_accuracy:.4f}, Test Acc = {test_accuracy:.4f}, "
            f"Train F1 = {train_f1:.4f}, Test F1 = {test_f1:.4f}"
        )

    # Save results
    ml_results_df = pd.DataFrame(
        [
            {
                "model": r["model"],
                "train_accuracy": r["train_accuracy"],
                "test_accuracy": r["test_accuracy"],
                "train_f1": r["train_f1"],
                "test_f1": r["test_f1"],
            }
            for r in results
        ]
    )
    ml_results_df.to_csv("ml_results.csv", index=False)
    print("Saved ml_results.csv")

    return results


# ============================================================================
# SECTION 12: Visualizations
# ============================================================================


def create_visualizations(
    df,
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
    feature_cols,
    knn_results,
    pca_results,
    X_train_pca,
    ml_results,
    X_tsne=None,
):
    """Create all visualizations."""
    print("Creating visualizations...")

    try:
        plt.style.use("seaborn-v0_8")
    except:
        try:
            plt.style.use("seaborn")
        except:
            plt.style.use("default")
    sns.set_palette("husl")

    # 1. Feature distributions
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    feature_subset = feature_cols[:20]  # First 20 features

    for i, feat in enumerate(feature_subset):
        if i < len(axes):
            df[feat].hist(ax=axes[i], bins=30, alpha=0.7)
            axes[i].set_title(feat, fontsize=10)
            axes[i].set_xlabel("")
            axes[i].set_ylabel("")

    plt.tight_layout()
    plt.savefig("plots/feature_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plots/feature_distributions.png")

    # 2. Class-wise boxplots (sample features)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    sample_features = [
        "density",
        "avg_degree",
        "global_clustering",
        "avg_betweenness",
        "assortativity",
        "spectral_radius",
    ]

    for i, feat in enumerate(sample_features):
        if i < len(axes) and feat in df.columns:
            df.boxplot(column=feat, by="label", ax=axes[i])
            axes[i].set_title(f"{feat} by Class", fontsize=12)
            axes[i].set_xlabel("Class")
            axes[i].set_ylabel(feat)

    plt.tight_layout()
    plt.savefig("plots/class_wise_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plots/class_wise_boxplots.png")

    # 3. Correlation heatmap
    feature_subset_corr = feature_cols[:15]  # First 15 for readability
    corr_matrix = df[feature_subset_corr].corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plots/correlation_heatmap.png")

    # 4. PCA variance bar chart
    explained_var = pca_results["explained_variance_ratio"]
    plt.figure(figsize=(10, 6))
    plt.bar(
        range(1, len(explained_var) + 1), explained_var, alpha=0.7, color="steelblue"
    )
    plt.xlabel("Principal Component", fontsize=12)
    plt.ylabel("Explained Variance Ratio", fontsize=12)
    plt.title("PCA Explained Variance by Component", fontsize=14, fontweight="bold")
    plt.xticks(range(1, len(explained_var) + 1))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/pca_variance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plots/pca_variance.png")

    # 5. PCA 2D plot
    if X_train_pca is not None and len(X_train_pca) > 0:
        plt.figure(figsize=(10, 8))
        labels_unique = y_train.unique()
        colors = ["red", "blue", "green"]
        for i, label in enumerate(labels_unique):
            mask = y_train == label
            plt.scatter(
                X_train_pca[mask, 0],
                X_train_pca[mask, 1],
                label=label,
                alpha=0.6,
                s=50,
                c=colors[i % len(colors)],
            )
        plt.xlabel("PC1", fontsize=12)
        plt.ylabel("PC2", fontsize=12)
        plt.title("PCA 2D Visualization (Train Set)", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/pca_2d.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved plots/pca_2d.png")

    # 6. t-SNE plot
    if X_tsne is not None:
        plt.figure(figsize=(10, 8))
        labels_all = pd.concat([y_train, y_test]).unique()
        colors = ["red", "blue", "green"]
        n_train = len(y_train)
        for i, label in enumerate(labels_all):
            # Plot train points
            mask_train = y_train == label
            if mask_train.sum() > 0:
                plt.scatter(
                    X_tsne[:n_train][mask_train, 0],
                    X_tsne[:n_train][mask_train, 1],
                    label=f"{label} (train)",
                    alpha=0.6,
                    s=50,
                    c=colors[i % len(colors)],
                    marker="o",
                )
            # Plot test points
            mask_test = y_test == label
            if mask_test.sum() > 0:
                plt.scatter(
                    X_tsne[n_train:][mask_test, 0],
                    X_tsne[n_train:][mask_test, 1],
                    label=f"{label} (test)",
                    alpha=0.6,
                    s=50,
                    c=colors[i % len(colors)],
                    marker="^",
                )
        plt.xlabel("t-SNE Dimension 1", fontsize=12)
        plt.ylabel("t-SNE Dimension 2", fontsize=12)
        plt.title("t-SNE 2D Visualization", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("plots/tsne_visualization.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved plots/tsne_visualization.png")

    # 7. Confusion matrices for kNN
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, result in enumerate(knn_results):
        cm = result["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[i],
            xticklabels=["ER", "WS", "BA"],
            yticklabels=["ER", "WS", "BA"],
        )
        axes[i].set_title(f'kNN Confusion Matrix ({result["metric"]})', fontsize=12)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig("plots/knn_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plots/knn_confusion_matrices.png")

    # 8. Model comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    models = [r["model"] for r in ml_results]
    train_acc = [r["train_accuracy"] for r in ml_results]
    test_acc = [r["test_accuracy"] for r in ml_results]
    train_f1 = [r["train_f1"] for r in ml_results]
    test_f1 = [r["test_f1"] for r in ml_results]

    x = np.arange(len(models))
    width = 0.35

    axes[0].bar(x - width / 2, train_acc, width, label="Train", alpha=0.8)
    axes[0].bar(x + width / 2, test_acc, width, label="Test", alpha=0.8)
    axes[0].set_xlabel("Model", fontsize=12)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_title("Model Comparison: Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x - width / 2, train_f1, width, label="Train", alpha=0.8)
    axes[1].bar(x + width / 2, test_f1, width, label="Test", alpha=0.8)
    axes[1].set_xlabel("Model", fontsize=12)
    axes[1].set_ylabel("F1-Score", fontsize=12)
    axes[1].set_title("Model Comparison: F1-Score", fontsize=14, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plots/model_comparison.png")


# ============================================================================
# Main Pipeline
# ============================================================================


def main():
    """Execute full pipeline."""
    print("=" * 80)
    print("GRAPH CLASSIFICATION PIPELINE")
    print("=" * 80)

    # Step 1: Generate graphs
    all_graphs = generate_all_graphs()

    # Step 2: Extract features
    df = extract_all_features(all_graphs)

    # Step 3: Preprocess and split
    (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        feature_cols,
        scaler,
        df_train,
        df_test,
    ) = preprocess_and_split(df)

    # Step 4: kNN evaluation
    knn_results = run_knn_evaluation(X_train_scaled, X_test_scaled, y_train, y_test)

    # Step 5: PCA analysis
    pca_results, X_train_pca = perform_pca(X_train_scaled, feature_cols, n_components=3)

    # Step 6: t-SNE
    X_all_scaled = np.vstack([X_train_scaled, X_test_scaled])
    y_all = pd.concat([y_train, y_test])
    X_tsne = perform_tsne(X_all_scaled, y_all)

    # Step 7: Additional ML models
    ml_results = train_and_evaluate_models(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    # Step 8: Visualizations
    create_visualizations(
        df,
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        feature_cols,
        knn_results,
        pca_results,
        X_train_pca,
        ml_results,
        X_tsne,
    )

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - graphs/*/graph_*.gpickle")
    print("  - graph_features.csv")
    print("  - graph_features_scaled.csv")
    print("  - train.csv")
    print("  - test.csv")
    print("  - pca_results.csv")
    print("  - ml_results.csv")
    print("  - plots/*.png")

    return {
        "df": df,
        "knn_results": knn_results,
        "pca_results": pca_results,
        "ml_results": ml_results,
    }


if __name__ == "__main__":
    results = main()
