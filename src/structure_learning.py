from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, PC
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
import warnings

# Silenciamos advertencias de librerías subyacentes
warnings.filterwarnings("ignore")


def learn_structure_hc(df):
    print("1. Aprendiendo estructura con Hill-Climbing (Score-based: BIC)...")
    hc = HillClimbSearch(df)
    # Estimamos el mejor DAG indicando 'bicscore' (o 'bic-d' en pgmpy 1.0+)
    best_model = hc.estimate(scoring_method='bic-d')
    return DiscreteBayesianNetwork(best_model.edges())


def learn_structure_pc(df):
    print("2. Aprendiendo estructura con Algoritmo PC (Constraint-based: Chi-Square)...")
    pc = PC(df)
    best_model = pc.estimate(
        return_type='dag', variant='stable', max_cond_vars=2)
    return DiscreteBayesianNetwork(best_model.edges())


def plot_and_save_dag(model, title, filename):
    print(f"   Generando visualización para {title}...")
    plt.figure(figsize=(14, 10))

    nx_graph = nx.DiGraph(model.edges())
    pos = nx.spring_layout(nx_graph, k=1.5, seed=42)

    nx.draw(nx_graph, pos, with_labels=True, node_size=3500, node_color="#87CEFA",
            font_size=9, font_weight="bold", arrows=True, arrowsize=20, edge_color="gray")

    plt.title(title, fontsize=16)

    os.makedirs("results/figures", exist_ok=True)
    filepath = os.path.join("results/figures", filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    train_path = "data/processed/train.csv"
    if not os.path.exists(train_path):
        train_path = "../data/processed/train.csv"

    df_train = pd.read_csv(train_path)
    print(f"Dataset cargado. Dimensiones: {df_train.shape}\n")

    # --- MÉTODO 1: Hill-Climbing ---
    hc_dag = learn_structure_hc(df_train)
    plot_and_save_dag(
        hc_dag, "Red Bayesiana - Hill Climbing (BIC)", "dag_hill_climbing.png")

    # --- MÉTODO 2: PC Algorithm ---
    pc_dag = learn_structure_pc(df_train)
    plot_and_save_dag(
        pc_dag, "Red Bayesiana - Algoritmo PC (Chi-Square)", "dag_pc_algorithm.png")

    print("\n3. Guardando estructuras descubiertas en disco...")
    os.makedirs("results/models", exist_ok=True)

    with open("results/models/hc_dag.pkl", "wb") as f:
        pickle.dump(hc_dag, f)

    with open("results/models/pc_dag.pkl", "wb") as f:
        pickle.dump(pc_dag, f)

    print("\n Proceso completado. Ver redes creadas.")
