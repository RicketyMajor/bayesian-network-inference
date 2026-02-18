import pandas as pd
import pickle
import os
import time
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")


def estimate_parameters(model, df, name):
    print(f"\n--- Estimando parámetros (CPDs) para {name} ---")
    # Utilizamos MLE para rellenar las tablas de probabilidad basadas en el dataset
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    print(
        f"Parámetros ajustados. La red tiene ahora {len(model.get_cpds())} CPDs.")
    return model


def perform_inferences(model, name):
    print(f"\n--- Realizando Inferencias en {name} ---")
    infer = VariableElimination(model)

    # Inferencia 1: Probabilidad de ingreso dado que es un Adulto con nivel educativo de Bachelors
    q1 = infer.query(variables=['income'], evidence={
                     'age': 'Adulto', 'education': 'Bachelors'})
    print("\n1. P(Income | Age=Adulto, Education=Bachelors):")
    print(q1)

    # Inferencia 2: Probabilidad de ingreso dado que trabaja Horas-Extra y es Hombre
    q2 = infer.query(variables=['income'], evidence={
                     'hours-per-week': 'Horas-Extra', 'sex': 'Male'})
    print("\n2. P(Income | Hours=Horas-Extra, Sex=Male):")
    print(q2)

    # Inferencia 3: Probabilidad de ingreso dado que es Ejecutiva (Exec-managerial) y Mujer
    q3 = infer.query(variables=['income'], evidence={
                     'occupation': 'Exec-managerial', 'sex': 'Female'})
    print("\n3. P(Income | Occupation=Exec-managerial, Sex=Female):")
    print(q3)

    # Inferencia 4: Inferencia causal inversa (Diagnóstico)
    # Si sabemos que la persona GANA MÁS DE 50K, ¿cuál es la probabilidad de su estado civil?
    q4 = infer.query(variables=['marital-status'], evidence={'income': '>50K'})
    print("\n4. P(Marital-Status | Income=>50K):")
    print(q4)


def validate_model(model, test_df, name):
    print(f"\n--- Validando el modelo {name} con el set de Prueba (30%) ---")

    # La inferencia exacta fila por fila en redes grandes es muy costosa computacionalmente.
    # Para la validación, tomaremos una muestra representativa aleatoria de 1000 filas
    # del test set para obtener el Accuracy en un tiempo razonable.
    sample_test = test_df.sample(n=1000, random_state=42)

    y_true = sample_test['income'].values
    test_data_no_target = sample_test.drop(columns=['income'])

    print("Prediciendo valores (esto puede tomar 1 o 2 minutos)...")
    start_time = time.time()

    # Predict busca la clase con mayor probabilidad para cada fila
    predictions = model.predict(test_data_no_target)
    y_pred = predictions['income'].values

    end_time = time.time()

    acc = accuracy_score(y_true, y_pred)
    print(f"Validación completada en {end_time - start_time:.2f} segundos.")
    print(f"Exactitud (Accuracy) de {name}: {acc * 100:.2f}%")
    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred))

    return acc


if __name__ == "__main__":
    # 1. Cargar Datos
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    # 2. Cargar Estructuras (Grafos)
    with open("results/models/hc_dag.pkl", "rb") as f:
        hc_model = pickle.load(f)

    with open("results/models/pc_dag.pkl", "rb") as f:
        pc_model = pickle.load(f)

    # 3. Estimación de Parámetros
    hc_model = estimate_parameters(hc_model, train_df, "Hill-Climbing Model")
    pc_model = estimate_parameters(pc_model, train_df, "PC Algorithm Model")

    # 4. Realizar Inferencias
    perform_inferences(hc_model, "Hill-Climbing Model")
    print("-" * 50)
    perform_inferences(pc_model, "PC Algorithm Model")

    # 5. Validación (Medición de Rendimiento)
    # Por temas de demostración y tiempo de cómputo, validaremos el modelo de Hill-Climbing.
    # Puedes descomentar la validación de PC si deseas comparar ambos formalmente.
    validate_model(hc_model, test_df, "Hill-Climbing Model")
    # validate_model(pc_model, test_df, "PC Algorithm Model")
