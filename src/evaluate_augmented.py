import pandas as pd
import pickle
import time
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")


def validate_model(model, test_df, name):
    print(
        f"\n--- Validando el modelo {name} con el set de Prueba Original (30%) ---")

    # Tomamos la misma muestra aleatoria de 1000 filas (misma semilla 'random_state=42')
    # para asegurar que la comparación sea 100% justa contra la Fase 4.
    sample_test = test_df.sample(n=1000, random_state=42)

    y_true = sample_test['income'].values
    test_data_no_target = sample_test.drop(columns=['income'])

    print("Prediciendo valores (esto puede tomar 1 o 2 minutos)...")
    start_time = time.time()

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
    print("1. Cargando el Dataset Aumentado (Entrenamiento) y el Test original (Prueba)...")
    train_aug_df = pd.read_csv("data/processed/train_augmented.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    print(f"   Filas para entrenamiento: {len(train_aug_df)}")

    print("\n2. Cargando la estructura topológica del modelo Hill-Climbing...")
    with open("results/models/hc_dag.pkl", "rb") as f:
        hc_model = pickle.load(f)

    print("\n3. Re-estimando parámetros (CPDs) con el Súper-Dataset...")
    # Aquí es donde ocurre la magia: las probabilidades se ajustan a la nueva gran cantidad de datos
    hc_model.fit(train_aug_df, estimator=MaximumLikelihoodEstimator)

    # 4. Validación Final
    validate_model(hc_model, test_df,
                   "Hill-Climbing Model (DATA AUGMENTATION)")
