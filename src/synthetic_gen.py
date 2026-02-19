import pandas as pd
import pickle
import os
import time
from pgmpy.sampling import BayesianModelSampling
import warnings

warnings.filterwarnings("ignore")


def generate_synthetic_data(model, num_samples):
    print(f"--- Iniciando Generación de {num_samples} Datos Sintéticos ---")
    print("Este proceso utiliza Forward Sampling. Puede tardar unos minutos...")

    start_time = time.time()

    # Inicializamos el muestreador con nuestra red entrenada
    sampler = BayesianModelSampling(model)

    # Generamos las muestras. 'forward_sample' respeta la topología del DAG.
    synthetic_data = sampler.forward_sample(size=num_samples)

    end_time = time.time()
    print(
        f"Datos generados en {end_time - start_time:.2f} segundos")

    return synthetic_data


if __name__ == "__main__":
    # 1. Cargar el dataset de entrenamiento original
    train_df = pd.read_csv("data/processed/train.csv")
    print(
        f"Tamaño del dataset de entrenamiento original: {train_df.shape[0]} filas.")

    # 2. Cargar el mejor modelo (Hill-Climbing)
    with open("results/models/hc_dag.pkl", "rb") as f:
        model = pickle.load(f)

    # NOTA: Como guardamos la estructura en la Fase 3, necesitamos ajustarle los parámetros
    # nuevamente antes de hacer sampling (ya que en la fase 4 no guardamos el modelo ajustado)
    from pgmpy.estimators import MaximumLikelihoodEstimator
    model.fit(train_df, estimator=MaximumLikelihoodEstimator)

    # 3. Calcular el 50% del tamaño original
    num_synthetic_samples = int(len(train_df) * 0.5)

    # 4. Generar datos
    df_synthetic = generate_synthetic_data(model, num_synthetic_samples)

    # 5. Guardar los datos puramente sintéticos por si queremos analizarlos
    os.makedirs("data/synthetic", exist_ok=True)
    df_synthetic.to_csv("data/synthetic/synthetic_only.csv", index=False)

    # 6. Combinar (Augmentation) y guardar el nuevo dataset masivo
    df_augmented = pd.concat([train_df, df_synthetic], ignore_index=True)

    print("\n--- Resumen de Data Augmentation ---")
    print(f"Originales: {len(train_df)}")
    print(f"Sintéticos: {len(df_synthetic)}")
    print(f"TOTAL AUMENTADO: {len(df_augmented)}")

    # Guardamos el dataset aumentado en processed
    df_augmented.to_csv("data/processed/train_augmented.csv", index=False)
    print("\nDataset aumentado guardado en 'data/processed/train_augmented.csv'.")
