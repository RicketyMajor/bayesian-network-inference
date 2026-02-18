import pandas as pd
from sklearn.model_selection import train_test_split
import os


def load_and_preprocess(filepath):
    print("Cargando datos limpios...")
    df = pd.read_csv(filepath)

    # 1. ELIMINACIÓN DE VARIABLES REDUNDANTES O NO CAUSALES
    # 'fnlwgt': Es un peso de muestreo del censo, no una característica inherente a la persona.
    # 'education-num': Es exactamente lo mismo que 'education' pero en número. Retener ambas crearía
    # una dependencia probabilística determinista (100% correlacionadas) que ensucia la red.
    df = df.drop(columns=['fnlwgt', 'education-num'])

    # 2. DISCRETIZACIÓN DE VARIABLES CONTINUAS
    print("Discretizando variables continuas...")

    # Edad: Agrupamos en etapas de la vida
    df['age'] = pd.cut(df['age'],
                       bins=[0, 25, 45, 65, 100],
                       labels=['Joven', 'Adulto', 'Mediana-Edad', 'Senior'])

    # Horas por semana: Agrupamos por jornadas laborales
    df['hours-per-week'] = pd.cut(df['hours-per-week'],
                                  bins=[0, 35, 40, 100],
                                  labels=['Medio-Tiempo', 'Tiempo-Completo', 'Horas-Extra'])

    # Ganancias y Pérdidas de Capital: La inmensa mayoría de estos datos son 0 en el dataset.
    # Lo más lógico probabilísticamente es binarizar: ¿Tuvo ganancias/pérdidas o no?
    df['capital-gain'] = pd.cut(df['capital-gain'],
                                bins=[-1, 0, 999999], labels=['Cero', 'Positivo'])
    df['capital-loss'] = pd.cut(df['capital-loss'],
                                bins=[-1, 0, 999999], labels=['Cero', 'Positivo'])

    return df


def split_and_save(df, output_dir):
    print("Dividiendo los datos en Entrenamiento (70%) y Prueba (30%)...")
    # 3. Split: 70% entrenamiento, 30% prueba.
    # Usamos random_state para reproducibilidad y estratificamos por la variable objetivo 'income'
    # para asegurar que la proporción de >50K y <=50K se mantenga igual en ambos sets.
    train_df, test_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['income'])

    # 4. Guardar los datasets
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n Proceso completado exitosament")
    print(f"Dimensión Train (70%): {train_df.shape}")
    print(f"Dimensión Test (30%):  {test_df.shape}")


if __name__ == "__main__":
    # Rutas relativas asumiendo que el script se ejecuta desde la raíz del proyecto
    INPUT_PATH = "data/raw/adult_clean.csv"
    OUTPUT_DIR = "data/processed"

    # Verificación de ruta por si el usuario lo ejecuta estando dentro de la carpeta src/
    if not os.path.exists(INPUT_PATH):
        INPUT_PATH = "../data/raw/adult_clean.csv"
        OUTPUT_DIR = "../data/processed"

    processed_df = load_and_preprocess(INPUT_PATH)
    split_and_save(processed_df, OUTPUT_DIR)
