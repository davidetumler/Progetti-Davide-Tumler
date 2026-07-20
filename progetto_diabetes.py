"""
Progetto 4 - Machine Learning per la regressione
Dataset Diabetes di scikit-learn.

Il programma:
1. carica e analizza il dataset;
2. crea grafici esplorativi;
3. divide i dati in training e test;
4. confronta sei modelli con K-Fold Cross-Validation;
5. ottimizza Ridge con GridSearchCV;
6. valuta il modello sul test set;
7. crea le learning curves;
8. applica la PCA e disegna il piano di regressione.

Tutti i risultati vengono salvati automaticamente nella cartella "risultati".
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

# Permette di salvare i grafici anche su computer senza interfaccia grafica.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from pandas.plotting import scatter_matrix
from sklearn.base import clone
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "risultati"


def crea_cartella_risultati() -> None:
    """Crea la cartella dei risultati, se non esiste."""
    RESULTS_DIR.mkdir(exist_ok=True)


def carica_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Carica il dataset Diabetes già incluso in scikit-learn."""
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.data.copy()
    y = diabetes.target.copy()

    df = X.copy()
    df["target"] = y
    return X, y, df


def salva_analisi_iniziale(df: pd.DataFrame) -> None:
    """Salva descrizione, prime righe e correlazioni del dataset."""
    df.head(10).to_csv(RESULTS_DIR / "prime_10_righe.csv", index=False)
    df.describe().T.to_csv(RESULTS_DIR / "descrizione_dataset.csv")
    df.corr(numeric_only=True).to_csv(RESULTS_DIR / "matrice_correlazioni.csv")

    with open(RESULTS_DIR / "informazioni_dataset.txt", "w", encoding="utf-8") as file:
        file.write("INFORMAZIONI SUL DATASET\n")
        file.write("=" * 50 + "\n")
        file.write(f"Numero di righe: {df.shape[0]}\n")
        file.write(f"Numero di colonne: {df.shape[1]}\n")
        file.write(f"Valori mancanti totali: {int(df.isna().sum().sum())}\n\n")
        file.write("Nomi delle colonne:\n")
        file.write(", ".join(df.columns) + "\n")


def crea_grafici_esplorativi(df: pd.DataFrame) -> None:
    """Crea istogrammi, boxplot, scatter matrix e matrice di correlazione."""
    feature_df = df.drop(columns="target")

    # 1. Istogrammi
    axes = feature_df.hist(figsize=(15, 10), bins=20)
    for row in np.asarray(axes):
        for axis in np.atleast_1d(row):
            axis.set_ylabel("Frequenza")
    plt.suptitle("Distribuzione delle feature", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(RESULTS_DIR / "01_istogrammi.png", dpi=160)
    plt.close()

    # 2. Boxplot
    plt.figure(figsize=(14, 7))
    feature_df.boxplot(rot=45)
    plt.title("Boxplot delle feature")
    plt.ylabel("Valori standardizzati")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "02_boxplot.png", dpi=160)
    plt.close()

    # 3. Scatter matrix
    scatter_matrix(feature_df, figsize=(18, 18), diagonal="hist", alpha=0.55)
    plt.suptitle("Scatter matrix delle feature", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.98))
    plt.savefig(RESULTS_DIR / "03_scatter_matrix.png", dpi=130)
    plt.close("all")

    # 4. Matrice di correlazione
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(11, 9))
    image = plt.imshow(corr, aspect="auto")
    plt.colorbar(image, label="Correlazione")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Matrice di correlazione")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "04_matrice_correlazioni.png", dpi=160)
    plt.close()


def crea_modelli() -> dict[str, Pipeline]:
    """Restituisce i sei modelli richiesti dal progetto."""
    return {
        "Regressione Lineare": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("model", DecisionTreeRegressor(random_state=RANDOM_STATE)),
            ]
        ),
        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge()),
            ]
        ),
        "Lasso": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Lasso(max_iter=20_000)),
            ]
        ),
        "K-Nearest Neighbor": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor()),
            ]
        ),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVR()),
            ]
        ),
    }


def confronta_modelli(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    kfold: KFold,
) -> pd.DataFrame:
    """Confronta i modelli usando la Negative Mean Squared Error."""
    rows: list[dict[str, float | str]] = []

    print("\nCONFRONTO DEI MODELLI CON K-FOLD CROSS-VALIDATION")
    print("-" * 65)

    for name, model in models.items():
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=kfold,
            scoring="neg_mean_squared_error",
            n_jobs=1,
        )

        nmse_mean = float(scores.mean())
        nmse_std = float(scores.std())
        mse_mean = -nmse_mean

        rows.append(
            {
                "Modello": name,
                "NMSE_medio": nmse_mean,
                "Deviazione_standard": nmse_std,
                "MSE_medio": mse_mean,
            }
        )

        print(
            f"{name:<27} NMSE medio: {nmse_mean:>10.2f} | "
            f"MSE medio: {mse_mean:>10.2f}"
        )

    results = pd.DataFrame(rows).sort_values("NMSE_medio", ascending=False)
    results.to_csv(RESULTS_DIR / "confronto_modelli.csv", index=False)

    plt.figure(figsize=(11, 6))
    plt.bar(results["Modello"], results["MSE_medio"])
    plt.title("Confronto modelli - MSE medio in Cross-Validation")
    plt.ylabel("MSE medio: più basso è meglio")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "05_confronto_modelli.png", dpi=160)
    plt.close()

    return results


def ottimizza_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    kfold: KFold,
) -> GridSearchCV:
    """Ottimizza l'iperparametro alpha del modello Ridge."""
    ridge_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge()),
        ]
    )

    param_grid = {
        "model__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    }

    grid_search = GridSearchCV(
        estimator=ridge_pipeline,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=kfold,
        n_jobs=1,
        refit=True,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    grid_results = pd.DataFrame(grid_search.cv_results_)
    columns_to_save = [
        "param_model__alpha",
        "mean_train_score",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]
    grid_results[columns_to_save].sort_values("rank_test_score").to_csv(
        RESULTS_DIR / "grid_search_ridge.csv",
        index=False,
    )

    print("\nGRID SEARCH DEL MODELLO RIDGE")
    print("-" * 65)
    print(f"Miglior valore di alpha: {grid_search.best_params_['model__alpha']}")
    print(f"Miglior NMSE medio: {grid_search.best_score_:.2f}")

    return grid_search


def valuta_modello(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[float, float, np.ndarray]:
    """Calcola MSE e R² sul test set."""
    predictions = model.predict(X_test)
    mse = float(mean_squared_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))

    prediction_df = pd.DataFrame(
        {
            "Valore_reale": y_test.to_numpy(),
            "Valore_previsto": predictions,
            "Errore": y_test.to_numpy() - predictions,
        }
    )
    prediction_df.to_csv(RESULTS_DIR / "previsioni_test.csv", index=False)

    print("\nVALUTAZIONE FINALE SUL TEST SET")
    print("-" * 65)
    print(f"MSE: {mse:.2f}")
    print(f"R²:  {r2:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.75)
    minimum = min(float(y_test.min()), float(predictions.min()))
    maximum = max(float(y_test.max()), float(predictions.max()))
    plt.plot([minimum, maximum], [minimum, maximum], linestyle="--")
    plt.xlabel("Valori reali")
    plt.ylabel("Valori previsti")
    plt.title("Confronto tra valori reali e previsti")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "06_reali_vs_previsti.png", dpi=160)
    plt.close()

    return mse, r2, predictions


def crea_learning_curve(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    kfold: KFold,
) -> tuple[float, float]:
    """Crea le learning curves e restituisce gli errori finali medi."""
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        train_sizes=np.linspace(0.15, 1.0, 8),
        cv=kfold,
        scoring="neg_mean_squared_error",
        n_jobs=1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_mse = -train_scores.mean(axis=1)
    validation_mse = -validation_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    validation_std = validation_scores.std(axis=1)

    plt.figure(figsize=(9, 6))
    plt.plot(train_sizes, train_mse, marker="o", label="Errore training")
    plt.plot(train_sizes, validation_mse, marker="o", label="Errore validazione")
    plt.fill_between(
        train_sizes,
        train_mse - train_std,
        train_mse + train_std,
        alpha=0.15,
    )
    plt.fill_between(
        train_sizes,
        validation_mse - validation_std,
        validation_mse + validation_std,
        alpha=0.15,
    )
    plt.xlabel("Numero di esempi usati per il training")
    plt.ylabel("MSE: più basso è meglio")
    plt.title("Learning curves del modello Ridge ottimizzato")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "07_learning_curves.png", dpi=160)
    plt.close()

    return float(train_mse[-1]), float(validation_mse[-1])


def analisi_learning_curve(train_error: float, validation_error: float) -> str:
    """Fornisce un commento semplice su bias e varianza."""
    gap = validation_error - train_error

    if gap > 1200:
        return (
            "La distanza tra errore di training e validazione è abbastanza alta. "
            "Il modello mostra una possibile tendenza all'overfitting."
        )
    if validation_error > 5000 and abs(gap) < 1000:
        return (
            "Gli errori di training e validazione sono simili ma ancora alti. "
            "Il modello mostra una possibile tendenza all'underfitting."
        )
    return (
        "Gli errori di training e validazione sono abbastanza vicini. "
        "Il modello presenta un equilibrio ragionevole tra bias e varianza."
    )


def crea_grafici_pca(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    best_alpha: float,
) -> float:
    """Riduce i dati a due componenti e crea il piano di regressione."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    explained_variance = float(pca.explained_variance_ratio_.sum())

    pca_train_df = pd.DataFrame(X_train_pca, columns=["PC1", "PC2"])
    pca_train_df["target"] = y_train.to_numpy()
    pca_train_df.to_csv(RESULTS_DIR / "dati_training_pca.csv", index=False)

    # Grafico 2D
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        X_train_pca[:, 0],
        X_train_pca[:, 1],
        c=y_train.to_numpy(),
        alpha=0.75,
    )
    plt.colorbar(scatter, label="Target")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Dataset ridotto a due dimensioni con PCA")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "08_pca_2d.png", dpi=160)
    plt.close()

    # Lo stesso tipo di modello Ridge, con lo stesso alpha, viene allenato sui due componenti.
    ridge_pca = Ridge(alpha=best_alpha)
    ridge_pca.fit(X_train_pca, y_train)
    pca_predictions = ridge_pca.predict(X_test_pca)
    pca_r2 = float(r2_score(y_test, pca_predictions))

    pc1_values = np.linspace(X_train_pca[:, 0].min(), X_train_pca[:, 0].max(), 45)
    pc2_values = np.linspace(X_train_pca[:, 1].min(), X_train_pca[:, 1].max(), 45)
    pc1_grid, pc2_grid = np.meshgrid(pc1_values, pc2_values)
    grid_points = np.column_stack([pc1_grid.ravel(), pc2_grid.ravel()])
    target_grid = ridge_pca.predict(grid_points).reshape(pc1_grid.shape)

    figure = plt.figure(figsize=(11, 8))
    axis = figure.add_subplot(111, projection="3d")
    points = axis.scatter(
        X_train_pca[:, 0],
        X_train_pca[:, 1],
        y_train.to_numpy(),
        c=y_train.to_numpy(),
        alpha=0.70,
    )
    axis.plot_surface(pc1_grid, pc2_grid, target_grid, alpha=0.30)
    figure.colorbar(points, ax=axis, pad=0.1, label="Target")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.set_zlabel("Target")
    axis.set_title("Piano di regressione Ridge nello spazio PCA")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "09_piano_regressione_pca.png", dpi=160)
    plt.close()

    return explained_variance, pca_r2


def salva_relazione_risultati(
    comparison: pd.DataFrame,
    best_alpha: float,
    mse: float,
    r2: float,
    train_error: float,
    validation_error: float,
    learning_comment: str,
    explained_variance: float,
    pca_r2: float,
) -> None:
    """Salva un riepilogo testuale dei risultati ottenuti."""
    best_initial_model = str(comparison.iloc[0]["Modello"])
    best_initial_nmse = float(comparison.iloc[0]["NMSE_medio"])

    text = f"""RISULTATI FINALI DEL PROGETTO
{'=' * 60}

1. CONFRONTO INIZIALE
Il modello con il miglior NMSE medio è: {best_initial_model}
NMSE medio: {best_initial_nmse:.2f}

Nota: nella Negative Mean Squared Error, il valore migliore è quello più vicino a zero.

2. OTTIMIZZAZIONE RIDGE
Iperparametro ottimizzato: alpha
Miglior valore di alpha: {best_alpha}

3. TEST SET
MSE finale: {mse:.2f}
R² finale: {r2:.4f}

4. LEARNING CURVES
MSE finale training: {train_error:.2f}
MSE finale validazione: {validation_error:.2f}
Commento: {learning_comment}

5. PCA
Varianza spiegata complessivamente da PC1 e PC2: {explained_variance * 100:.2f}%
R² del modello Ridge allenato soltanto su PC1 e PC2: {pca_r2:.4f}

La PCA perde parte delle informazioni perché riduce dieci feature a soltanto due componenti.
Per questo il modello nello spazio PCA può essere meno preciso del modello completo.
"""

    with open(RESULTS_DIR / "risultati_finali.txt", "w", encoding="utf-8") as file:
        file.write(text)


def salva_versioni() -> None:
    """Salva le versioni principali, utili per riprodurre il progetto."""
    versions = (
        f"Python: {sys.version.split()[0]}\n"
        f"NumPy: {np.__version__}\n"
        f"Pandas: {pd.__version__}\n"
        f"Matplotlib: {matplotlib.__version__}\n"
        f"scikit-learn: {sklearn.__version__}\n"
    )
    with open(RESULTS_DIR / "versioni_ambiente.txt", "w", encoding="utf-8") as file:
        file.write(versions)


def main() -> None:
    """Esegue tutto il progetto dall'inizio alla fine."""
    crea_cartella_risultati()
    salva_versioni()

    print("PROGETTO MACHINE LEARNING - DATASET DIABETES")
    print("=" * 65)

    X, y, df = carica_dataset()

    print(f"Righe del dataset: {df.shape[0]}")
    print(f"Feature disponibili: {X.shape[1]}")
    print(f"Valori mancanti: {int(df.isna().sum().sum())}")
    print("\nAnalisi descrittiva iniziale:")
    print(df.describe().round(3))

    salva_analisi_iniziale(df)
    crea_grafici_esplorativi(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    print("\nDIVISIONE DEI DATI")
    print("-" * 65)
    print(f"Dati di training: {X_train.shape[0]} righe")
    print(f"Dati di test: {X_test.shape[0]} righe")

    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = crea_modelli()
    comparison = confronta_modelli(models, X_train, y_train, kfold)

    grid_search = ottimizza_ridge(X_train, y_train, kfold)
    optimized_model = clone(grid_search.best_estimator_)
    optimized_model.fit(X_train, y_train)

    mse, r2, _ = valuta_modello(optimized_model, X_test, y_test)

    train_error, validation_error = crea_learning_curve(
        optimized_model,
        X_train,
        y_train,
        kfold,
    )
    learning_comment = analisi_learning_curve(train_error, validation_error)

    best_alpha = float(grid_search.best_params_["model__alpha"])
    explained_variance, pca_r2 = crea_grafici_pca(
        X_train,
        X_test,
        y_train,
        y_test,
        best_alpha,
    )

    salva_relazione_risultati(
        comparison=comparison,
        best_alpha=best_alpha,
        mse=mse,
        r2=r2,
        train_error=train_error,
        validation_error=validation_error,
        learning_comment=learning_comment,
        explained_variance=explained_variance,
        pca_r2=pca_r2,
    )

    print("\nANALISI DELLE LEARNING CURVES")
    print("-" * 65)
    print(learning_comment)

    print("\nPCA")
    print("-" * 65)
    print(f"Varianza spiegata da PC1 e PC2: {explained_variance * 100:.2f}%")
    print(f"R² usando soltanto PC1 e PC2: {pca_r2:.4f}")

    print("\nPROGRAMMA COMPLETATO CORRETTAMENTE")
    print(f"Apri la cartella dei risultati: {RESULTS_DIR}")


if __name__ == "__main__":
    main()