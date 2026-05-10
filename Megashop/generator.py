import pandas as pd
import numpy as np
import os

# creo le cartelle
os.makedirs("dataset/transazioni_json", exist_ok=True)
os.makedirs("dataset/transazioni_parquet", exist_ok=True)
os.makedirs("dataset/anagrafiche", exist_ok=True)

# -----------------------------
# tabella regioni
# -----------------------------

regioni = pd.DataFrame({

    "region_id": [1, 2, 3, 4],

    "region_name": [
        "Nord",
        "Centro",
        "Sud",
        "Isole"
    ]
})

regioni.to_parquet(
    "dataset/anagrafiche/regions.parquet"
)

# -----------------------------
# tabella prodotti
# -----------------------------

prodotti = pd.DataFrame({

    "product_id": [101, 102, 103, 104],

    "category": [
        "Elettronica",
        "Casa",
        "Sport",
        "Abbigliamento"
    ]
})

prodotti.to_parquet(
    "dataset/anagrafiche/products.parquet"
)

# -----------------------------
# transazioni
# -----------------------------

for i in range(5):

    df = pd.DataFrame({

        "transaction_id": range(i * 100, (i + 1) * 100),

        "product_id": np.random.choice(
            [101, 102, 103, 104],
            100
        ),

        "region_id": np.random.choice(
            [1, 2, 3, 4],
            100
        ),

        "amount": np.random.randint(
            10,
            500,
            100
        ),

        "year": np.random.choice(
            [2023, 2024],
            100
        )
    })

    # json
    df.to_json(
        f"dataset/transazioni_json/file_{i}.jsonl",
        orient="records",
        lines=True
    )

    # parquet
    df.to_parquet(
        f"dataset/transazioni_parquet/file_{i}.parquet"
    )

print("Dataset creato")