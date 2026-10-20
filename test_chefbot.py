"""Test rapidi della logica ChefBot SENZA scaricare Food101 o modelli AI."""

import unittest
import numpy as np
from PIL import Image

from chefbot import load_database, lookup, format_dish, predict_pil, semantic_search


class FakeVision:
    """Modello finto: serve solo a testare l'instradamento dell'immagine."""
    def predict(self, batch, verbose=0):
        assert batch.shape == (1, 160, 160, 3)
        assert batch.dtype == np.float32
        return np.array([[0.75, 0.20, 0.05]])


class FakeTextModel:
    """Embeddings fittizi per verificare la formula del coseno."""
    def encode(self, texts, **kwargs):
        assert kwargs.get("normalize_embeddings") is True
        return np.array([[1.0, 0.0]])


class ChefBotTests(unittest.TestCase):
    def test_database(self):
        db = load_database()
        self.assertEqual(len(db), 15)
        self.assertIn("miele", lookup("baklava", db)["ingredienti"])
        self.assertIn("calorie", format_dish("baklava", db).lower())

    def test_predizione_immagine(self):
        image = Image.new("RGB", (340, 260), color="orange")
        guesses = predict_pil(image, FakeVision(), ["pizza", "sushi", "ramen"])
        self.assertEqual(guesses[0][0], "pizza")
        self.assertAlmostEqual(guesses[0][1], 0.75)

    def test_coseno(self):
        names = ["baklava", "pizza", "sushi"]
        vectors = np.array([[1., 0.], [0., 1.], [-1., 0.]])
        results = semantic_search("qualcosa di dolce", FakeTextModel(), names, vectors)
        self.assertEqual(results[0][0], "baklava")
        self.assertAlmostEqual(results[0][1], 1.0)
        self.assertEqual(results[-1][0], "sushi")


if __name__ == "__main__":
    unittest.main()
