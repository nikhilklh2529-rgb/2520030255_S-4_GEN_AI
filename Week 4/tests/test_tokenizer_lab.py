"""Offline tests: no internet, pip install, or Hugging Face download needed."""
import csv
import json
import logging
from pathlib import Path
import tempfile
import unittest

from tokenizer_lab import SENTENCES, TOKENIZER_MODELS, analyze, make_html, save_outputs


class DummyTokenizer:
    def __init__(self, size):
        self.size = size

    def encode(self, sentence, add_special_tokens=True):
        assert add_special_tokens is False
        return list(range(self.size))

    def convert_ids_to_tokens(self, ids):
        return [f"piece-{n}" for n in ids]


class Week4Tests(unittest.TestCase):
    def test_exact_specification(self):
        self.assertEqual(list(SENTENCES), ["English", "Telugu", "Hindi"])
        self.assertEqual(len(TOKENIZER_MODELS), 3)
        self.assertEqual(TOKENIZER_MODELS["SentencePiece (mT5 Unigram)"]["use_fast"], False)
        self.assertIn("कृत्रिम", SENTENCES["Hindi"])

    def test_nine_rows_and_correct_ratios(self):
        test_tokenizers = {name: DummyTokenizer(i + 2) for i, name in enumerate(TOKENIZER_MODELS)}
        rows, details = analyze(test_tokenizers)
        self.assertEqual(len(rows), 9)
        self.assertEqual(len(details), 9)
        for row in rows:
            self.assertEqual(row["Token_Ratio"], round(row["Token_Count"] / row["Word_Count"], 3))
        self.assertEqual(rows[0]["Token_Count"], 2)
        self.assertEqual(rows[1]["Token_Count"], 3)

    def test_writes_csv_json_and_html_in_unicode(self):
        rows, details = analyze({"BPE (GPT-2)": DummyTokenizer(3)})
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            save_outputs(output, rows, details)
            with (output / "tokenizer_results.csv").open(encoding="utf-8-sig", newline="") as file:
                csv_rows = list(csv.DictReader(file))
            self.assertEqual(len(csv_rows), 3)
            json_data = json.loads((output / "tokenizer_details.json").read_text(encoding="utf-8"))
            self.assertEqual(len(json_data["results"]), 3)
            html = (output / "tokenizer_report.html").read_text(encoding="utf-8")
            self.assertIn(SENTENCES["Telugu"], html)
            self.assertIn(SENTENCES["Hindi"], html)
            self.assertIn("token ratio", html.lower())


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    unittest.main()
