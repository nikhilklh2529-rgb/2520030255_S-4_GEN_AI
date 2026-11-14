"""Week 4: compare BPE, WordPiece, and SentencePiece in 3 languages.

Only tokenizer configuration/vocab files are downloaded; no neural model weights.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from html import escape
import json
import logging
from pathlib import Path
import sys
from typing import Any, Mapping


SENTENCES = {
    "English": "Artificial Intelligence is transforming education and healthcare.",
    "Telugu": "కృత్రిమ మేధస్సు విద్య మరియు వైద్య రంగాలను మారుస్తోంది.",
    "Hindi": "कृत्रिम बुद्धिमत्ता शिक्षा और स्वास्थ्य सेवा को बदल रही है।",
}

# These model IDs and tokenizer settings match the supplied Week-4 lab document.
TOKENIZER_MODELS = {
    "BPE (GPT-2)": {"model": "openai-community/gpt2", "use_fast": True},
    "WordPiece (Multilingual BERT)": {
        "model": "google-bert/bert-base-multilingual-cased",
        "use_fast": True,
    },
    "SentencePiece (mT5 Unigram)": {"model": "google/mt5-small", "use_fast": False},
}

CSV_COLUMNS = ("Language", "Tokenizer", "Word_Count", "Token_Count", "Token_Ratio")
LOGGER = logging.getLogger("week4")


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "tokenizer_run.log", encoding="utf-8", mode="w")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(console)
    LOGGER.addHandler(file_handler)


def load_tokenizers() -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Missing required package(s). Run SETUP_AND_RUN_WEEK4.bat, "
            "or: python -m pip install -r requirements.txt"
        ) from error

    loaded: dict[str, Any] = {}
    LOGGER.info("\nLoading tokenizers (first run downloads tokenizer files)...")
    for label, config in TOKENIZER_MODELS.items():
        LOGGER.info("  Loading %s | %s", label, config["model"])
        try:
            loaded[label] = AutoTokenizer.from_pretrained(
                config["model"], use_fast=config["use_fast"]
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not load {label} ({config['model']}). "
                "On the first run, check the internet connection and whether "
                "huggingface.co is accessible. Details: " + str(error)
            ) from error
        LOGGER.info("  Loaded successfully.")
    return loaded


def analyze(
    tokenizers: Mapping[str, Any], sentences: Mapping[str, str] = SENTENCES
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run a real tokenizer comparison; dependency injection allows offline tests."""
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    LOGGER.info("\n========== TOKENIZATION RESULTS ==========")
    for language, sentence in sentences.items():
        word_count = len(sentence.split())
        LOGGER.info("\n----------------------------------------")
        LOGGER.info("Language: %s", language)
        LOGGER.info("Original Sentence: %s", sentence)
        for name, tokenizer in tokenizers.items():
            token_ids = tokenizer.encode(sentence, add_special_tokens=False)
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            if len(token_ids) != len(tokens):
                raise ValueError(f"Inconsistent tokens/IDs for {language}, {name}")
            token_count = len(token_ids)
            ratio = round(token_count / word_count, 3) if word_count else 0.0
            row = {
                "Language": language,
                "Tokenizer": name,
                "Word_Count": word_count,
                "Token_Count": token_count,
                "Token_Ratio": ratio,
            }
            rows.append(row)
            details.append({
                **row,
                "Sentence": sentence,
                "Model": TOKENIZER_MODELS.get(name, {}).get("model", "custom-test-tokenizer"),
                "Token_IDs": [int(t) for t in token_ids],
                "Tokens": [str(t) for t in tokens],
            })
            LOGGER.info("\nTokenizer: %s", name)
            LOGGER.info("Tokens: %s", tokens)
            LOGGER.info("Token Count: %s", token_count)
            LOGGER.info("Word Count: %s", word_count)
            LOGGER.info("Token Ratio: %.3f", ratio)
    return rows, details


def print_comparison(rows: list[dict[str, Any]]) -> None:
    LOGGER.info("\n========== TOKEN RATIO COMPARISON ==========")
    LOGGER.info("%-12s %-35s %-10s %-10s %-10s", "Language", "Tokenizer", "Words", "Tokens", "Ratio")
    LOGGER.info("-" * 84)
    for row in rows:
        LOGGER.info(
            "%-12s %-35s %-10s %-10s %-10s",
            row["Language"], row["Tokenizer"], row["Word_Count"],
            row["Token_Count"], row["Token_Ratio"],
        )


def make_html(rows: list[dict[str, Any]], details: list[dict[str, Any]]) -> str:
    """Self-contained HTML report; can be opened offline without a web server."""
    names = list(dict.fromkeys(r["Tokenizer"] for r in rows))
    languages = list(dict.fromkeys(r["Language"] for r in rows))
    max_count = max((r["Token_Count"] for r in rows), default=1)
    palette = ["#E9B567", "#88BDB1", "#B7A3EC"]
    styles = {name: palette[i % len(palette)] for i, name in enumerate(names)}

    cards = "".join(
        '<div class="metric"><div class="metric-label">{}</div><div class="metric-value">{}</div>'
        '<div class="metric-caption">tokenizer comparisons</div></div>'.format(
            escape(language), sum(r["Language"] == language for r in rows)
        )
        for language in languages
    )

    table_rows = []
    for row in rows:
        name = row["Tokenizer"]
        bar_percent = max(1, 100 * row["Token_Count"] / max_count)
        table_rows.append(
            '<tr><td>{language}</td><td><span class="dot" style="background:{color}"></span>'
            '{name}</td><td>{words}</td><td><div class="count-cell"><b>{tokens}</b>'
            '<div class="bar-track"><div class="bar" style="width:{pct:.2f}%;background:{color}"></div>'
            '</div></div></td><td>{ratio:.3f}</td></tr>'.format(
                language=escape(row["Language"]), name=escape(name), color=styles[name],
                words=row["Word_Count"], tokens=row["Token_Count"],
                pct=bar_percent, ratio=row["Token_Ratio"],
            )
        )

    detail_groups = []
    for d in details:
        tokens_html = "".join(
            f'<span class="token" title="ID: {token_id}">{escape(token)}</span>'
            for token, token_id in zip(d["Tokens"], d["Token_IDs"])
        )
        detail_groups.append(
            '<details><summary><span>{language} &nbsp;·&nbsp; {name}</span>'
            '<span class="summary-count">{count} tokens</span></summary>'
            '<div class="detail-body"><p class="sentence">{sentence}</p>'
            '<p class="small">Model: {model} · Word count: {words} · Ratio: {ratio:.3f}</p>'
            '<div class="tokens">{tokens}</div></div></details>'.format(
                language=escape(d["Language"]), name=escape(d["Tokenizer"]),
                count=d["Token_Count"], sentence=escape(d["Sentence"]),
                model=escape(d["Model"]), words=d["Word_Count"],
                ratio=d["Token_Ratio"], tokens=tokens_html,
            )
        )

    generated = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    page = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Week 4 • Tokenizer Comparison</title><style>
:root{color-scheme:dark;--bg:#111316;--panel:#1c2025;--border:#31363d;--txt:#f0f0eb;--muted:#a9b0b8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);font:15px/1.55 system-ui,-apple-system,Segoe UI,Arial,sans-serif}
.container{max-width:1160px;padding:52px 24px 76px;margin:auto}.eyebrow{font-size:12px;color:#E9B567;text-transform:uppercase;letter-spacing:2px;font-weight:700}
h1{font-size:clamp(32px,5vw,60px);line-height:1.1;letter-spacing:-2px;margin:10px 0 15px}h2{font-size:23px;letter-spacing:-.4px;margin:48px 0 16px}
p.lead{max-width:710px;color:var(--muted);font-size:17px;margin:0 0 30px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.metric{border:1px solid var(--border);background:var(--panel);padding:23px;border-radius:16px}.metric-label{color:var(--muted)}
.metric-value{font-size:43px;font-weight:750;line-height:1.2;margin:7px 0}.metric-caption{color:var(--muted);font-size:13px}
.table-scroll{overflow-x:auto;border:1px solid var(--border);border-radius:16px;background:var(--panel)}table{width:100%;border-collapse:collapse;min-width:760px}
th,td{text-align:left;padding:16px 19px;border-bottom:1px solid var(--border)}tr:last-child td{border-bottom:none}th{font-size:12px;letter-spacing:.5px;color:var(--muted);text-transform:uppercase;font-weight:650}td:nth-child(3),td:nth-child(5){font-variant-numeric:tabular-nums}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:9px}.count-cell{display:flex;align-items:center;gap:14px}.count-cell b{min-width:29px}.bar-track{height:6px;border-radius:8px;min-width:110px;width:85%;background:#343940;overflow:hidden}.bar{height:100%;border-radius:8px}
details{border:1px solid var(--border);border-radius:12px;background:var(--panel);margin-bottom:10px}summary{cursor:pointer;display:flex;align-items:center;justify-content:space-between;gap:18px;padding:17px 20px;font-weight:650}
.summary-count{font-weight:450;color:var(--muted);font-size:13px;white-space:nowrap}.detail-body{border-top:1px solid var(--border);padding:15px 20px 22px}.sentence{font-size:16px;overflow-wrap:anywhere}.small{font-size:13px;color:var(--muted);overflow-wrap:anywhere}.tokens{display:flex;flex-wrap:wrap;gap:8px;margin-top:17px}.token{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere;max-width:100%;padding:6px 10px;background:#303b43;color:#d9eee9;border:1px solid #45545e;border-radius:7px}
.note{margin-top:32px;color:var(--muted);font-size:13px}.legend{color:var(--muted);font-size:13px;margin-top:11px}@media(max-width:650px){.container{padding:32px 14px 60px}.metrics{grid-template-columns:1fr}.metric{padding:15px}.metric-value{font-size:31px}.summary-count{font-size:11px}summary{padding:13px}}
</style></head><body><main class="container"><div class="eyebrow">NLP / LLM Laboratory &nbsp;·&nbsp; Week 04</div>
<h1>Tokenizer Comparison</h1><p class="lead">An experiment comparing Byte Pair Encoding (GPT-2), WordPiece (Multilingual BERT) and SentencePiece (mT5) across English, Telugu and Hindi.</p>
<section class="metrics">__CARDS__</section><h2>Token ratio comparison</h2><section class="table-scroll"><table>
<thead><tr><th>Language</th><th>Tokenizer</th><th>Words</th><th>Tokens</th><th>Token ratio</th></tr></thead>
<tbody>__ROWS__</tbody></table></section><p class="legend">Token ratio = token count ÷ whitespace-separated word count. Special tokens are excluded. Lower ratios mean fewer tokens for this specific text, not inherently a better model.</p>
<h2>Inspect every token</h2><p class="legend">Open a row to view its token pieces. Hover a piece for its token ID.</p>
__DETAILS__<p class="note">Created __DATE__. All analysis is computed from real tokenizer outputs, not hard-coded example results.</p></main></body></html>"""
    return (page.replace("__CARDS__", cards).replace("__ROWS__", "".join(table_rows))
            .replace("__DETAILS__", "".join(detail_groups)).replace("__DATE__", escape(generated)))


def save_outputs(output_dir: Path, rows: list[dict[str, Any]], details: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "tokenizer_results.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "tokenizer_details.json").open("w", encoding="utf-8") as file:
        json.dump({"sentences": SENTENCES, "results": details}, file, ensure_ascii=False, indent=2)
    (output_dir / "tokenizer_report.html").write_text(make_html(rows, details), encoding="utf-8")
    LOGGER.info("\nFiles saved in: %s", output_dir.resolve())
    LOGGER.info("  - tokenizer_results.csv")
    LOGGER.info("  - tokenizer_details.json")
    LOGGER.info("  - tokenizer_report.html")
    LOGGER.info("  - tokenizer_run.log")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "results",
        help="Directory for results (default: project/results)",
    )
    args = parser.parse_args(argv)
    configure_logging(args.output_dir)
    LOGGER.info("WEEK 4: TOKENIZER COMPARISON LAB")
    LOGGER.info("Models: GPT-2 BPE, Multilingual BERT WordPiece, mT5 SentencePiece")
    try:
        tokenizers = load_tokenizers()
        rows, details = analyze(tokenizers)
        print_comparison(rows)
        save_outputs(args.output_dir, rows, details)
        LOGGER.info("\nWeek 4 experiment completed successfully!")
        return 0
    except KeyboardInterrupt:
        LOGGER.error("\nStopped by user.")
        return 130
    except Exception:
        LOGGER.exception("\nExperiment failed. See the error above and results/tokenizer_run.log.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
