# Week 4 — Tokenizer Comparison Lab

A complete, GitHub-ready Python project for the Week-4 NLP/LLM experiment. It compares **Byte Pair Encoding (BPE)**, **WordPiece**, and **SentencePiece** on identical **English, Telugu, and Hindi** sentences, printing each token, counting tokens, calculating token/word ratios, and generating CSV, JSON, and a responsive HTML report.

**No API key, database, GPU, PyTorch, training, or `.env` secrets are needed.** Only the tokenizer vocabulary/configuration files are fetched from Hugging Face on the first successful run; the full neural model weights are **not** downloaded.

## 1. Run it on Windows — easiest method

1. Download the ZIP and **Extract All**. Do not try to run a `.bat` inside ZIP preview.
2. Open the extracted `Week-4-Tokenizer-Lab` folder.
3. Install [Python 3.10+](https://www.python.org/downloads/) if you do not already have it. During Python installation, enable **Add python.exe to PATH**.
4. Connect to the internet (required for the first run).
5. Double-click **`SETUP_AND_RUN_WEEK4.bat`**.
6. It finds Python, creates `.venv`, installs dependencies, downloads tokenizer data, runs the experiment, and opens `results/tokenizer_report.html` in your browser.

> Windows may show a security warning for a `.bat` downloaded from the internet. Inspect the file first and run only if you trust it. The file calls the provided Python/pip scripts; it does not need Administrator permission.

**Next time:** double-click `RUN_WEEK4.bat` (uses the existing environment, does not reinstall packages) or `OPEN_REPORT.bat` to view the last saved report. Use `SETUP_ENV.bat` to install/update the libraries without running the lab.

## 2. How to set up the Python environment manually

Open **Command Prompt** in the project folder and enter:

```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python tokenizer_lab.py
```

You should see `(.venv)` at the start of the Command Prompt line after activation.

**PowerShell (without needing to change PowerShell execution policy):**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tokenizer_lab.py
```

Using `\.venv\Scripts\python.exe` directly also works in Command Prompt. Activation is optional.

To install the dependencies again later:

```bat
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3. macOS / Linux

In Terminal, from the extracted project directory:

```bash
bash setup_and_run.sh
```

Or do it manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tokenizer_lab.py
```

## 4. Open the project in VS Code

1. VS Code → **File → Open Folder** → select `Week-4-Tokenizer-Lab`.
2. Install the Microsoft **Python** extension if needed.
3. Press `Ctrl + Shift + P` → **Python: Select Interpreter** → choose `.venv\Scripts\python.exe` on Windows (or `.venv/bin/python` on macOS/Linux).
4. Use **Terminal → New Terminal**. Run `python tokenizer_lab.py` when the virtual environment is selected/activated. Otherwise run `.venv\Scripts\python.exe tokenizer_lab.py` directly.

## 5. What the code does

| Method | Actual tokenizer model | Setting |
|---|---|---|
| Byte Pair Encoding | `openai-community/gpt2` | `use_fast=True` |
| WordPiece | `google-bert/bert-base-multilingual-cased` | `use_fast=True` |
| SentencePiece (Unigram) | `google/mt5-small` | `use_fast=False` |

The three input sentences are exactly those in the uploaded Week-4 assignment. We call `tokenizer.encode(sentence, add_special_tokens=False)` for each language/model combination, then `convert_ids_to_tokens` to show the subword pieces. Each token ratio is **token count divided by whitespace-separated word count**, rounded to 3 decimal places.

The output includes **9 comparison rows (3 languages × 3 tokenizers)**. Your actual token counts depend on the downloaded tokenizer versions/files; the program never invents or hardcodes counts.

## 6. Where are my outputs?

All generated files appear **after a successful first run** inside `results/`:

| File | Contains |
|---|---|
| `tokenizer_results.csv` | Spreadsheet-friendly 9-row comparison with word counts, token counts, ratios |
| `tokenizer_details.json` | Original sentences, token pieces, token IDs, counts, model names |
| `tokenizer_report.html` | Responsive visual results page, comparison bars, inspectable tokens |
| `tokenizer_run.log` | Experiment progress, token breakdowns, and error messages |

Open the HTML file by double-clicking it. Import the CSV into Excel or Google Sheets. **The CSV and HTML are generated from live tokenizer results, not prefilled screenshots.** The `.gitignore` intentionally excludes generated files and `.venv` from GitHub.

## 7. Publish it to GitHub

### Option A — GitHub website (no Git commands)

1. Sign in to <https://github.com/> and create a **New repository** named e.g. `Week-4-Tokenizer-Lab`.
2. Keep it **empty**: don't tick *Add a README*, `.gitignore`, or a license (those files already exist or are optional).
3. Click **uploading an existing file**.
4. **Extract the ZIP first.** Upload the **contents** of the `Week-4-Tokenizer-Lab` folder, not the outer ZIP. Make sure `tokenizer_lab.py`, `requirements.txt`, README, Windows BAT files, and `.github/workflows/python-checks.yml` are committed.
5. Commit the files. GitHub Actions will automatically check Python syntax and run offline tests on pushes and pull requests.

If the browser uploader does not preserve folders or hidden `.github` files, use Option B instead.

### Option B — Git + Command Prompt (recommended, preserves the full folder structure)

Install [Git for Windows](https://git-scm.com/downloads/win), then create an **empty** repository on GitHub named `Week-4-Tokenizer-Lab` and replace `YOUR_USERNAME` in the command below:

```bat
git init
git add .
git commit -m "Add Week 4 tokenizer comparison lab"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/Week-4-Tokenizer-Lab.git
git push -u origin main
```

Run these commands **from inside the extracted project folder**. GitHub will prompt for authentication as needed; do not put a password or access token into source files. You can also use the GitHub Desktop app to publish the local folder.

## 8. Quick verification before submitting

The included unit tests run **offline**, without installing transformers or downloading any models:

```bat
py -3 -m unittest discover -s tests -v
```

They validate the 3 languages and 3 model specifications, token counts/ratios, Unicode text, CSV creation, JSON data, and HTML content.

A GitHub Actions workflow is included at `.github/workflows/python-checks.yml`. It runs only quick offline tests. It does **not** run the network-dependent experiment on each push.

## 9. Troubleshooting

- **`Python 3.10 or newer was not found`**: Install Python from python.org, enable *Add Python to PATH*, open a fresh terminal, and rerun `SETUP_AND_RUN_WEEK4.bat`. Verify with `py -3 --version` or `python --version`.
- **`pip install` fails**: Check your internet connection or university proxy/firewall. Rerun `SETUP_ENV.bat`; existing libraries are reused when possible.
- **`Could not load ... huggingface.co`**: The tokenizers must be downloaded on the first run. Check your internet connection and Hugging Face access, then rerun `RUN_WEEK4.bat`. The error is also recorded in `results/tokenizer_run.log`.
- **`No module named ...`**: Use the Python interpreter inside `.venv`, or rerun `SETUP_ENV.bat`. Do not install packages into a different Python accidentally.
- **PowerShell activation blocked**: Use `\.venv\Scripts\python.exe tokenizer_lab.py` instead of activating, or run the provided `.bat`.
- **`results/` is empty**: Results are intentionally generated after the live experiment succeeds. Read `results/tokenizer_run.log` and solve the preceding error.
- **GitHub rejects `.venv` or other huge folders**: `.venv/` should never be uploaded. Its contents depend on your OS and Python version, and are recreated from `requirements.txt`. The repository's `.gitignore` already excludes it.

## Repository layout

```text
Week-4-Tokenizer-Lab/
├── SETUP_AND_RUN_WEEK4.bat    # Double-click first time (Windows)
├── SETUP_ENV.bat              # Create venv and install packages only
├── RUN_WEEK4.bat              # Run again using the existing venv
├── OPEN_REPORT.bat            # Reopen the latest HTML report
├── setup_and_run.sh           # macOS/Linux setup and run
├── tokenizer_lab.py           # Full Week-4 program
├── requirements.txt           # Exact dependency families/version ranges
├── tests/
│   └── test_tokenizer_lab.py  # Offline unit tests
├── results/
│   └── .gitkeep               # Actual report files appear after run
├── .github/
│   └── workflows/
│       └── python-checks.yml # Automated GitHub syntax and unit checks
├── .gitignore                 # Excludes generated results/venv/secrets
├── .gitattributes             # Cross-platform script line endings
└── README.md                  # This guide
```

**Assignment:** Week-4 — compare BPE, WordPiece, SentencePiece for English, Telugu, Hindi, then compare token counts and token ratios. All core assignment steps are implemented.
