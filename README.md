# MAUD Qwen Evaluation

Evaluate `Qwen/Qwen3.5-9B` on contract question-answering rows from the MAUD dataset using DeepInfra's OpenAI-compatible API. The script samples rows, asks the model to choose an allowed answer, and saves predictions with exact-match accuracy. It runs inference through a hosted API; it does not train a model locally.

## Dataset and paper

It is recommended to download the official MAUD dataset, but I also provide a copy (`data.zip`) in this repository.

- **Official Dataset Page:** [MAUD — The Atticus Project](https://www.atticusprojectai.org/maud/)
- **Dataset Download:** [MAUD v1 on Zenodo](https://zenodo.org/records/7500064)
- **Paper:** [MAUD: An Expert-Annotated Legal NLP Dataset for Merger Agreement Understanding](https://aclanthology.org/2023.emnlp-main.1019/) (EMNLP 2023)
- **Official repository:** [The-Atticus-Project/maud](https://github.com/The-Atticus-Project/maud)

## Setup

Requires Python **3.13 or newer**, `uv`, and a DeepInfra API token.

Run these commands from the project directory:

```sh
uv sync --locked
```

Create a `.env` file containing your token, or set the environment variable in your shell:

```dotenv
DEEPINFRA_TOKEN=your_deepinfra_api_token
```

If `data/` has not been extracted, unpack the included archive:

```sh
unzip data.zip
```

The evaluator expects these files beside `main.py`:

```text
data/
├── MAUD_train.csv
├── MAUD_dev.csv
└── MAUD_test.csv
```

Input CSVs must contain `contract_name`, `text`, `question`, `subquestion`, `text_type`, `category`, `answer`, and `data_type`. The training CSV is required even when evaluating development or test data because it supplies the allowed answers.

## Run an evaluation

The default run samples five `abridged` training rows with seed `42`:

```sh
uv run main.py
```

To evaluate a larger test sample and save it separately:

```sh
uv run main.py --split test --type abridged --limit 100 --seed 42 --output results/test_abridged.csv
```

| Option | Default | Description |
| --- | --- | --- |
| `--split` | `train` | Dataset split: `train`, `dev`, or `test`. |
| `--type` | `abridged` | Filter by `data_type`: `main`, `abridged`, or `rare_answers`. |
| `--limit` | `5` | Positive number of rows to sample, capped at the available row count. |
| `--seed` | `42` | Random seed for row sampling. |
| `--output` | `results/qwen_predictions.csv` | CSV destination, overwritten on each run. |

Use `uv run main.py --help` for CLI help. Data paths are relative to the script directory; relative output paths are resolved from the working directory.

## Evaluation behavior

- Allowed answers are the sorted unique training answers for each `(question, subquestion)` pair, collected across all training data types.
- Each sampled row makes one sequential API request containing its contract text, question, subquestion, and allowed answers.
- Requests use temperature `0`, `reasoning_effort="none"`, a maximum of `2048` output tokens, and a strict JSON schema for the answer. The model and request settings are defined in `run_qwen()` in `main.py`.
- Predictions count as correct only when they exactly equal the dataset answer. The terminal displays each result and the final sample accuracy.

The seed controls row selection; it does not guarantee identical hosted-model responses. Reported accuracy describes the selected sample, which is only five rows by default.

## Output and failures

The output CSV contains:

```text
contract_name,text,question,subquestion,text_type,category,prediction,expected,matched
```

Parent directories are created automatically. Each completed prediction is flushed to disk, so completed rows remain available if a later request fails. Runs do not resume or append: choose a new `--output` path to preserve an earlier result.

The client uses a 60-second timeout and disables automatic retries. API failures, incomplete responses, empty content, or invalid answers stop the run. If startup fails, check that `DEEPINFRA_TOKEN` is set, the required CSVs exist, and the selected split contains rows of the requested type.

## Project layout

| Path | Purpose |
| --- | --- |
| `main.py` | Dataset loading, sampling, API requests, and CSV evaluation output. |
| `pyproject.toml` | Python requirement and dependencies. |
| `uv.lock` | Locked dependency versions. |
| `data.zip` | Dataset archive. |
| `data/` | Extracted split CSVs, raw data, and contract text. |
| `results/` | Prediction CSVs and optional local analysis files. |

`.env`, `.venv`, `data/`, and `results/` are ignored by Git. The optional plotting script and existing reports under `results/` therefore may not be present in a fresh checkout.
