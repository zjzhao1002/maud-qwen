# MAUD × Qwen

Evaluate Qwen on MAUD contract questions by supplying a text excerpt, question, and subquestion, then comparing the model's answer with the dataset answer. This project runs inference through DeepInfra; it does not fine-tune the model.

Predictions are constrained to the answer options observed in training for each `(question, subquestion)` pair. Results are saved to CSV and visualized by category and text type.

## Setup

Use Python 3.13 or later and `uv`. Run commands from the repository root.

```bash
uv sync --extra eda
```

The `eda` extra includes Matplotlib, which is needed for the results plots, and dependencies for the dataset analysis. For inference only, `uv sync` is sufficient.

Create a `.env` file in the repository root with your DeepInfra token:

```dotenv
DEEPINFRA_TOKEN=your_token_here
```

`.env` is excluded from Git. Inference requires network access and uses your DeepInfra account; plotting and dataset analysis run locally without API calls.

## Dataset

MAUD (Merger Agreement Understanding Dataset) is curated by The Atticus Project.

- **Official dataset page:** [MAUD — The Atticus Project](https://www.atticusprojectai.org/maud/)
- **Dataset download:** [MAUD v1 on Zenodo](https://zenodo.org/records/7500064)
- **Original code repository:** [The-Atticus-Project/maud](https://github.com/The-Atticus-Project/maud)
- **Paper:** Steven H. Wang et al. (2023), [MAUD: An Expert-Annotated Legal NLP Dataset for Merger Agreement Understanding](https://arxiv.org/abs/2301.00876).

The inference script reads the prepared split files directly:

```text
data/
├── MAUD_train.csv
├── MAUD_dev.csv
├── MAUD_test.csv
├── raw/
└── contracts/
```

Select abridged examples using `--type abridged`; no separate cleaning step is required. The script filters the chosen split on `data_type` and preserves literal strings such as `None` when loading CSVs.

## Run Qwen

Start with five random abridged training examples:

```bash
uv run python main.py
```

Run 100 examples with reproducible row selection:

```bash
uv run python main.py --split train --type abridged --limit 100 --seed 42
```

Evaluate a dev sample and save it separately:

```bash
uv run python main.py --split dev --type abridged --limit 100 --seed 42 --output results/qwen_dev_predictions.csv
```

| Argument | Default | Description |
|---|---|---|
| `--split` | `train` | Dataset split: `train`, `dev`, or `test`. |
| `--type` | `abridged` | Source subset: `main`, `abridged`, or `rare_answers`. The selected split must contain that subset. |
| `--limit` | `5` | Positive number of rows to sample; capped at the available subset size. |
| `--seed` | `42` | Seed for random sampling without replacement. |
| `--output` | `results/qwen_predictions.csv` | CSV destination, relative to the working directory unless absolute. Overwritten each run. |

The current configuration in [main.py](main.py) uses `Qwen/Qwen3.5-9B`, `temperature=0`, `reasoning_effort="none"`, and a maximum of 2,048 output tokens. The request asks the provider to disable reasoning; the script does not record reasoning usage to verify this. Model and generation settings are configured in the code, not command-line arguments.

The full training split supplies the allowed answer vocabulary, even when evaluating dev or test. A JSON Schema enum constrains each answer, and the response is validated before saving. The prompt does not include the current row's expected answer. Constraining the output ensures a valid choice, not a correct prediction.

The seed controls sample selection. Temperature zero reduces sampling randomness but does not guarantee identical hosted-model responses across runs.

## Prediction output

The CSV contains these columns, in order:

```text
contract_name,text,question,subquestion,text_type,category,prediction,expected,matched
```

`matched` is `True` when `prediction == expected`, using exact string equality. The script prints each result and the overall accuracy when the run completes.

Each completed prediction is flushed to disk, so earlier results remain if a later request fails. Requests use a 60-second timeout with automatic retries disabled. A failed request or invalid response stops the run; there is no automatic resume, and rerunning to the same output path overwrites the partial CSV.

## Plot results

Generate all six plots from the default predictions file:

```bash
uv run --extra eda python results/plot.py
```

Use another predictions file:

```bash
uv run --extra eda python results/plot.py --input results/qwen_dev_predictions.csv
```

Plots are always saved under `results/` with fixed filenames. There is no `--output` option for plotting, and each run overwrites the existing figures.

| File | Meaning |
|---|---|
| `category_accuracy.png` | Exact-match accuracy within each category. |
| `text_type_accuracy.png` | Exact-match accuracy within each text type. |
| `matched_category_pie.png` | Each category's share of all correct predictions. |
| `mismatched_category_pie.png` | Each category's share of all incorrect predictions. |
| `matched_text_type_pie.png` | Each text type's share of all correct predictions. |
| `mismatched_text_type_pie.png` | Each text type's share of all incorrect predictions. |

The plotting script recomputes matches from `prediction` and `expected`. Accuracy plots divide correct rows by all rows in that group; pie charts divide group counts by all matched or all mismatched rows. These are different quantities. Counts accompany percentages, and colors are consistent across each matched/mismatched pie pair.

## Current analysis

The recorded 100-row, seed-42 abridged training sample achieved **49/100 exact matches (49%)**, spanning 60 contracts, 7 categories, and 17 text types. Conditions to Closing accounted for 25 of the 51 errors.

See the [analysis report](results/analysis_report.md) for detailed tables, figures, methodology, and limitations. This is a snapshot of the recorded run; it is not automatically updated by the scripts.

The earlier first-100-row sample contained only consideration-type questions. Random sampling provides broader task coverage, but it does not balance tasks. Small groups, related examples from the same contract, and strict scoring of compound answer labels limit the conclusions that can be drawn from a single small sample. Training-split inference is useful for initial checks; after tuning prompts on these examples, use separate dev examples to assess changes and reserve test data for final evaluation.

## Project files

| Path | Purpose |
|---|---|
| [main.py](main.py) | Sample data, call Qwen, validate answers, and save predictions. |
| [results/plot.py](results/plot.py) | Generate accuracy charts and outcome-composition pie charts. |
| [results/analysis_report.md](results/analysis_report.md) | Written analysis of the recorded sample. |
| `data/` | Prepared splits, raw annotations, and contract text. |
| [pyproject.toml](pyproject.toml) / [uv.lock](uv.lock) | Dependency configuration and lockfile. |
