# Autoresearch

This is an experiment to have an AI agent do its own research. It gives a small but real LLM training setup to an AI researcher and lets it experiment autonomously. The agent modifies the code, trains for a fixed time budget, checks if the result improved, and repeats.

This project is a fork and adaptation of the `autoresearch` repository, 
originally created and developed by [Andrej Karpathy](https://github.com/karpathy).

This version is optimized for **CPU-only** training, making it accessible even if you don't have a high-end NVIDIA GPU.

## Setup

The repository is designed to be beginner-friendly. You can use either `uv` (recommended) or `pip`.

### Requirements
- **Python 3.10 to 3.12** (tested with 3.10)
- (Optional) [uv](https://docs.astral.sh/uv/) for faster dependency management.

### Installation

#### Using uv (Recommended)
```bash
# 1. Install dependencies
uv sync

# 2. Download data and train tokenizer (~2-5 min)
uv run prepare.py

# 3. Establish the baseline run (~30 min)
uv run train.py
```

#### Using pip
```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data and train tokenizer
python prepare.py

# 4. Establish the baseline run
python train.py
```

## Experimentation logic

To set up a new experiment, follow these steps:

1. **Agree on a run tag**: Choose a tag based on today's date (e.g., `mar12`). Ensure the branch `autoresearch/<tag>` does not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from your main branch.
3. **Understand the files**:
   - `README.md`: This context.
   - `prepare.py`: Fixed constants, data prep, tokenizer, evaluation. **Do not modify.**
   - `train.py`: The file the agent (or you) modifies. Architecture, optimizer, training loop.
4. **Verify data**: Ensure `~/.cache/autoresearch/` contains data shards and a tokenizer.
5. **Initialize results.tsv**: Create a `results.tsv` file with the header row:
   `commit	val_bpb	memory_gb	status	description` (Tab-separated).

## Running Experiments

Each experiment runs on the CPU for a **fixed time budget of 30 minutes** (training time, excluding startup/compilation).

**Rules:**
- **Modify ONLY `train.py`**: Everything is fair game: model architecture, optimizer, hyperparameters, batch size, etc.
- **Do NOT modify `prepare.py`**: It contains the fixed evaluation harness and data loading.
- **Goal**: Reach the lowest `val_bpb` (validation bits per byte).
- **RAM is a constraint**: Monitor your system's memory usage. Small increases are okay, but don't crash the system.

### Simplicity Criterion
All else being equal, simpler is better. A 0.001 improvement from deleting code is a huge win. A 0.001 improvement that adds 50 lines of complex logic is probably not worth the maintenance cost.

## The Autonomous Loop

Once setup is complete, the loop begins:

1. **Tune**: Hack `train.py` with an experimental idea.
2. **Commit**: `git commit -am "experiment description"`
3. **Run**: `python train.py > run.log 2>&1`
4. **Evaluate**: Check the results at the end of the log: `grep "^val_bpb:" run.log`
5. **Log**: Record the result in `results.tsv` (keep as untracked in git).
6. **Iterate**:
   - If `val_bpb` improved: Keep the commit and advance.
   - If it failed/worsened: `git reset --hard HEAD~1` and try a new idea.

## Output Summary format

At the end of a run, the script prints:
```
val_bpb:          0.997900
training_seconds: 1800.0
total_seconds:    1825.9
total_tokens_M:   15.5
num_params_M:     5.3
depth:            8
```

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for detailed information.

Additional attribution details can be found in the [NOTICE](NOTICE) file.
