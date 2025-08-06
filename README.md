# akainu

This repository is configured for Linux with CUDA 12.6.
If your environment differs, please update `pyproject.toml` and delete `uv.lock`.

### Installing dependencies

```
curl -LsSf https://astral.sh/uv/install.sh | sh  # install uv if not already installed
uv --version  # e.g. uv 0.8.3
uv sync  # install dependencies
```

### Running commands

```
uv run python run.py
```
