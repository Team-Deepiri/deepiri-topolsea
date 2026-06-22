# Contributing to Deepiri Topolsea

## Development setup

```bash
# Rust
cargo build --workspace
cargo test --workspace

# Python
poetry install
poetry run pytest
```

## Architecture

See [README.md](README.md) for the crate layout.

## Commit conventions

- One logical change per commit
- Run `cargo test` and `poetry run pytest` before pushing
