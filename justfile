# evalcore - a generic, consumer-agnostic eval engine. Standard uv project.

# Sync the environment (editable install + dev/extras).
sync:
    uv sync --all-extras

# Run the engine unit tests with coverage.
test:
    uv run coverage run
    uv run coverage report

# Run the quickstart example as the implementation test (offline replay).
test-example:
    uv run python -m unittest discover -s examples -t . --buffer --verbose

# Run the full suite: engine units + the example implementation test.
test-all: test test-example

# Reformat and auto-fix.
lint:
    uv run ruff format
    uv run ruff check --fix

# Run the quickstart suite offline against recorded fixtures.
example:
    uv run evalcore --plugins examples.quickstart.graders gate --suite examples/quickstart/suite.yaml --mode replay

# Run the quickstart suite through the Python API (no CLI), offline.
example-api:
    uv run python -m examples.quickstart.run_eval

# Head-to-head A-vs-B win-rate over the quickstart suite (offline).
example-pairwise:
    uv run evalcore --plugins examples.quickstart.graders pairwise --suite examples/quickstart/suite.yaml --a baseline --b candidate --mode replay
