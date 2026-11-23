# Contributing

1. Fork, create a branch, `pip install -e ".[dev]"`.
2. Run `ruff check src tests` and `pytest` before opening a pull request.
3. New intervention: subclass `Intervention` (`sample_params`, `_apply`), register it in
   `interventions/__init__.py`, add a test that it changes the signal measurably.
4. New feature: add it to `features/`, add a description to `DESCRIPTIONS` so evidence text
   stays readable, and make sure `group_of` classifies it.
5. New dataset adapter: only metadata scanning and splitting code; do not commit audio.

Good first issues: real replay recordings, neural VC adapters, ASVspoof/CodecFake
adapters, additional languages, a cross-corpus protocol runner.

By contributing you agree your work is released under the MIT license. Be respectful and
constructive in all discussions.

Maintainers: see [docs/repository_setup.md](docs/repository_setup.md) for releases and repository settings.
