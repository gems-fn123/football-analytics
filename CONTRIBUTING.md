# Contributing

## Setup

```bash
make setup
make test
```

## Rules that matter

1. **No GPL or AGPL imports in `src/footy/`.** Wrap them in `scripts/` and call via
   subprocess. See `docs/licensing.md`.
2. **Stage boundaries are schema-enforced.** If you change what a stage emits, update
   `src/footy/schemas.py` and `docs/data_contracts.md` in the same PR.
3. **Nullable beats guessed.** If OCR is not confident, emit null. A wrong shirt number
   silently corrupts every per-player metric downstream.
4. **Validate on a clip, not a claim.** PRs that change a CV stage should state which
   clip was used and what changed numerically.

## Adding a backend

Backends are selected by config, not by code edits. Add a yaml under `configs/<stage>/`,
implement the branch in the stage's `setup()`, and record the licence in `NOTICE.md`.
