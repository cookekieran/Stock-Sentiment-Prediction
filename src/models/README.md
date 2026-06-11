# Active Model Workflows

The recurrent market-state models previously stored here were archived on
2026-06-11 under `archive/legacy_experiments/gru_pipelines/`.

The active market-state architecture is documented in
`src/end_to_end_neurosymbolic/README.md`. It uses the transformer's causal
context window to form the state representation directly.

The remaining DeepSeek scripts are non-recurrent article-level comparison
models. They do not implement the final contextual market-state architecture.
