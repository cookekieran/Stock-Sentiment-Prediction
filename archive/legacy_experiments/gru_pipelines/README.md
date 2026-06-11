# Archived GRU Pipelines

Status: superseded by the transformer-context architecture on 2026-06-11.

This directory preserves all source workflows and generated model outputs whose
market-state representation depended on a GRU. They remain available for
dissertation traceability, but they are not part of the final project.

The archived source keeps its former repository-relative layout under `src/`.
The `docs/` directory preserves the superseded final-study protocol and results.
The `model_outputs/` directory contains outputs from the archived GRU trainers
and is intentionally ignored by Git.

The active replacement is documented in
`src/end_to_end_neurosymbolic/README.md`. It constructs the state representation
directly with causal self-attention over an ordered ticker-history context
window; it does not add a recurrent state module after the transformer.
