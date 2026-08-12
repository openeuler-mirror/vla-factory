"""Shared constants for the VLA Factory training engine."""

# Inference metadata directory name (relative to output_dir)
INFERENCE_META_DIR = "inference_metadata"

# File names inside inference_metadata/
# assembly.json is the execution contract the inference engine runs; recipe.yaml
# supplies model selection + tunables. schema.json / norm_stats.json are
# readable copies for inspect and external tooling — the engine reads the
# descriptions out of the assembly instead, so a checkpoint can never run
# against a description it was not trained with.
ASSEMBLY_FILE = "assembly.json"
RECIPE_FILE = "recipe.yaml"
SCHEMA_FILE = "schema.json"
NORM_STATS_FILE = "norm_stats.json"

# Final model save directory (relative to output_dir)
FINAL_DIR = "final"
MODEL_WEIGHTS_FILE = "model.pt"
