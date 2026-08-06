"""Shared constants for the VLA Factory training engine."""

# Inference metadata directory name (relative to output_dir)
INFERENCE_META_DIR = "inference_metadata"

# File names inside inference_metadata/
RECIPE_FILE = "recipe.yaml"
SCHEMA_FILE = "schema.json"
NORM_STATS_FILE = "norm_stats.json"

# Final model save directory (relative to output_dir)
FINAL_DIR = "final"
MODEL_WEIGHTS_FILE = "model.pt"

# Per-model default-profile directory. Lives under vla_factory/recipe/,
# following Hydra's group convention (group "model" → recipe/model/<name>.yaml).
MODEL_CONFIG_DIR = "model"
