"""Shared constants for the VLA Factory training engine."""

# Inference metadata directory name (relative to output_dir)
INFERENCE_META_DIR = "inference_metadata"

# File names inside inference_metadata/. ``assembly.json`` is the complete
# execution contract (including schema and normalization statistics), while
# ``recipe.yaml`` supplies model selection and tunables.
ASSEMBLY_FILE = "assembly.json"
RECIPE_FILE = "recipe.yaml"

# Declares whether weights are a bare full model, a wrapped LoRA model, or a
# LoRA delta.  It makes loading independent of the checkpoint directory name.
WEIGHTS_META_FILE = "weights.json"
MODEL_WEIGHTS_FILE = "model.pt"
