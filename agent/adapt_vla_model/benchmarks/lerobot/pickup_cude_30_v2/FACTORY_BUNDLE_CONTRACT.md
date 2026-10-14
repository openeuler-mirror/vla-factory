# Factory Bundle Contract: vla-factory-inference/v1

This contract applies to the offline LeRobot benchmark
`leisaac/pickup_cude_30_v2`. The Adapter must create a self-contained
`.judge/factory_checkpoint/` from the Judge-provided source checkpoint, plus
`factory_bundle.json`, an installer, and a smoke test.

## Judge handoff manifest

Create this exact top-level `.judge/factory_bundle.json` shape. It is a Judge
handoff manifest, not a description of the model's internal weights or
architecture; put the latter in `factory_checkpoint/inference_metadata/`.

```json
{
  "format": "vla-factory-inference/v1",
  "factory_checkpoint": "factory_checkpoint",
  "install_script": "install_smolvla.sh",
  "smoke_test": "smoke_test_smolvla.py"
}
```

The Judge rejects a manifest missing any of these fields, an identity other
than exactly `vla-factory-inference/v1`, unsafe artifact paths, an incomplete
checkpoint, or a `weights.json` format other than `bare_full`,
`lora_wrapped_full`, or `lora_delta`. Extra model metadata is ignored and does
not replace these fields.

Judge owns runtime construction. Before invoking the install script it supplies
the Judge-only Python 3.12 `JUDGE_BOOTSTRAP_PYTHON`, the role-isolated target
`JUDGE_RUNTIME_DIR`, and the resulting `JUDGE_PYTHON`. The installer must
create a fresh venv at that target from the bootstrap Python, then install and
run only through `JUDGE_PYTHON`. Do not search `$HOME`, use `/home/opencode`,
or put a Python path in `factory_bundle.json`. Judge uses only that role
runtime for smoke and parity capture. The installer is solely responsible for
installing the complete set of runtime dependencies required by the Factory
bundle; Judge does not inject model or framework dependencies into it.

## LeRobot boundary

The Factory checkpoint is evaluated with the recorded host observation:

```text
front: uint8[480, 640, 3]
wrist: uint8[480, 640, 3]
shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos,
wrist_roll.pos, gripper.pos: six scalar float32 joint positions
instruction: "Lift the red cube up."
```

The state and action order is exactly the listed order. The Factory model
returns `float[action_horizon, 6]` joint positions. Preserve the upstream
model's own input/output widths and processor behaviour; map this host
boundary outside the model.

## Bundle and offline restoration

The bundle layout and self-contained requirements are unchanged:

```text
factory_checkpoint/
├── model.pt
├── weights.json
├── model_artifacts/             # config, processor, tokenizer, etc. when needed
└── inference_metadata/
    ├── assembly.json
    └── recipe.yaml
```

`weights.json` declares `bare_full`, `lora_wrapped_full`, or `lora_delta`.
`assembly.json` must record `inference_needs_base_checkpoint: false`. All
model-owned assets needed offline belong under `model_artifacts/`, with only
checkpoint-relative paths in metadata. The Judge sets a fresh offline
`HF_HOME`; a user cache, network access, or absolute conversion paths are not
available.

The smoke test must load `InferenceEngine(factory_checkpoint)` and run one
valid LeRobot-format inference through `LerobotHostObsAdapter`, with the six
state keys and two cameras above. It must not use the RoboTwin wrapper.
