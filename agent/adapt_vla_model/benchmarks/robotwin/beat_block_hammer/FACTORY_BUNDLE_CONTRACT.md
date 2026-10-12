# Factory Bundle Contract: vla-factory-inference/v1

The Adapter Agent must create `.judge/factory_checkpoint/` from the
Judge-provided source checkpoint. This is a VLA-Factory native inference bundle, not an
upstream Hugging Face checkpoint. It must not create `policy_impl.py`,
`policy_spec.json`, or `runtime.json`.

## RoboTwin boundary

The Factory bridge receives `robotwin_observation` with `head_camera`,
`left_camera`, and `right_camera` RGB `uint8[240, 320, 3]` frames, plus
`joint_action.vector: float32[14]` in this order:
`left_arm[6] + left_gripper[1] + right_arm[6] + right_gripper[1]`.
It returns a qpos chunk `float[action_horizon, 14]`, where `action_horizon`
comes from the selected checkpoint. Map this boundary through the saved Factory
assembly; do not construct a TCP policy endpoint.

## Judge handoff manifest

Create this exact top-level `.judge/factory_bundle.json` shape. These are the
only fields the Judge uses to locate the bundle and run its lifecycle; model
metadata belongs in `factory_checkpoint/inference_metadata/`, not here.

```json
{
  "format": "vla-factory-inference/v1",
  "factory_checkpoint": "factory_checkpoint",
  "install_script": "install_model.sh",
  "smoke_test": "smoke_test_model.py"
}
```

`format` must be exactly `vla-factory-inference/v1`. The three named artifact
paths are relative to `.judge/`. Do not include a `python` field: Judge owns
the runtime. It supplies Judge-only Python 3.12 in
`JUDGE_BOOTSTRAP_PYTHON`, the role-isolated target `JUDGE_RUNTIME_DIR`, and
the resulting `JUDGE_PYTHON`. The installer creates a fresh venv at that
target from the bootstrap Python, then installs and runs only through
`JUDGE_PYTHON`; it must not search `$HOME`, use `/home/opencode`, or provide a
fallback runtime path. The installer is solely responsible for installing the
complete set of runtime dependencies required by the Factory bundle; Judge
does not inject model or framework dependencies into that environment. Judge
verifies the resulting executable and invokes the smoke test and parity runners
with it.

## Bundle layout and offline runtime

Its required layout is:

```text
factory_checkpoint/
├── model.pt
├── weights.json
├── model_artifacts/             # optional: config, processor, tokenizer, etc.
│   └── ...
└── inference_metadata/
    ├── assembly.json
    └── recipe.yaml
```

`weights.json` declares one of `bare_full`, `lora_wrapped_full`, or
`lora_delta`. `assembly.json` and `recipe.yaml` are the resolved execution
contract consumed by `vla_factory.inference.InferenceEngine`.

Judge invokes the installer, smoke test, and parity capture with a fresh
`$HF_HOME`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`. A native bundle
must restore inference without a user cache, network access, host-local paths,
or a runtime created only inside the Agent sandbox. Persist every model-owned
asset needed in addition to the weights in `factory_checkpoint/model_artifacts/`.
Record only checkpoint-relative asset paths in metadata.

The smoke test must load `InferenceEngine(factory_checkpoint)` and make one
RoboTwin-format inference. This native-bundle contract requires
`inference_needs_base_checkpoint: false`: `model.pt` plus optional
`model_artifacts/` must restore the model offline.
