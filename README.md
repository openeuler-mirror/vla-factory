# vla-factory

A unified fine-tuning framework for Vision-Language-Action (VLA) models on openEuler. It unifies mainstream VLA models (PI0, OpenVLA, ACT, etc.) and multiple data formats (LeRobot v2/v3, HDF5, ROSbag, etc.), supporting LoRA/Freeze/Full/Selective fine-tuning strategies with native Ascend NPU training path, producing weights deployable downstream via IB_Robot.