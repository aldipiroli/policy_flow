# Policy Flow

Behavioral Cloning via Flow Matching.

> Inspired by [Diffusion policy: Visuomotor policy learning via action diffusion](https://arxiv.org/pdf/2303.04137) and [CS 185/285](https://rail.eecs.berkeley.edu/deeprlcourse/).

### Clone and install dependencies

```
git clone https://github.com/aldipiroli/policy_flow.git
cd policy_flow && uv sync
```

### Training

```
uv run policy_flow/train.py policy_flow/configs/transformer_flow_policy.yaml
```

### Evaluate Policy

```
uv run policy_flow/evaluate.py policy_flow/configs/flow_policy.yaml --ckpt path/to/ckpt
```

### Qualitative Results
https://github.com/user-attachments/assets/51448aa6-73e8-4708-a60f-29e88fae2e1d

