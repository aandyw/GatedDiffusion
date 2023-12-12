# Gated Diffusion


```bash
accelerate config
accelerate launch --mixed_precision="fp16" --multi_gpu train.py --name <RUN_NAME> --config configs/train.yaml
```

accelerate launch --main_process_port 29501 --mixed_precision="fp16" --multi_gpu train.py --name GD_run_8-freeze --config configs/train.yaml