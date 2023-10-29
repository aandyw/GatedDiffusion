# Gated Diffusion

```
conda env create -f environment.yaml
conda activate ip2p
bash scripts/download_checkpoints.sh
```

### Training
```
python main.py --name default --base configs/train.yaml --train --gpus 0,1,2,3,4,5,6,7
```

python main.py --name GDv2_debug_01 --base configs/train.yaml --train --gpus 7,