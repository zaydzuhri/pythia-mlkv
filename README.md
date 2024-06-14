# MLKV: Multi-Layer Key-Value Sharing
# Experiments on EleutherAI's Pythia models

## Setup

```bash
git clone https://github.com/zaydzuhri/pythia-mlkv.git
cd pythia-mlkv
pip install -r requirements.txt
```

## Convert Pythia models to MQA/GQA/MLKV models

```bash
git lfs install
git clone https://huggingface.co/EleutherAI/pythia-160m-deduped
rm -rf pythia-160m-deduped/.git
python3 convert_to_mlkv.py --weights_path pythia-160m-deduped --num-key-value-layers 6 --num-key-value-heads 1
```

Here are all the 8+1 configs needed for all experiments:

| **Name** | **Num. of layers** | **Num. of attention heads** | **Num. of layers with KV heads (num-key-value-layers)** | **Num. of KV heads in a layer (num-key-value-heads)** | **Total num. of KV heads** | **Num. of parameters** |
|----------|--------------------|-----------------------------|----------------------------------|---------------------------------|---------------------------|------------------------|
| MHA-144  | 12                 | 12                          | 12                               | 12                              | 144                       | 160M                   |
| GQA-48   | 12                 | 12                          | 12                               | 4                               | 48                        | 160M                   |
| MLKV-48  | 12                 | 12                          | 4                                | 12                              | 48                        | 160M                   |
| MQA-12   | 12                 | 12                          | 12                               | 1                               | 12                        | 160M                   |
| MLKV-12  | 12                 | 12                          | 4                                | 3                               | 12                        | 160M                   |
| MLKV-6   | 12                 | 12                          | 6                                | 1                               | 6                         | 160M                   |
| MLKV-4   | 12                 | 12                          | 4                                | 1                               | 4                         | 160M                   |
| MLKV-2   | 12                 | 12                          | 2                                | 1                               | 2                         | 160M                   |
| MLKV-1   | 12                 | 12                          | 1                                | 1                               | 1                         | 160M                   |

## Uptraining

The dataset has been prepared to Huggingface, so you can directly uptrain:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 uptrain.py --output-dir pythia-160m-mlkv-6-b12-g2-v1 --model pythia-160m-deduped_mlkv_6_1 --batch-size 12 --gradient-accumulate-every 1 --learning-rate 6e-4 --warmup-ratio 0.2  --wandb pythia-160m-mlkv-6-b12-g2-v1
```
