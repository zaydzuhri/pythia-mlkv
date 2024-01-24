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
python3 convert_to_mlkv.py --weights_path pythia-160m-deduped --num-key-value-layers 4 --num-key-value-heads 3
```

Here are all the 6 configs needed for all experiments:

| **Name** | **Num. of layers** | **Num. of attention heads** | **Num. of layers with KV heads (num-key-value-layers)** | **Num. of KV heads in a layer (num-key-value-heads)** | **Total num. of KV heads** | **Num. of parameters** |
|----------|--------------------|-----------------------------|----------------------------------|---------------------------------|---------------------------|------------------------|
| MHA-144  | 12                 | 12                          | 12                               | 12                              | 144                       | 160M                   |
| GQA-48   | 12                 | 12                          | 12                               | 4                               | 48                        | 160M                   |
| MLKV-48  | 12                 | 12                          | 4                                | 12                              | 48                        | 160M                   |
| MQA-12   | 12                 | 12                          | 12                               | 1                               | 12                        | 160M                   |
| MLKV-12  | 12                 | 12                          | 4                                | 3                               | 12                        | 160M                   |
| MLKV-2   | 12                 | 12                          | 2                                | 1                               | 2                         | 160M                   |

## Uptraining

The dataset has been prepared to Huggingface, so you can directly uptrain:

```bash
python3 uptrain.py --output-dir pythia-160m-mlkv-12 --model pythia-160m-deduped_mlkv_4_3 --batch-size 32 --learning-rate 6e-4 --warmup-steps 1000 --gradient-accumulate-every 4 --wandb pythia-160m-mlkv-12-v1
```
