# BASD
## Environment requirement 
```
dgl
prettytable
scikit-learn
tensorboard
tensorflow
tensorflow-gan
torch
torch-geometric
tqdm
wandb
```
and dependencies from  https://github.com/uoguelph-mlrg/GGM-metrics, https://github.com/hheidrich/CELL and https://github.com/ehoogeboom/multinomial_diffusion.

## Dataset Download
Download the dataset files and place them in the graphs folder.

1、DGraph: https://dgraph.xinye.com/

2、Elliptic: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

3、T-Finance: https://github.com/squareRoot3/Rethinking-Anomaly-Detection

4、Reddit and Photo: https://github.com/mala-lab/GGAD

## 1、Generate Subgraph Dataset
```
#!/bin/bash
cd createSubGraphDataset
python createDataset.py
```
## 2、Generate GAE Weights
You can either generate the GAE weights or directly use the pre-generated weights in the weight folder.
```
#!/bin/bash
python saveGAEdataset.py
python trainGAE.py
```
## 3、Train Diffusion Model
```
#!/bin/bash
python -u train.py --epochs 50 --num_generation 64 --diffusion_dim 64 --diffusion_steps 128 --device cuda:0 --dataset photo --batch_size 64 --clip_value 1 --lr 1e-4 --optimizer adam --final_prob_edge 1 0 --sample_time_method importance --check_every 5 --eval_every 5 --noise_schedule linear --dp_rate 0.1 --loss_type vb_ce_xt --arch TGNN_embedding_guided --parametrization xt --empty_graph_sampler empirical --num_heads 8 8 8 8 1 --log_wandb False
```

## 4、Graph Data Synthesis
```
#!/bin/bash
python evaluate.py --run_name 2025-01-20_18-29-35 --dataset photo --num_samples 8 --checkpoints 50
```

