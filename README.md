# MLERE

This repository is the official implementation of the paper **"Multi-dimensional Logical Reasoning for Event Relation Extraction"**.

We propose the **MLERE (Multi-dimensional Logical Reasoning Model for Event Relation Extraction)** framework to address the challenges of ambiguity in event understanding and the neglect of implicit relational constraints in Event Relation Extraction (ERE). MLERE consists of two key modules: the **Multi-dimensional Clues Discovery** module, which refines event representations by constructing an event-centric heterogeneous graph ; and the **Constraint-guided Reasoning** module, which leverages this graph to capture and integrate both inter- and intra-relational constraints, facilitating the interpretation of complex event dependencies and the identification of implicit event relations.

## 🚀Getting Start

### 1.Create a virtual environment

```
conda create -n mlere python=3.10
conda activate mlere
```

### 2.Install dependencies

```
pip install -r requirements.txt
```

### 3.Data Preparation

Use the scripts provided in the `data` directory to download the pre-processed datasets.

```
bash data/download_maven.sh
```

## 🛠️Training & Inference

This repository supports both **Joint Training** and **Single-Task Training** for specific relations (Temporal, Causal, Subevent, Coreference). Below are example commands to run the experiments.

### Joint Model Training

```
python joint/main.py \
    --seed 42 \
    --batch_size 4 \
    --lr 1e-4 \
    --bert_lr 2e-5 \
    --epochs 100 \
    --eval_steps 200
```

### Single-Task Training

You can also run training scripts for specific tasks within their respective directories.

```
python temporal/main.py \
    --batch_size 8 \
    --lr 1e-4 \
    --epochs 20
```

Note: If you only want to perform inference (evaluation), add the `--eval_only` flag to the commands above and specify the checkpoint path using `--load_ckpt`.

## 📑Citation

If you use the MLERE framework or this code in your research, please cite our paper:

 