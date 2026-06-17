# BotHGAT: Human–Bot Interaction Perception

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/PyTorch%20Geometric-%233C2179.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch Geometric">
</p>

This is the official repository for the paper:  
**Human–Bot Interaction Perception: Heterophily Gated Attention Networks for Social Bot Detection**

## 📌 Repository Structure

```text
BotHGAT-master/
├── models/
│   ├── BotHGAT.py        # Core BotHGAT model
│   ├── BotGNN.py         # Baseline GNN models
│   └── layer.py          # Custom graph layers for baseline models
├── train.py              # Main training and evaluation script for BotHGAT
├── trainGNN.py           # Training script for all baseline GNN models
└── utils.py              # Evaluation metrics and heuristic extraction utilities
```

## ⚙️ Requirements

Ensure you have the following dependencies installed:

- Python 3.8+
- PyTorch
- PyTorch Geometric (PyG)
- NetworkX
- scikit-learn

*Tip: For optimal execution and hardware acceleration, ensure you have `pyg-lib`, `torch-scatter`, and other associated PyG dependencies properly installed.*

## 🚀 Usage

### 1. Datasets

The framework expects processed data located in the `../datasets/<dataset_name>/processed_data/` directory relative to the repository root. Required files include `label.pt`, `edge_index.pt`, `edge_type.pt`, and node feature tensors (`des_tensor.pt`, `tweets_tensor.pt`, etc.). 

For detailed data preprocessing instructions, please refer to the [BotRGCN repository](https://github.com/LuoUndergradXJTU/TwiBot-22/tree/master/src/BotRGCN).

**Download Links:**
We gratefully acknowledge the creators of the following datasets used in this work:
- 📊 **TwiBot-20**: [GitHub Repository](https://github.com/BunsenFeng/TwiBot-20)
- 📊 **TwiBot-22**: [GitHub Repository](https://github.com/LuoUndergradXJTU/TwiBot-22)
- 📊 **MGTAB**: [GitHub Repository](https://github.com/GraphDetec/MGTAB)

### 2. Training BotHGAT

To train the BotHGAT model, execute the `train.py` script:

```bash
python train.py --dataset TwiBot-20
```

> **Note:** On the very first run, the script will automatically pre-compute and cache topological heuristics (out/in Jaccard, out/in Adamic-Adar). This may take a few minutes depending on the graph size.

The script supports various hyperparameters. For example:
```bash
python train.py --dataset TwiBot-20 --hidden_dim 128 --heads 4 --num_layers 2 --max_epoch 200 --batch_size 4096
```

### 3. Training Baselines

To benchmark against standard baseline models, use `trainGNN.py`:

```bash
python trainGNN.py --dataset TwiBot-20 --model BotRGCN
```

**Available baseline models** (`--model`):  
`BotRGCN`, `GCN`, `GAT`, `GraphSAGE`, `FAGCN`, `HGT`, `SimpleHGN`, `RGT`, `SRGAT`.

---

## 📖 Citation

If you find this repository useful in your research, please consider citing our work:

```bibtex
@article{ren_humanbot_2026,
  title   = {Human–Bot Interaction Perception: Heterophily Gated Attention Networks for Social Bot Detection},
  author  = {Ren, Guojing and Xu, Xiao-Ke and Di, Zengru},
  year    = {2026}
}
```
