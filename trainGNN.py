import os.path as osp
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from argparse import ArgumentParser
from sklearn.utils import shuffle
# 1. Imported the models
from models.BotGNN import BotRGCN, GCN, GAT, GraphSAGE, FAGCN, HGT, SimpleHGN, RGT, SRGAT
from utils import calc_metrics
import random
import os
import warnings

# Filter out the specific PyG warning
warnings.filterwarnings("ignore", message=".*without a 'pyg-lib' installation.*")

def auto_configure_tf32():
    """
    Automatically disables TF32 if a Hopper GPU (H100) is detected to prevent 
    cuBLAS execution failures on unaligned tensor dimensions. 
    Leaves TF32 enabled for Ampere (A100/A40) and older architectures.
    """
    if not torch.cuda.is_available():
        return

    # get_device_capability returns a tuple like (9, 0) for H100 or (8, 0) for A100
    major_capability, minor_capability = torch.cuda.get_device_capability()
    device_name = torch.cuda.get_device_name()

    if major_capability >= 9:
        print(f"⚠️ [Hardware Alert] Detected {device_name} (Compute {major_capability}.{minor_capability}).")
        print("   -> Disabling TF32 to prevent odd-dimension cuBLAS crashes.")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        print(f"✅ [Hardware Alert] Detected {device_name} (Compute {major_capability}.{minor_capability}).")
        print("   -> Keeping TF32 enabled for maximum performance.")

# Execute the check immediately
# auto_configure_tf32()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int = 0):
    
    # Standard libraries
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # cuDNN settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # MUST be False for determinism
    
    # GNN-specific: Force deterministic algorithms for scatter/gather ops
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)

parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='TwiBot-20')
parser.add_argument('--model', type=str, default='BotRGCN', 
                    choices=['BotRGCN', 'GCN', 'GAT', 'GraphSAGE', 'FAGCN', 'HGT', 'SimpleHGN', 'RGT', 'SRGAT'], 
                    help='Which baseline model to train')
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--num_rel', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument("--fanouts", nargs="+", type=int, default=[-1, -1], help="Neighbor sampling fanouts")
parser.add_argument("--ce_weights", nargs="+", type=float, default=[1.0, 1.0], help="weights for cross entropy loss")
parser.add_argument('--verbose', action='store_true', help="Print metrics during training epochs")

# 2. Toggle flags for normalization, residuals, and jumping knowledge (default is True)
parser.add_argument('--no_norm', action='store_false', dest='use_norm', help="Disable Layer Normalization")
parser.add_argument('--no_residual', action='store_false', dest='use_residual', help="Disable Residual connections")
parser.add_argument('--no_jk', action='store_false', dest='use_jk', help="Disable Jumping Knowledge")

parser.add_argument('--train_ratio', type=float, default=None, 
                    help="Ratio of total labeled nodes to use for training (e.g., 0.1 for 10%).")

args = parser.parse_args()
print(args)

def get_data(dataset_name,seed=42):
    path = '../datasets/' + dataset_name + '/processed_data'
    if not osp.exists(path): raise KeyError(f'processed_data not found at {path}')

    labels = torch.load(osp.join(path, 'label.pt'), weights_only=True).long()
    edge_index = torch.load(osp.join(path, 'edge_index.pt'), weights_only=True).long()
    edge_type = torch.load(osp.join(path, 'edge_type.pt'), weights_only=True).long()

    if dataset_name in ['TwiBot-20-Format22', 'TwiBot-22']:
        des_emb = torch.load(osp.join(path, 'des_tensor.pt'), weights_only=True)
        tweet_emb = torch.load(osp.join(path, 'tweets_tensor.pt'), weights_only=True)
        num_prop = torch.load(osp.join(path, 'num_properties_tensor.pt'), weights_only=True)
        cat_prop = torch.load(osp.join(path, 'cat_properties_tensor.pt'), weights_only=True)
        num_nodes = num_prop.size(0)

        train_idx = torch.load(osp.join(path, 'train_idx.pt'), weights_only=True)
        val_idx  = torch.load(osp.join(path, 'val_idx.pt'), weights_only=True)
        test_idx = torch.load(osp.join(path, 'test_idx.pt'), weights_only=True)

        # --- ABLATION LOGIC FOR TRAINING RATIO ---
        if args.train_ratio is not None:
            # Safely calculate target size relative ONLY to the training split
            target_size = int(args.train_ratio * len(train_idx))
            target_size = max(1, min(target_size, len(train_idx))) # safety clamp
            
            # Randomly subsample the training pool using the current run's seed
            shuffled_train = shuffle(train_idx.numpy(), random_state=seed)
            train_idx = torch.tensor(shuffled_train[:target_size], dtype=torch.long)
            
            if args.verbose:
                print(f"--- [Ablation] Adjusted Train Ratio to {args.train_ratio*100:.0f}% ({target_size} nodes) ---")

        if labels.size(0) < num_nodes:
            labels = torch.cat([labels, torch.full((num_nodes - labels.size(0),), 2, dtype=torch.long)])

        common_kwargs = dict(
            des_embedding=des_emb, 
            tweet_embedding=tweet_emb,
            num_property_embedding=num_prop, 
            cat_property_embedding=cat_prop,
            train_idx=train_idx, 
            val_idx=val_idx, 
            test_idx=test_idx,
            num_nodes=num_nodes, 
        )
    
    elif dataset_name in ['MGTAB']:
        if args.num_rel == 2:
            mask = (edge_type == 0) | (edge_type == 1)
            edge_index = edge_index[:, mask]
            edge_type = edge_type[mask]

        embedding = torch.load(osp.join(path, 'features.pt'), weights_only=True)
        num_nodes = embedding.size(0)

        sample_idx = shuffle(np.array(range(num_nodes)), random_state=seed)
        train_idx = sample_idx[:int(0.7 * num_nodes)]
        val_idx = sample_idx[int(0.7 * num_nodes):int(0.9 * num_nodes)]
        test_idx = sample_idx[int(0.9 * num_nodes):]

        if labels.size(0) < num_nodes:
            labels = torch.cat([labels, torch.full((num_nodes - labels.size(0),), 2, dtype=torch.long)])

        common_kwargs = dict(
            embedding=embedding, 
            train_idx=train_idx, 
            val_idx=val_idx, 
            test_idx=test_idx,
            num_nodes=num_nodes, 
        )
    else:
        raise KeyError(f'dataset name not found at {path}')
    
    data = Data(edge_index=edge_index, edge_type=edge_type, y=labels, **common_kwargs)
    return data

@torch.no_grad()
def evaluate(model, loader, desc="Val"):
    model.eval()
    preds = []
    labels = []
    
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
            
        if args.dataset in ['MGTAB']:
            out = model(
                edge_index=batch.edge_index, edge_type=batch.edge_type, 
                feat=batch.embedding
            )
        else:  
            out = model(
                edge_index=batch.edge_index, edge_type=batch.edge_type, 
                des=batch.des_embedding, tweet=batch.tweet_embedding, 
                num_prop=batch.num_property_embedding, cat_prop=batch.cat_property_embedding
            )
            
        batch_size = batch.batch_size
        preds.append(out[:batch_size].detach().cpu())
        labels.append(batch.y[:batch_size].detach().cpu())
        
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    
    metrics, _ = calc_metrics(labels, preds.float())
    return metrics

def train(seed=42):
    data = get_data(args.dataset, seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)
    
    # 3. Add generator and worker_init_fn to loader_kwargs
    loader_kwargs = dict(
        num_workers=0,
        persistent_workers=False, 
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    # loader_kwargs = dict(
    #     num_workers=0, 
    #     persistent_workers=False, 
    #     pin_memory=True,
    # )

    train_loader = NeighborLoader(
        data, 
        num_neighbors=args.fanouts, 
        batch_size=args.batch_size,
        input_nodes=data.train_idx,
        shuffle=True,
        **loader_kwargs
    )
    
    val_loader = NeighborLoader(
        data, 
        num_neighbors=[-1] * args.num_layers,
        batch_size=args.batch_size, 
        input_nodes=data.val_idx,
        shuffle=False,
        **loader_kwargs
    )
    
    test_loader = NeighborLoader(
        data, 
        num_neighbors=[-1] * args.num_layers,
        batch_size=args.batch_size,
        input_nodes=data.test_idx,
        shuffle=False,
        **loader_kwargs
    )

    use_mgtab = args.dataset in ['MGTAB']

    model_class = globals()[args.model]
    
    # 3. Handle Jumping Knowledge assignment dynamically
    jk_mode = 'cat' if args.use_jk else None
    
    model = model_class(
        embedding_dimension=args.hidden_dim,
        dropout=args.dropout, 
        heads=args.heads, 
        num_rel=args.num_rel,
        num_layers=args.num_layers,
        use_mgtab=use_mgtab,
        use_norm=args.use_norm,
        use_residual=args.use_residual,
        jk=jk_mode  # Passed to the model here
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = torch.tensor(args.ce_weights).to(device)
    loss_class = nn.CrossEntropyLoss(weight=weights)
    
    best_acc = 0.0
    best_state_acc = None
    best_epoch_acc = 0
    
    best_f1 = 0.0
    best_state_f1 = None
    best_epoch_f1 = 0
    
    best_f1_macro = 0.0
    best_state_f1_macro = None
    best_epoch_f1_macro = 0
    
    for epoch in range(args.max_epoch):
        model.train()
        total_loss = 0
        total_correct = 0
        total_train_samples = 0
        
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch = batch.to(device, non_blocking=True)
                    
            if use_mgtab:
                out = model(
                    edge_index=batch.edge_index, edge_type=batch.edge_type, 
                    feat=batch.embedding
                )
            else:  
                out = model(
                    edge_index=batch.edge_index, edge_type=batch.edge_type, 
                    des=batch.des_embedding, tweet=batch.tweet_embedding, 
                    num_prop=batch.num_property_embedding, cat_prop=batch.cat_property_embedding
                )
                
            batch_size = batch.batch_size
            out_seed = out[:batch_size]
            y_seed = batch.y[:batch_size]
                
            loss = loss_class(out_seed, y_seed)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            pred_seed = out_seed.argmax(dim=-1)
            total_correct += (pred_seed == y_seed).sum().item()
            total_train_samples += batch_size
            total_loss += loss.item()

        val_metrics = evaluate(model, val_loader, desc="Val")
        
        avg_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_train_samples
        
        if args.verbose:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_metrics.get('acc', 0):.4f}")

        if val_metrics.get('acc', 0) > best_acc:
            best_acc = val_metrics['acc']
            best_state_acc = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch_acc = epoch
            
        if val_metrics.get('f1', 0) > best_f1:
            best_f1 = val_metrics['f1']
            best_state_f1 = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch_f1 = epoch
            
        if val_metrics.get('f1-macro', 0) > best_f1_macro:
            best_f1_macro = val_metrics['f1-macro']
            best_state_f1_macro = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch_f1_macro = epoch

    test_res_acc, test_res_f1, test_res_f1_macro = None, None, None
    
    if best_state_acc is not None:
        model.load_state_dict(best_state_acc)
        # torch.save(model.state_dict(), f'{args.model}_{args.dataset}_best_acc_seed={seed}.pth')
        model.to(device)
        test_res_acc = evaluate(model, test_loader, desc="Test (Acc Checkpoint)")
        print(f'-- Best Accuracy Checkpoint (Epoch {best_epoch_acc}) | Test Acc: {test_res_acc.get("acc", 0):.4f}')
        
    if best_state_f1 is not None:
        model.load_state_dict(best_state_f1)
        model.to(device)
        test_res_f1 = evaluate(model, test_loader, desc="Test (F1 Checkpoint)")
        print(f'-- Best F1-Score Checkpoint (Epoch {best_epoch_f1}) | Test F1: {test_res_f1.get("f1", 0):.4f}')
        
    if best_state_f1_macro is not None:
        model.load_state_dict(best_state_f1_macro)
        model.to(device)
        test_res_f1_macro = evaluate(model, test_loader, desc="Test (Macro F1 Checkpoint)")
        print(f'-- Best F1-Macro Checkpoint (Epoch {best_epoch_f1_macro}) | Test F1-Macro: {test_res_f1_macro.get("f1-macro",0):.4f}')

    return test_res_acc, test_res_f1, test_res_f1_macro

if __name__ == '__main__':
    metrics_dict_acc = {}
    metrics_dict_f1 = {}
    metrics_dict_f1_macro = {}
    
    print(f"Training Baseline: {args.model} on {args.dataset}")
    
    for i in range(10):
        print(f'\n==== Run {i} ====')
        res_acc, res_f1, res_f1_macro = train(i+1)
        
        if res_acc:
            for key, value in res_acc.items():
                metrics_dict_acc.setdefault(key, []).append(value)
                
        if res_f1:
            for key, value in res_f1.items():
                metrics_dict_f1.setdefault(key, []).append(value)
                
        if res_f1_macro:
            for key, value in res_f1_macro.items():
                metrics_dict_f1_macro.setdefault(key, []).append(value)

    print(f"\n==== Final Results across 10 runs for {args.model} ====")
    
    print("\n--- Checkpoint: Best Validation Accuracy ---")
    for key, value_list in metrics_dict_acc.items():
        print(key, f'{np.mean(value_list)*100:.2f} ± {np.std(value_list)*100:.2f}')
        
    print("\n--- Checkpoint: Best Validation F1-Score ---")
    for key, value_list in metrics_dict_f1.items():
        print(key, f'{np.mean(value_list)*100:.2f} ± {np.std(value_list)*100:.2f}')
        
    if metrics_dict_f1_macro:
        print("\n--- Checkpoint: Best Validation Macro-F1 ---")
        for key, value_list in metrics_dict_f1_macro.items():
            print(key, f'{np.mean(value_list)*100:.2f} ± {np.std(value_list)*100:.2f}')