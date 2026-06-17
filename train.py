import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_networkx
from argparse import ArgumentParser
from tqdm import tqdm
from utils import calc_metrics, get_directional_heuristics
from models.BotHGAT import BotHGAT
import math
from sklearn.utils import shuffle
import random

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

    # --- ADD THIS: Force PyG scatter operations to be deterministic ---
    os.environ['PYTORCH_SCATTER_DETERMINISTIC'] = '1'
    
    # GNN-specific: Force deterministic algorithms for scatter/gather ops
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)#, warn_only=True)

parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='TwiBot-20')
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_rel', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.5)

parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--router_lr', type=float, default=1e-3, help="Learning rate for router")
parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay for optimizer")

parser.add_argument('--batch_size', type=int, default=4096, help="Increased batch size for GPU utilization")
parser.add_argument("--fanouts", nargs="+", type=int, default=[-1,-1], help="Neighbor sampling fanouts")

parser.add_argument('--warmup_epochs', type=int, default=200, help="Epochs to keep router detached to prevent early gradient noise")

parser.add_argument("--ce_weights", nargs="+", type=float, default=[1.0, 1.0])
parser.add_argument('--aux_weight', type=float, default=5.0, help="Weight for Router Aux Loss")

parser.add_argument('--verbose', action='store_true', help="Print metrics during training epochs")

parser.add_argument('--no_norm', action='store_false', dest='use_norm', help="Disable Layer Normalization")
parser.add_argument('--no_residual', action='store_false', dest='use_residual', help="Disable Residual connections")
parser.add_argument('--no_jk', action='store_false', dest='use_jk', help="Disable Jumping Knowledge")

args = parser.parse_args()
print(args)

def compute_cb_router_weights(edge_index, edge_type, labels_train, num_relations, beta=0.9999):
    src, dst = edge_index
    ls, ld = labels_train[src], labels_train[dst]
    
    valid_mask = (ls != 2) & (ld != 2)
    gt_types = ls[valid_mask] * 2 + ld[valid_mask]
    valid_edge_types = edge_type[valid_mask]
    
    relational_weights = torch.ones(num_relations, 4)
    
    print("\n--- CB Router Weights ---")
    for r in range(num_relations):
        mask_r = (valid_edge_types == r)
        gt_r = gt_types[mask_r]
        
        if gt_r.numel() == 0:
            print(f"Relation {r}: No labeled edges found. Defaulting to 1.0")
            continue
            
        counts = torch.bincount(gt_r, minlength=4).float()
        counts = torch.clamp(counts, min=1.0)
        
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights[0]
        weights = torch.clamp(weights, max=20.0)
        
        relational_weights[r] = torch.round(weights * 100) / 100
        print(f"Relation {r} Weights: {relational_weights[r].tolist()}")
        
    print("-------------------------\n")
    return relational_weights

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight 
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss) 
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def get_batch_masks(edge_index, node_labels):
    src, dst = edge_index
    ls, ld = node_labels[src], node_labels[dst]
    
    # Base valid mask: both nodes must be labeled (not 2)
    valid_mask = (ls != 2) & (ld != 2)
        
    gt_types = torch.full((edge_index.size(1),), -1, dtype=torch.long, device=edge_index.device)
    gt_types[valid_mask] = ls[valid_mask] * 2 + ld[valid_mask]
    
    return valid_mask, gt_types
    
def get_data(dataset_name, seed=42):
    path = '../datasets/' + dataset_name + '/processed_data'
    if not osp.exists(path): raise KeyError(f'processed_data not found at {path}')

    labels = torch.load(osp.join(path, 'label.pt'), weights_only=True).long()
    edge_index = torch.load(osp.join(path, 'edge_index.pt'), weights_only=True).long()
    edge_type = torch.load(osp.join(path, 'edge_type.pt'), weights_only=True).long()

    if dataset_name in ['TwiBot-20-Format22', 'TwiBot-22']:
        if args.verbose:
            print(f"Graph: {edge_index.shape[1]} edges.")

        des_emb = torch.load(osp.join(path, 'des_tensor.pt'), weights_only=True)
        tweet_emb = torch.load(osp.join(path, 'tweets_tensor.pt'), weights_only=True)
        num_prop = torch.load(osp.join(path, 'num_properties_tensor.pt'), weights_only=True)
        cat_prop = torch.load(osp.join(path, 'cat_properties_tensor.pt'), weights_only=True)
        num_nodes = num_prop.size(0)

        train_idx = torch.load(osp.join(path, 'train_idx.pt'), weights_only=True)
        val_idx  = torch.load(osp.join(path, 'val_idx.pt'), weights_only=True)
        test_idx = torch.load(osp.join(path, 'test_idx.pt'), weights_only=True)

        if labels.size(0) < num_nodes:
            labels = torch.cat([labels, torch.full((num_nodes - labels.size(0),), 2, dtype=torch.long)])

        labels_train = torch.full((num_nodes,), 2, dtype=torch.long)
        labels_train[train_idx] = labels[train_idx]

        common_kwargs = dict(
            des_embedding=des_emb, 
            tweet_embedding=tweet_emb,
            num_property_embedding=num_prop, 
            cat_property_embedding=cat_prop,
            train_idx=train_idx, 
            val_idx=val_idx, 
            test_idx=test_idx,
            num_nodes=num_nodes, 
            labels_train=labels_train
        )
    
    elif dataset_name in ['MGTAB']:
        if args.num_rel == 2:
            mask = (edge_type == 0) | (edge_type == 1)
            edge_index = edge_index[:, mask]
            edge_type = edge_type[mask]

        if args.verbose:
            print(f"Graph: {edge_index.shape[1]} edges.")

        embedding = torch.load(osp.join(path, 'features.pt'), weights_only=True)
        num_nodes = embedding.size(0)

        sample_idx = shuffle(np.array(range(num_nodes)), random_state=seed)
        
        train_idx = sample_idx[:int(0.7 * num_nodes)]
        val_idx = sample_idx[int(0.7 * num_nodes):int(0.9 * num_nodes)]
        test_idx = sample_idx[int(0.9 * num_nodes):]

        if labels.size(0) < num_nodes:
            labels = torch.cat([labels, torch.full((num_nodes - labels.size(0),), 2, dtype=torch.long)])

        labels_train = torch.full((num_nodes,), 2, dtype=torch.long)
        labels_train[train_idx] = labels[train_idx]

        common_kwargs = dict(
            embedding=embedding, 
            train_idx=train_idx, 
            val_idx=val_idx, 
            test_idx=test_idx,
            num_nodes=num_nodes, 
            labels_train=labels_train
        )

    else:
        raise KeyError(f'dataset name not found at {path}')
    
    # --- FIX: Enforce contiguous memory layout before passing to PyG ---
    edge_index = edge_index.contiguous()
    edge_type = edge_type.contiguous()
    
    data = Data(edge_index=edge_index, edge_type=edge_type, y=labels, **common_kwargs)
    return data

@torch.no_grad()
def evaluate(model, loader, desc="Val"):
    model.eval()
    preds = []
    labels = []
    
    total_router_correct = 0
    total_router_edges = 0
    
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
            
        valid_mask, gt_types_eval = get_batch_masks(batch.edge_index, batch.y)

        src, dst = batch.edge_index
        
        current_edge_attr = batch.edge_attr

        if args.dataset in ['MGTAB']:
            out, raw_router_logits = model(
                edge_index=batch.edge_index, edge_type=batch.edge_type, edge_attr=current_edge_attr,
                feat=batch.embedding
            )
        else:  
            out, raw_router_logits = model(
                edge_index=batch.edge_index, edge_type=batch.edge_type, edge_attr=current_edge_attr,
                des=batch.des_embedding, tweet=batch.tweet_embedding, 
                num_prop=batch.num_property_embedding, cat_prop=batch.cat_property_embedding
            )
            
        if valid_mask.sum() > 0:
            router_preds = raw_router_logits[valid_mask].argmax(dim=-1)
            total_router_correct += (router_preds == gt_types_eval[valid_mask]).sum().item()
            total_router_edges += valid_mask.sum().item()
            
        batch_size = batch.batch_size
        preds.append(out[:batch_size].detach().cpu())
        labels.append(batch.y[:batch_size].detach().cpu())
        
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    
    metrics, _ = calc_metrics(labels, preds.float())
    
    router_acc = total_router_correct / total_router_edges if total_router_edges > 0 else 0.0
    
    return metrics, router_acc

def train(seed=42):
    set_seed(seed)

    data = get_data(args.dataset, seed)
    
    heuristics_path = f"{args.dataset}_edge_attr_full.pt"

    if os.path.exists(heuristics_path):
        if args.verbose:
            print(f"Loading pre-computed topological heuristics from {heuristics_path}...")
        edge_attr = torch.load(heuristics_path, weights_only=True)
    else:
        print("Pre-computing relational directional heuristics... This will take a few minutes.")
        
        num_relations = args.num_rel 
        graphs = {}
        for r in range(num_relations):
            mask = (data.edge_type == r)
            edge_index_r = data.edge_index[:, mask]
            temp_data = Data(edge_index=edge_index_r, num_nodes=data.num_nodes)
            graphs[r] = to_networkx(temp_data, to_undirected=False)
    
        edge_attr_list = []
        edge_index_t = data.edge_index.t().tolist() 
        edge_type_list = data.edge_type.tolist() 
    
        for i, (u, v) in enumerate(tqdm(edge_index_t, desc="Calculating Relational Heuristics")):
            r = edge_type_list[i]
            G_r = graphs[r] 
            metrics = get_directional_heuristics(G_r, u, v)
            edge_attr_list.append(metrics)
        
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
        torch.save(edge_attr, heuristics_path)

    data.edge_attr = edge_attr

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
    
    jk_mode = 'cat' if args.use_jk else None

    model = BotHGAT(
        embedding_dimension=args.hidden_dim,
        dropout=args.dropout, 
        heads=args.heads, 
        num_rel=args.num_rel,
        num_layers=args.num_layers,
        use_mgtab=use_mgtab,
        use_norm=args.use_norm,
        use_residual=args.use_residual,
        jk=jk_mode
    ).to(device)
    
    router_params = []
    main_params = []
    for name, param in model.named_parameters():
        if 'edge_router' in name:
            router_params.append(param)
        else:
            main_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': main_params, 'lr': args.lr},
        {'params': router_params, 'lr': args.router_lr}
    ], weight_decay=args.weight_decay)

    weights = torch.tensor(args.ce_weights).to(device)
    loss_class = nn.CrossEntropyLoss(weight=weights)
    
    relational_router_weights = compute_cb_router_weights(
        data.edge_index, data.edge_type, data.labels_train, args.num_rel, beta=0.9999
    ).to(device)
    
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
        total_router_loss = 0
        total_correct = 0
        total_train_samples = 0
        
        total_router_correct = 0.0 
        total_router_edges = 0.0 
        
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch = batch.to(device, non_blocking=True)
            
            valid_mask, gt_types = get_batch_masks(batch.edge_index, batch.labels_train)
                    
            src, dst = batch.edge_index

            current_edge_attr = batch.edge_attr

            if args.dataset in ['MGTAB']:
                out, raw_router_logits = model(
                    edge_index=batch.edge_index, edge_type=batch.edge_type, edge_attr=current_edge_attr,
                    feat=batch.embedding, current_epoch=epoch, warmup=args.warmup_epochs
                )
            else:  
                out, raw_router_logits = model(
                    edge_index=batch.edge_index, edge_type=batch.edge_type, edge_attr=current_edge_attr,
                    des=batch.des_embedding, tweet=batch.tweet_embedding, 
                    num_prop=batch.num_property_embedding, cat_prop=batch.cat_property_embedding,
                    current_epoch=epoch, warmup=args.warmup_epochs
                )
                
            batch_size = batch.batch_size
            out_seed = out[:batch_size]
            y_seed = batch.y[:batch_size]
                
            loss_main = loss_class(out_seed, y_seed)
                
            if valid_mask.sum() > 0:
                l_router = 0.0
                active_relations = 0
                
                for r in range(args.num_rel):
                    mask_r = valid_mask & (batch.edge_type == r)
                    
                    if mask_r.sum() > 0:
                        weights_r = relational_router_weights[r]
                        
                        loss_fn_r = FocalLoss(weight=weights_r, gamma=2.0)
                        l_router += loss_fn_r(raw_router_logits[mask_r], gt_types[mask_r])
                            
                        active_relations += 1
                
                if active_relations > 0:
                    l_router = l_router / active_relations 
                else:
                    l_router = torch.tensor(0.0, device=device)
            else:
                l_router = torch.tensor(0.0, device=device)
                
            loss = loss_main + args.aux_weight * l_router
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            pred_seed = out_seed.argmax(dim=-1)
            total_correct += (pred_seed == y_seed).sum().item()
            total_train_samples += batch_size
            
            total_loss += loss.item()
            total_router_loss += l_router.item()
            
            if valid_mask.sum() > 0:
                router_pred = raw_router_logits[valid_mask].argmax(dim=-1)
                total_router_correct += (router_pred == gt_types[valid_mask]).float().sum().item()
                total_router_edges += valid_mask.sum().item()

        val_metrics, val_router_acc = evaluate(model, val_loader, desc="Val")
        
        avg_loss = total_loss / len(train_loader)
        avg_router = total_router_loss / len(train_loader)
        train_acc = total_correct / total_train_samples
        
        train_router_acc = total_router_correct / total_router_edges if total_router_edges > 0 else 0.0
        
        if args.verbose:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Router: {avg_router:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Train R-Acc: {train_router_acc:.4f} | "
                  f"Val Acc: {val_metrics.get('acc', 0):.4f} | Val R-Acc: {val_router_acc:.4f}")

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
        # torch.save(model.state_dict(), f'BotHGAT_{args.dataset}_best_acc_seed={seed}.pth')
        model.to(device)
        test_res_acc, router_acc_1 = evaluate(model, test_loader, desc="Test (Acc Checkpoint)")
        print(f'-- Best Accuracy Checkpoint (Epoch {best_epoch_acc}) | Test Acc: {test_res_acc.get("acc", 0):.4f} | R-Acc: {router_acc_1:.4f}')
        
    if best_state_f1 is not None:
        model.load_state_dict(best_state_f1)
        model.to(device)
        test_res_f1, router_acc_2 = evaluate(model, test_loader, desc="Test (F1 Checkpoint)")
        print(f'-- Best F1-Score Checkpoint (Epoch {best_epoch_f1}) | Test F1: {test_res_f1.get("f1", 0):.4f} | R-Acc: {router_acc_2:.4f}')
        
    if best_state_f1_macro is not None:
        model.load_state_dict(best_state_f1_macro)
        model.to(device)
        test_res_f1_macro, router_acc_3 = evaluate(model, test_loader, desc="Test (Macro F1 Checkpoint)")
        print(f'-- Best F1-Macro Checkpoint (Epoch {best_epoch_f1_macro}) | Test F1-Macro: {test_res_f1_macro.get("f1-macro", 0):.4f} | R-Acc: {router_acc_3:.4f}')

    return test_res_acc, test_res_f1, test_res_f1_macro

if __name__ == '__main__':
    metrics_dict_acc = {}
    metrics_dict_f1 = {}
    metrics_dict_f1_macro = {}

    print(f"Training BotHGAT on {args.dataset}")
    
    for i in range(2):
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

    print(f"\n==== Final Results across 10 runs for BotHGAT ====")
    
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