from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, \
    roc_auc_score, precision_recall_curve, auc
import torch
import torch.nn.functional as func

import networkx as nx
import math

def get_directional_heuristics(G, u, v):
    """
    Computes directional Jaccard and Adamic-Adar heuristics for a directed edge (u -> v).
    
    Args:
        G (nx.DiGraph): The directed Twitter network.
        u (node): Source node.
        v (node): Destination node.
        
    Returns:
        tuple: (out_jaccard, in_jaccard, out_aa, in_aa)
    """
    
    # 1. Get Out-edges (Following) and In-edges (Followers) as sets
    out_u = set(G.successors(u))
    out_v = set(G.successors(v))
    
    in_u = set(G.predecessors(u))
    in_v = set(G.predecessors(v))

    # --- JACCARD COEFFICIENTS ---
    
    # Out-Jaccard (Co-Following: Do they follow the same people?)
    out_intersection = len(out_u.intersection(out_v))
    out_union = len(out_u.union(out_v))
    out_jaccard = out_intersection / out_union if out_union > 0 else 0.0
    
    # In-Jaccard (Co-Followers: Do they share the same audience?)
    in_intersection = len(in_u.intersection(in_v))
    in_union = len(in_u.union(in_v))
    in_jaccard = in_intersection / in_union if in_union > 0 else 0.0

    # --- ADAMIC-ADAR COEFFICIENTS ---
    
    out_aa = 0.0
    in_aa = 0.0
    
    # Out-AA (Shared Followees Penalized by Popularity / In-degree)
    common_out = out_u.intersection(out_v)
    for z in common_out:
        in_deg_z = G.in_degree(z)
        # log(1) is 0, which causes division by zero. 
        # If a shared node has only 1 follower, we skip the penalty or ignore it.
        if in_deg_z > 1:
            out_aa += 1.0 / math.log(in_deg_z)
            
    # In-AA (Shared Followers Penalized by Activeness / Out-degree)
    common_in = in_u.intersection(in_v)
    for z in common_in:
        out_deg_z = G.out_degree(z)
        # Again, prevent division by zero for inactive accounts
        if out_deg_z > 1:
            in_aa += 1.0 / math.log(out_deg_z)

    return out_jaccard, in_jaccard, out_aa, in_aa

def null_metrics():
    return {
        'acc': 0.0,
        'f1': 0.0,
        'f1-macro': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'mcc': 0.0,
        'roc-auc': 0.0,
        'pr-auc': 0.0
    }


def calc_metrics(y, pred):
    assert y.dim() == 1 and pred.dim() == 2
    if torch.any(torch.isnan(pred)):
        metrics = null_metrics() # Assuming null_metrics() is defined elsewhere
        plog = ''
        for key, value in metrics.items():
            plog += ' {}: {:.6f}'.format(key, value)
        return metrics, plog

    pred = func.softmax(pred, dim=-1)
    pred_label = torch.argmax(pred, dim=-1)
    pred_score = pred[:, -1] # Assumes positive class (Bot) is at index 1

    y = y.to('cpu').numpy().tolist()
    pred_label = pred_label.to('cpu').tolist()
    pred_score = pred_score.to('cpu').tolist()

    precision, recall, _thresholds = precision_recall_curve(y, pred_score)

    metrics = {
        'acc': accuracy_score(y, pred_label),
        
        # Binary Metrics: Good for testing operational bot detection capability
        'f1': f1_score(y, pred_label, average='binary', zero_division=0),
        'precision': precision_score(y, pred_label, average='binary', zero_division=0),
        'recall': recall_score(y, pred_label, average='binary', zero_division=0),
        
        # Macro Metrics: CRITICAL for validation/early stopping on imbalanced graphs
        'f1-macro': f1_score(y, pred_label, average='macro', zero_division=0),
        
        # Robust Metrics for Imbalance
        'mcc': matthews_corrcoef(y, pred_label),
        'roc-auc': roc_auc_score(y, pred_score),
        'pr-auc': auc(recall, precision)
    }

    plog = ''
    # Log the most important metrics for monitoring training health
    for key in ['acc', 'f1-macro', 'f1', 'mcc', 'roc-auc']:
        plog += ' {}: {:.6f}'.format(key, metrics[key])

    return metrics, plog



