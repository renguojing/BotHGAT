import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_add 
from torch_geometric.utils import softmax
from torch_geometric.nn import JumpingKnowledge

class EdgeRouter(nn.Module):
    """
    Predicts interaction weights using node features + structural topology.
    """
    def __init__(self, in_dim, structural_dim=4, num_types=4):
        super().__init__()
        self.attr_proj = nn.Sequential(
            nn.BatchNorm1d(structural_dim), 
            nn.Linear(structural_dim, 32),  
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, num_types)
        )

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        edge_feat = torch.cat([x[src], x[dst]], dim=-1)
        
        expanded_attr = self.attr_proj(edge_attr)
        edge_feat = torch.cat([edge_feat, expanded_attr], dim=-1)
            
        return self.mlp(edge_feat)
    

class GATExpertWeighted(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.3):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.head_dim = out_dim // heads
        self.dropout = dropout

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.att = nn.Parameter(torch.empty(heads, 2 * self.head_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, edge_weights):
        if edge_index.size(1) == 0:
            return torch.zeros(x.size(0), self.out_dim, device=x.device)

        src, dst = edge_index
        h = self.W(x).view(-1, self.heads, self.head_dim)
        h_src, h_dst = h[src], h[dst]

        cat = torch.cat([h_dst, h_src], dim=-1)
        score = (cat * self.att).sum(dim=-1)
        score = F.leaky_relu(score, 0.2)
        
        # Mask the topological score before softmax (Crucial for routed logic)
        mask_penalty = -1e9 * (1.0 - edge_weights.view(-1, 1))
        score = score + mask_penalty
        
        alpha = softmax(score, dst, num_nodes=None) 
        # Attention Dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        weighted_msg = (h_src * alpha.unsqueeze(-1)) * edge_weights.view(-1, 1, 1)
        out = scatter_add(weighted_msg, dst, dim=0, dim_size=x.size(0))
        return out.reshape(-1, self.out_dim)
    



class RoutedExpertConv(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.3):
        super().__init__()
        
        self.num_expert = 4
        self.specialized_experts = nn.ModuleList()

        for _ in range(self.num_expert):
            self.specialized_experts.append(GATExpertWeighted(in_dim, out_dim, heads, dropout))

        self.gates = nn.ModuleList([
            nn.Linear(in_dim + out_dim, out_dim, bias=False) 
            for _ in range(self.num_expert)
        ])

    def forward(self, x, edge_index, router_probs):
        h_sum = torch.zeros(x.size(0), self.specialized_experts[0].out_dim, device=x.device)
        
        if edge_index.size(1) == 0:
            return h_sum

        for r in range(self.num_expert):
            weights = router_probs[:, r]
            if weights.sum() > 1e-5:
                h_expert = self.specialized_experts[r](x, edge_index, weights)
                
                gate_input = torch.cat([x, h_expert], dim=-1)
                gate_proj = self.gates[r](gate_input)
                gate = torch.sigmoid(gate_proj)
                
                h_sum += gate * h_expert

        return h_sum
    

class RelationalRoutedExpertConv(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.3, num_rel=2):
        super().__init__()
        self.num_rel = num_rel
        self.routed_experts = nn.ModuleList([
            RoutedExpertConv(in_dim, out_dim, heads, dropout)
            for _ in range(self.num_rel)
        ])

        self.rel_gates = nn.ModuleList([
            nn.Linear(in_dim + out_dim, out_dim, bias=False) for _ in range(self.num_rel)
        ])

        self.W_self = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_indices, router_probs_list):
        h_sum = torch.zeros(x.size(0), self.W_self.out_features, device=x.device)
        
        for r in range(self.num_rel):
            h_routed_expert = self.routed_experts[r](x, edge_indices[r], router_probs_list[r])
            
            gate_input = torch.cat([x, h_routed_expert], dim=-1)
            gate_proj = self.rel_gates[r](gate_input)
            gate = torch.sigmoid(gate_proj)
            
            h_sum += gate * h_routed_expert

        h_self = self.W_self(x)
        out = h_self + h_sum
        
        return out
    

class RelationalRoutedExpert(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.3, num_rel=2, num_layers=2, 
                 use_norm=True, use_residual=True, jk='cat'):
        super().__init__()
        self.num_rel = num_rel
        self.dropout = dropout
        self.use_norm = use_norm
        self.use_residual = use_residual
        self.num_layers = num_layers
        self.jk_mode = jk
        
        self.edge_router = nn.ModuleList([EdgeRouter(in_dim, structural_dim=4) for _ in range(self.num_rel)])

        self.gnn_layers = nn.ModuleList()
        if self.use_norm:
            self.norm_layers = nn.ModuleList()
            
        for _ in range(num_layers):
            self.gnn_layers.append(RelationalRoutedExpertConv(in_dim, out_dim, heads, dropout, num_rel))
            if self.use_norm:
                self.norm_layers.append(nn.LayerNorm(out_dim))
                
        if self.jk_mode is not None:
            self.jk = JumpingKnowledge(self.jk_mode, channels=out_dim, num_layers=num_layers)
    
    def forward(self, x, edge_index, edge_type, edge_attr, tau, current_epoch=0, warmup=20):
        router_probs_list = []
        edge_indices = []
        
        full_raw_logits = torch.zeros(edge_index.size(1), 4, device=x.device)
        
        for r in range(self.num_rel):
            mask = (edge_type == r)
            edge_index_r = edge_index[:, mask]
            edge_attr_r = edge_attr[mask]
            edge_indices.append(edge_index_r)
            
            raw_prob = self.edge_router[r](x, edge_index_r, edge_attr_r)
            full_raw_logits[mask] = raw_prob
            
            # --- WARM-UP DETACH LOGIC ---
            if current_epoch < warmup:
                # Protect the router from main network noise early on
                router_signal = raw_prob.detach() 
            else:
                # Allow joint end-to-end training later
                router_signal = raw_prob 
            
            if self.training:
                logit = F.gumbel_softmax(router_signal, tau=tau, hard=True, dim=-1)
            else:
                indices = router_signal.argmax(dim=-1)
                logit = F.one_hot(indices, num_classes=router_signal.size(-1)).float()
                
            router_probs_list.append(logit)

        x = F.dropout(x, p=self.dropout, training=self.training)

        xs = []
        for i in range(self.num_layers):
            x_in = self.norm_layers[i](x) if self.use_norm else x
            h = self.gnn_layers[i](x_in, edge_indices, router_probs_list)
            h = F.elu(h)  
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h if self.use_residual else h
            xs.append(x)
            
        if self.jk_mode is not None:
            x = self.jk(xs)
        else:
            x = xs[-1]
        
        return x, full_raw_logits


class BotHGAT(nn.Module):
    def __init__(self, embedding_dimension=128, dropout=0.5, heads=4, num_rel=2, num_layers=2, 
                des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3, feat_size=788, 
                use_mgtab=False, use_norm=True, use_residual=True, jk='cat'):
        super().__init__()
        self.dropout = dropout
        self.tau = 1.0
        self.use_mgtab = use_mgtab
        
        if self.use_mgtab:
            self.linear_relu_feat = nn.Sequential(nn.Linear(feat_size, embedding_dimension), nn.LeakyReLU())
        else:
            self.linear_relu_des = nn.Sequential(nn.Linear(des_size, embedding_dimension // 4), nn.LeakyReLU())
            self.linear_relu_tweet = nn.Sequential(nn.Linear(tweet_size, embedding_dimension // 4), nn.LeakyReLU())
            self.linear_relu_num_prop = nn.Sequential(nn.Linear(num_prop_size, embedding_dimension // 4), nn.LeakyReLU())
            self.linear_relu_cat_prop = nn.Sequential(nn.Linear(cat_prop_size, embedding_dimension // 4), nn.LeakyReLU())

        self.linear_relu_input = nn.Sequential(nn.Linear(embedding_dimension, embedding_dimension), nn.LeakyReLU())
        
        self.gnn = RelationalRoutedExpert(
            embedding_dimension, embedding_dimension, heads, dropout, num_rel, num_layers, 
            use_norm=use_norm, use_residual=use_residual, jk=jk
        )
            
        out_dim = embedding_dimension * num_layers if jk == 'cat' else embedding_dimension
            
        self.linear_output1 = nn.Linear(out_dim, embedding_dimension)
        self.relu = nn.LeakyReLU()
        self.linear_output2 = nn.Linear(embedding_dimension, 2, bias=False)

    def forward(self, edge_index, edge_type, edge_attr, des=None, tweet=None, 
                num_prop=None, cat_prop=None, feat=None, current_epoch=0, warmup=20):
        if self.use_mgtab:
            x = self.linear_relu_feat(feat)
        else:
            d = self.linear_relu_des(des)
            t = self.linear_relu_tweet(tweet)
            n = self.linear_relu_num_prop(num_prop)
            c = self.linear_relu_cat_prop(cat_prop)
            x = torch.cat([d, t, n, c], dim=1)
        
        x = self.linear_relu_input(x)
        
        x, raw_logits = self.gnn(x, edge_index, edge_type, edge_attr, self.tau, current_epoch, warmup)
        
        out = x
        out = self.linear_output1(out)
        out = self.relu(out)
        out = self.linear_output2(out)
        
        return out, raw_logits