import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, TransformerConv
from torch_geometric.utils import softmax

class SimpleHGNConv(MessagePassing):
    """
    Custom SimpleHGN Layer with Residual Attention (beta).
    """
    def __init__(self, in_channels, out_channels, num_edge_type, rel_dim, beta=None, final_layer=False):
        super(SimpleHGNConv, self).__init__(aggr="add", node_dim=0)
        self.W = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.W_r = torch.nn.Linear(rel_dim, out_channels, bias=False)
        self.a = torch.nn.Linear(3 * out_channels, 1, bias=False)
        self.W_res = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.rel_emb = torch.nn.Embedding(num_edge_type, rel_dim)
        self.beta = beta
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.ELU = torch.nn.ELU()
        self.final = final_layer
        
        self.init_weight()
        
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                    
    def forward(self, x, edge_index, edge_type, pre_alpha=None):
        # Propagate messages
        node_emb = self.propagate(x=x, edge_index=edge_index, edge_type=edge_type, pre_alpha=pre_alpha)
        
        # Internal Residual and Activation
        output = node_emb + self.W_res(x)
        output = self.ELU(output)
        
        if self.final:
            output = F.normalize(output, dim=1)
            
        return output, self.alpha.detach()
      
    def message(self, x_i, x_j, edge_type, pre_alpha, index, ptr, size_i):
        out = self.W(x_j)
        rel_emb = self.rel_emb(edge_type)
        
        # Calculate attention score
        alpha = self.leaky_relu(self.a(torch.cat((self.W(x_i), self.W(x_j), self.W_r(rel_emb)), dim=1)))
        alpha = softmax(alpha, index, ptr, size_i)
        
        # Apply residual attention smoothing (beta)
        if pre_alpha is not None and self.beta is not None:
            self.alpha = alpha * (1 - self.beta) + pre_alpha * self.beta
        else:
            self.alpha = alpha
            
        out = out * self.alpha.view(-1, 1)
        return out


def masked_edge_index(edge_index, edge_mask):
    return edge_index[:, edge_mask]

class SemanticAttention(torch.nn.Module):
    def __init__(self, in_channel, num_head, hidden_size=128):
        super(SemanticAttention, self).__init__()
        
        self.num_head = num_head
        self.att_layers = torch.nn.ModuleList()
        # multi-head attention
        for i in range(num_head):
            self.att_layers.append(
            torch.nn.Sequential(
                torch.nn.Linear(in_channel, hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_size, 1, bias=False))
            )
       
    def forward(self, z):
        w = self.att_layers[0](z).mean(0)                    
        beta = torch.softmax(w, dim=0)                
    
        beta = beta.expand((z.shape[0],) + beta.shape)
        output = (beta * z).sum(1)

        for i in range(1, self.num_head):
            w = self.att_layers[i](z).mean(0)
            beta = torch.softmax(w, dim=0)
            
            beta = beta.expand((z.shape[0],) + beta.shape)
            temp = (beta * z).sum(1)
            output += temp 
            
        return output / self.num_head

class SRGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=False, dropout=0.0):
        super(SRGATConv, self).__init__(aggr="add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att)
        
    def forward(self, x, edge_index):
        # x: [N, in_channels]
        
        # Linear transformation
        x_transformed = self.W(x).view(-1, self.heads, self.out_channels) # [N, heads, out_channels]
        
        # Start message passing
        out = self.propagate(edge_index, x=x_transformed)
        
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            
        return out

    def message(self, x_i, x_j, index, ptr, size_i):
        # x_i, x_j: [E, heads, out_channels]
        
        # Calculate attention
        # att: [1, heads, 2 * out_channels]
        # x_i, x_j: [E, heads, out_channels]
        cat_features = torch.cat([x_i, x_j], dim=-1) # [E, heads, 2 * out_channels]
        
        alpha = (cat_features * self.att).sum(dim=-1) # [E, heads]
        alpha = self.leaky_relu(alpha)
        
        alpha = softmax(alpha, index, ptr, size_i) # [E, heads]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        out = x_j * alpha.view(-1, self.heads, 1) # [E, heads, out_channels]
        return out

class SRGATLayer(torch.nn.Module):
    def __init__(self, num_edge_type, in_channel, out_channel, trans_heads, semantic_head, dropout):
        super(SRGATLayer, self).__init__()
        
        # Social Environment Feature weights (Eq 5)
        self.W_env = torch.nn.ModuleList([
            torch.nn.Linear(in_channel, out_channel) for _ in range(int(num_edge_type))
        ])
        
        # Fusion gate (Eq 6): z_r = σ(W_z · (e ⊙ env) + b_z)
        self.mlp_fuse = torch.nn.Sequential(
            torch.nn.Linear(out_channel, out_channel),
            torch.nn.Sigmoid()
        )

        self.activation = torch.nn.ELU()
        self.transformer_list = torch.nn.ModuleList()
        for i in range(int(num_edge_type)):
            self.transformer_list.append(SRGATConv(in_channels=in_channel, out_channels=out_channel, heads=trans_heads, concat=False, dropout=dropout))
        
        self.num_edge_type = num_edge_type
        self.semantic_attention = SemanticAttention(in_channel=out_channel, num_head=semantic_head)

    def forward(self, features, edge_index, edge_type):
        edge_index_list = []
        for i in range(self.num_edge_type):
            tmp = masked_edge_index(edge_index, edge_type == i)
            edge_index_list.append(tmp)

        semantic_embeddings_list = []
        for i in range(self.num_edge_type):
            # RGAT output e_i^{r(n)} with activation
            u = self.activation(self.transformer_list[i](features, edge_index_list[i])) # [N, out_channel]
            
            # Social Environment Feature x_{R, i'}^{(n)} (Eq 5)
            # Aggregate only over nodes participating in the relation-specific subgraph
            if edge_index_list[i].size(1) > 0:
                unique_nodes = torch.unique(edge_index_list[i])
                env_feat_raw = features[unique_nodes].mean(dim=0, keepdim=True) # [1, in_channel]
            else:
                env_feat_raw = features.mean(dim=0, keepdim=True) # fallback: [1, in_channel]
            env_feat = torch.sigmoid(self.W_env[i](env_feat_raw)) # [1, out_channel]
            env_feat = env_feat.expand(features.size(0), -1) # [N, out_channel]
            
            # Gating mechanism z_r^{(n)} (Eq 6): Hadamard product
            z_r = self.mlp_fuse(u * env_feat) # [N, out_channel]
            
            # Fusion h_i^{r(n)} (Eq 7)
            output = torch.mul(torch.tanh(u), z_r) + torch.mul(env_feat, (1 - z_r)) # [N, out_channel]
            
            semantic_embeddings_list.append(output.unsqueeze(1))
            
        semantic_embeddings = torch.cat(semantic_embeddings_list, dim=1)
        return self.semantic_attention(semantic_embeddings)



class RGTLayer(torch.nn.Module):
    def __init__(self, num_edge_type, in_channel, out_channel, trans_heads, semantic_head, dropout):
        super(RGTLayer, self).__init__()
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(in_channel + out_channel, in_channel),
            torch.nn.Sigmoid()
        )

        self.activation = torch.nn.ELU()
        self.transformer_list = torch.nn.ModuleList()
        for i in range(int(num_edge_type)):
            # TransformerConv inherently outputs shape [N, out_channels * heads] if concat=True
            # Here concat=False is used, so output is [N, out_channels] averaged over heads
            self.transformer_list.append(TransformerConv(in_channels=in_channel, out_channels=out_channel, heads=trans_heads, dropout=dropout, concat=False))
        
        self.num_edge_type = num_edge_type
        self.semantic_attention = SemanticAttention(in_channel=out_channel, num_head=semantic_head)

    def forward(self, features, edge_index, edge_type):
        edge_index_list = []
        for i in range(self.num_edge_type):
            tmp = masked_edge_index(edge_index, edge_type == i)
            edge_index_list.append(tmp)

        # Process first relation
        u = self.transformer_list[0](features, edge_index_list[0]).flatten(1) 
        a = self.gate(torch.cat((u, features), dim=1))
        semantic_embeddings = (torch.mul(torch.tanh(u), a) + torch.mul(features, (1-a))).unsqueeze(1)
        
        # Process remaining relations
        for i in range(1, len(edge_index_list)):
            u = self.transformer_list[i](features, edge_index_list[i]).flatten(1)
            a = self.gate(torch.cat((u, features), dim=1))
            output = torch.mul(torch.tanh(u), a) + torch.mul(features, (1-a))
            semantic_embeddings = torch.cat((semantic_embeddings, output.unsqueeze(1)), dim=1)
            
        # FIXED INDENTATION: Return after the loop finishes aggregating all relations
        return self.semantic_attention(semantic_embeddings)