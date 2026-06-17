import torch
from torch import nn
from torch_geometric.nn import RGCNConv, GCNConv, GATConv, SAGEConv, FAConv, HGTConv, JumpingKnowledge
import torch.nn.functional as F
from models.layer import SimpleHGNConv, RGTLayer, SRGATLayer

def activation_resolver(act_name: str):
    """Resolves a string to a PyTorch activation module."""
    if act_name is None or act_name.lower() == 'none':
        return nn.Identity()
    
    act_name = act_name.lower()
    if act_name == 'relu':
        return nn.ReLU()
    elif act_name == 'elu':
        return nn.ELU()
    elif act_name == 'gelu':
        return nn.GELU()
    elif act_name == 'leaky_relu':
        return nn.LeakyReLU(0.2)
    elif act_name == 'silu' or act_name == 'swish':
        return nn.SiLU()
    elif act_name == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation function: {act_name}")


class BotRGCN(nn.Module):
    def __init__(self, embedding_dimension=128, dropout=0.3, heads=4, num_rel=2, num_layers=2, 
                 des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3, feat_size=788, 
                 use_mgtab=False, use_norm=True, use_residual=True, activation='relu', jk='cat'):
        super(BotRGCN, self).__init__()
        self.dropout = dropout
        self.use_mgtab = use_mgtab
        self.use_norm = use_norm
        self.use_residual = use_residual
        self.num_layers = num_layers
        self.jk_mode = jk
        self.act = activation_resolver(activation)
        
        if self.use_mgtab:
            self.linear_relu_feat = nn.Sequential(nn.Linear(feat_size, embedding_dimension), nn.LeakyReLU())
        else:
            dim_quarter = embedding_dimension // 4
            self.linear_relu_des = nn.Sequential(nn.Linear(des_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_tweet = nn.Sequential(nn.Linear(tweet_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_num_prop = nn.Sequential(nn.Linear(num_prop_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_cat_prop = nn.Sequential(nn.Linear(cat_prop_size, dim_quarter), nn.LeakyReLU())
        
        self.linear_relu_input = nn.Sequential(nn.Linear(embedding_dimension, embedding_dimension), nn.LeakyReLU())
        
        self.convs = nn.ModuleList()
        if self.use_norm:
            self.norms = nn.ModuleList()
            
        for _ in range(num_layers):
            self.convs.append(RGCNConv(embedding_dimension, embedding_dimension, num_relations=num_rel))
            if self.use_norm:
                self.norms.append(nn.LayerNorm(embedding_dimension))
                
        if self.jk_mode is not None:
            self.jk = JumpingKnowledge(self.jk_mode, channels=embedding_dimension, num_layers=num_layers)
            
        out_dim = embedding_dimension * num_layers if self.jk_mode == 'cat' else embedding_dimension
        
        self.linear_relu_output1 = nn.Sequential(nn.Linear(out_dim, embedding_dimension), nn.LeakyReLU())
        self.linear_output2 = nn.Linear(embedding_dimension, 2)
        
    def forward(self, edge_index, edge_type=None, des=None, tweet=None, num_prop=None, cat_prop=None, feat=None):
        if self.use_mgtab:
            x = self.linear_relu_feat(feat)
        else:
            d = self.linear_relu_des(des)
            t = self.linear_relu_tweet(tweet)
            n = self.linear_relu_num_prop(num_prop)
            c = self.linear_relu_cat_prop(cat_prop)
            x = torch.cat((d, t, n, c), dim=1)
        
        x = self.linear_relu_input(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        xs = []
        for i in range(self.num_layers):
            x_in = self.norms[i](x) if self.use_norm else x
            h = self.convs[i](x_in, edge_index, edge_type)
            h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h if self.use_residual else h
            xs.append(x)
            
        if self.jk_mode is not None:
            x = self.jk(xs)
        else:
            x = xs[-1]
        
        x = self.linear_relu_output1(x)
        out = self.linear_output2(x)
        
        return out
            
    
class GCN(nn.Module):
    def __init__(self, embedding_dimension=128, dropout=0.3, heads=4, num_rel=2, num_layers=2, 
                 des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3, feat_size=788, 
                 use_mgtab=False, use_norm=True, use_residual=True, activation='relu', jk='cat'):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.use_mgtab = use_mgtab
        self.use_norm = use_norm
        self.use_residual = use_residual
        self.num_layers = num_layers
        self.jk_mode = jk
        self.act = activation_resolver(activation)
        
        if self.use_mgtab:
            self.linear_relu_feat = nn.Sequential(nn.Linear(feat_size, embedding_dimension), nn.LeakyReLU())
        else:
            dim_quarter = embedding_dimension // 4
            self.linear_relu_des = nn.Sequential(nn.Linear(des_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_tweet = nn.Sequential(nn.Linear(tweet_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_num_prop = nn.Sequential(nn.Linear(num_prop_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_cat_prop = nn.Sequential(nn.Linear(cat_prop_size, dim_quarter), nn.LeakyReLU())
        
        self.linear_relu_input = nn.Sequential(nn.Linear(embedding_dimension, embedding_dimension), nn.LeakyReLU())
        
        self.convs = nn.ModuleList()
        if self.use_norm:
            self.norms = nn.ModuleList()
            
        for _ in range(num_layers):
            self.convs.append(GCNConv(embedding_dimension, embedding_dimension))
            if self.use_norm:
                self.norms.append(nn.LayerNorm(embedding_dimension))
                
        if self.jk_mode is not None:
            self.jk = JumpingKnowledge(self.jk_mode, channels=embedding_dimension, num_layers=num_layers)
            
        out_dim = embedding_dimension * num_layers if self.jk_mode == 'cat' else embedding_dimension
        
        self.linear_relu_output1 = nn.Sequential(nn.Linear(out_dim, embedding_dimension), nn.LeakyReLU())
        self.linear_output2 = nn.Linear(embedding_dimension, 2)
        
    def forward(self, edge_index, edge_type=None, des=None, tweet=None, num_prop=None, cat_prop=None, feat=None):
        if self.use_mgtab:
            x = self.linear_relu_feat(feat)
        else:
            d = self.linear_relu_des(des)
            t = self.linear_relu_tweet(tweet)
            n = self.linear_relu_num_prop(num_prop)
            c = self.linear_relu_cat_prop(cat_prop)
            x = torch.cat((d, t, n, c), dim=1)
        
        x = self.linear_relu_input(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        xs = []
        for i in range(self.num_layers):
            x_in = self.norms[i](x) if self.use_norm else x
            h = self.convs[i](x_in, edge_index)
            h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h if self.use_residual else h
            xs.append(x)
            
        if self.jk_mode is not None:
            x = self.jk(xs)
        else:
            x = xs[-1]
        
        x = self.linear_relu_output1(x)
        out = self.linear_output2(x)
        
        return out
    

class GAT(nn.Module):
    def __init__(self, embedding_dimension=128, dropout=0.3, heads=4, num_rel=2, num_layers=2, 
                 des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3, feat_size=788, 
                 use_mgtab=False, use_norm=True, use_residual=True, activation='elu', jk='cat'):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.use_mgtab = use_mgtab
        self.use_norm = use_norm
        self.use_residual = use_residual
        self.num_layers = num_layers
        self.jk_mode = jk
        self.act = activation_resolver(activation)
        
        if self.use_mgtab:
            self.linear_relu_feat = nn.Sequential(nn.Linear(feat_size, embedding_dimension), nn.LeakyReLU())
        else:
            dim_quarter = embedding_dimension // 4
            self.linear_relu_des = nn.Sequential(nn.Linear(des_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_tweet = nn.Sequential(nn.Linear(tweet_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_num_prop = nn.Sequential(nn.Linear(num_prop_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_cat_prop = nn.Sequential(nn.Linear(cat_prop_size, dim_quarter), nn.LeakyReLU())
        
        self.linear_relu_input = nn.Sequential(nn.Linear(embedding_dimension, embedding_dimension), nn.LeakyReLU())
        
        self.convs = nn.ModuleList()
        if self.use_norm:
            self.norms = nn.ModuleList()
            
        for _ in range(num_layers):
            self.convs.append(GATConv(embedding_dimension, embedding_dimension // heads, heads=heads))
            if self.use_norm:
                self.norms.append(nn.LayerNorm(embedding_dimension))
                
        if self.jk_mode is not None:
            self.jk = JumpingKnowledge(self.jk_mode, channels=embedding_dimension, num_layers=num_layers)
            
        out_dim = embedding_dimension * num_layers if self.jk_mode == 'cat' else embedding_dimension
        
        self.linear_relu_output1 = nn.Sequential(nn.Linear(out_dim, embedding_dimension), nn.LeakyReLU())
        self.linear_output2 = nn.Linear(embedding_dimension, 2)
        
    def forward(self, edge_index, edge_type=None, des=None, tweet=None, num_prop=None, cat_prop=None, feat=None):
        if self.use_mgtab:
            x = self.linear_relu_feat(feat)
        else:
            d = self.linear_relu_des(des)
            t = self.linear_relu_tweet(tweet)
            n = self.linear_relu_num_prop(num_prop)
            c = self.linear_relu_cat_prop(cat_prop)
            x = torch.cat((d, t, n, c), dim=1)
        
        x = self.linear_relu_input(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        xs = []
        for i in range(self.num_layers):
            x_in = self.norms[i](x) if self.use_norm else x
            h = self.convs[i](x_in, edge_index)
            h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h if self.use_residual else h
            xs.append(x)
            
        if self.jk_mode is not None:
            x = self.jk(xs)
        else:
            x = xs[-1]
        
        x = self.linear_relu_output1(x)
        out = self.linear_output2(x)
            
        return out


class GraphSAGE(nn.Module):
    def __init__(self, embedding_dimension=128, dropout=0.3, heads=4, num_rel=2, num_layers=2, 
                 des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3, feat_size=788, 
                 use_mgtab=False, use_norm=True, use_residual=True, activation='relu', jk='cat'):
        super(GraphSAGE, self).__init__()
        self.dropout = dropout
        self.use_mgtab = use_mgtab
        self.use_norm = use_norm
        self.use_residual = use_residual
        self.num_layers = num_layers
        self.jk_mode = jk
        self.act = activation_resolver(activation)
        
        if self.use_mgtab:
            self.linear_relu_feat = nn.Sequential(nn.Linear(feat_size, embedding_dimension), nn.LeakyReLU())
        else:
            dim_quarter = embedding_dimension // 4
            self.linear_relu_des = nn.Sequential(nn.Linear(des_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_tweet = nn.Sequential(nn.Linear(tweet_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_num_prop = nn.Sequential(nn.Linear(num_prop_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_cat_prop = nn.Sequential(nn.Linear(cat_prop_size, dim_quarter), nn.LeakyReLU())
        
        self.linear_relu_input = nn.Sequential(nn.Linear(embedding_dimension, embedding_dimension), nn.LeakyReLU())
        
        self.convs = nn.ModuleList()
        if self.use_norm:
            self.norms = nn.ModuleList()
            
        for _ in range(num_layers):
            self.convs.append(SAGEConv(embedding_dimension, embedding_dimension))
            if self.use_norm:
                self.norms.append(nn.LayerNorm(embedding_dimension))
                
        if self.jk_mode is not None:
            self.jk = JumpingKnowledge(self.jk_mode, channels=embedding_dimension, num_layers=num_layers)
            
        out_dim = embedding_dimension * num_layers if self.jk_mode == 'cat' else embedding_dimension
        
        self.linear_relu_output1 = nn.Sequential(nn.Linear(out_dim, embedding_dimension), nn.LeakyReLU())
        self.linear_output2 = nn.Linear(embedding_dimension, 2)
        
    def forward(self, edge_index, edge_type=None, des=None, tweet=None, num_prop=None, cat_prop=None, feat=None):
        if self.use_mgtab:
            x = self.linear_relu_feat(feat)
        else:
            d = self.linear_relu_des(des)
            t = self.linear_relu_tweet(tweet)
            n = self.linear_relu_num_prop(num_prop)
            c = self.linear_relu_cat_prop(cat_prop)
            x = torch.cat((d, t, n, c), dim=1)
        
        x = self.linear_relu_input(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        xs = []
        for i in range(self.num_layers):
            x_in = self.norms[i](x) if self.use_norm else x
            h = self.convs[i](x_in, edge_index)
            h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h if self.use_residual else h
            xs.append(x)
            
        if self.jk_mode is not None:
            x = self.jk(xs)
        else:
            x = xs[-1]
        
        x = self.linear_relu_output1(x)
        out = self.linear_output2(x)
            
        return out
    

class FAGCN(nn.Module):
    def __init__(self, embedding_dimension=128, dropout=0.3, heads=4, num_rel=2, num_layers=2, 
                 des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3, feat_size=788, 
                 use_mgtab=False, use_norm=True, use_residual=True, activation='relu', jk='cat'):
        super(FAGCN, self).__init__()
        self.dropout = dropout
        self.use_mgtab = use_mgtab
        self.use_norm = use_norm
        self.use_residual = use_residual
        self.num_layers = num_layers
        self.jk_mode = jk
        self.act = activation_resolver(activation)
        
        if self.use_mgtab:
            self.linear_relu_feat = nn.Sequential(nn.Linear(feat_size, embedding_dimension), nn.LeakyReLU())
        else:
            dim_quarter = embedding_dimension // 4
            self.linear_relu_des = nn.Sequential(nn.Linear(des_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_tweet = nn.Sequential(nn.Linear(tweet_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_num_prop = nn.Sequential(nn.Linear(num_prop_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_cat_prop = nn.Sequential(nn.Linear(cat_prop_size, dim_quarter), nn.LeakyReLU())
        
        self.linear_relu_input = nn.Sequential(nn.Linear(embedding_dimension, embedding_dimension), nn.LeakyReLU())
        
        self.convs = nn.ModuleList()
        if self.use_norm:
            self.norms = nn.ModuleList()
            
        for _ in range(num_layers):
            self.convs.append(FAConv(embedding_dimension, eps=0.1, dropout=dropout))
            if self.use_norm:
                self.norms.append(nn.LayerNorm(embedding_dimension))
                
        if self.jk_mode is not None:
            self.jk = JumpingKnowledge(self.jk_mode, channels=embedding_dimension, num_layers=num_layers)
            
        out_dim = embedding_dimension * num_layers if self.jk_mode == 'cat' else embedding_dimension
        
        self.linear_relu_output1 = nn.Sequential(nn.Linear(out_dim, embedding_dimension), nn.LeakyReLU())
        self.linear_output2 = nn.Linear(embedding_dimension, 2)
        
    def forward(self, edge_index, edge_type=None, des=None, tweet=None, num_prop=None, cat_prop=None, feat=None):
        if self.use_mgtab:
            x = self.linear_relu_feat(feat)
        else:
            d = self.linear_relu_des(des)
            t = self.linear_relu_tweet(tweet)
            n = self.linear_relu_num_prop(num_prop)
            c = self.linear_relu_cat_prop(cat_prop)
            x = torch.cat((d, t, n, c), dim=1)
        
        x = self.linear_relu_input(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x_0 = x  
        xs = []
        for i in range(self.num_layers):
            x_in = self.norms[i](x) if self.use_norm else x
            h = self.convs[i](x_in, x_0, edge_index)
            h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h if self.use_residual else h
            xs.append(x)
            
        if self.jk_mode is not None:
            x = self.jk(xs)
        else:
            x = xs[-1]
        
        x = self.linear_relu_output1(x)
        out = self.linear_output2(x)
            
        return out


class HGT(nn.Module):
    def __init__(self, embedding_dimension=128, dropout=0.3, heads=4, num_rel=2, num_layers=2, 
                 des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3, feat_size=788, 
                 use_mgtab=False, use_norm=True, use_residual=True, activation='gelu', jk='cat'):
        super(HGT, self).__init__()
        self.dropout = dropout
        self.use_mgtab = use_mgtab
        self.num_rel = num_rel
        self.use_norm = use_norm
        self.use_residual = use_residual
        self.num_layers = num_layers
        self.jk_mode = jk
        self.act = activation_resolver(activation)
        
        if self.use_mgtab:
            self.linear_relu_feat = nn.Sequential(nn.Linear(feat_size, embedding_dimension), nn.LeakyReLU())
        else:
            dim_quarter = embedding_dimension // 4
            self.linear_relu_des = nn.Sequential(nn.Linear(des_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_tweet = nn.Sequential(nn.Linear(tweet_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_num_prop = nn.Sequential(nn.Linear(num_prop_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_cat_prop = nn.Sequential(nn.Linear(cat_prop_size, dim_quarter), nn.LeakyReLU())
        
        self.linear_relu_input = nn.Sequential(nn.Linear(embedding_dimension, embedding_dimension), nn.LeakyReLU())
        
        node_types = ['node']
        edge_types = [('node', str(i), 'node') for i in range(num_rel)]
        metadata = (node_types, edge_types)
        
        self.convs = nn.ModuleList()
        if self.use_norm:
            self.norms = nn.ModuleList()
            
        for _ in range(num_layers):
            self.convs.append(HGTConv(in_channels=embedding_dimension, out_channels=embedding_dimension, 
                                      metadata=metadata, heads=heads))
            if self.use_norm:
                self.norms.append(nn.LayerNorm(embedding_dimension))
                
        if self.jk_mode is not None:
            self.jk = JumpingKnowledge(self.jk_mode, channels=embedding_dimension, num_layers=num_layers)
            
        out_dim = embedding_dimension * num_layers if self.jk_mode == 'cat' else embedding_dimension
        
        self.linear_relu_output1 = nn.Sequential(nn.Linear(out_dim, embedding_dimension), nn.LeakyReLU())
        self.linear_output2 = nn.Linear(embedding_dimension, 2)
        
    def forward(self, edge_index, edge_type=None, des=None, tweet=None, num_prop=None, cat_prop=None, feat=None):
        if self.use_mgtab:
            x = self.linear_relu_feat(feat)
        else:
            d = self.linear_relu_des(des)
            t = self.linear_relu_tweet(tweet)
            n = self.linear_relu_num_prop(num_prop)
            c = self.linear_relu_cat_prop(cat_prop)
            x = torch.cat((d, t, n, c), dim=1)
        
        x = self.linear_relu_input(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
            
        edge_index_dict = {}
        for i in range(self.num_rel):
            mask = (edge_type == i)
            edge_index_dict[('node', str(i), 'node')] = edge_index[:, mask]
            
        xs = []
        for i in range(self.num_layers):
            x_in = self.norms[i](x) if self.use_norm else x
            x_dict = {'node': x_in}
            h_dict = self.convs[i](x_dict, edge_index_dict)
            h = self.act(h_dict['node'])
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h if self.use_residual else h
            xs.append(x)
            
        if self.jk_mode is not None:
            x = self.jk(xs)
        else:
            x = xs[-1]
        
        x = self.linear_relu_output1(x)
        out = self.linear_output2(x)
            
        return out


class SimpleHGN(nn.Module):
    def __init__(self, embedding_dimension=128, dropout=0.3, heads=4, num_rel=2, num_layers=2, 
                 des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3, feat_size=788, 
                 use_mgtab=False, use_norm=True, use_residual=True, beta=0.05, jk='cat'):
        super(SimpleHGN, self).__init__()
        self.dropout = dropout
        self.use_mgtab = use_mgtab
        self.use_norm = use_norm
        self.use_residual = use_residual
        self.num_layers = num_layers
        self.jk_mode = jk
        
        if self.use_mgtab:
            self.linear_relu_feat = nn.Sequential(nn.Linear(feat_size, embedding_dimension), nn.LeakyReLU())
        else:
            dim_quarter = embedding_dimension // 4
            self.linear_relu_des = nn.Sequential(nn.Linear(des_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_tweet = nn.Sequential(nn.Linear(tweet_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_num_prop = nn.Sequential(nn.Linear(num_prop_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_cat_prop = nn.Sequential(nn.Linear(cat_prop_size, dim_quarter), nn.LeakyReLU())
        
        self.linear_relu_input = nn.Sequential(nn.Linear(embedding_dimension, embedding_dimension), nn.LeakyReLU())
        
        self.convs = nn.ModuleList()
        if self.use_norm:
            self.norms = nn.ModuleList()
            
        for i in range(num_layers):
            final_layer = (i == num_layers - 1)
            self.convs.append(SimpleHGNConv(embedding_dimension, embedding_dimension, num_rel, 
                                            rel_dim=embedding_dimension, beta=beta, final_layer=final_layer))
            if self.use_norm:
                self.norms.append(nn.LayerNorm(embedding_dimension))
                
        if self.jk_mode is not None:
            self.jk = JumpingKnowledge(self.jk_mode, channels=embedding_dimension, num_layers=num_layers)
            
        out_dim = embedding_dimension * num_layers if self.jk_mode == 'cat' else embedding_dimension
        
        self.linear_relu_output1 = nn.Sequential(nn.Linear(out_dim, embedding_dimension), nn.LeakyReLU())
        self.linear_output2 = nn.Linear(embedding_dimension, 2)
        
    def forward(self, edge_index, edge_type=None, des=None, tweet=None, num_prop=None, cat_prop=None, feat=None):
        if self.use_mgtab:
            x = self.linear_relu_feat(feat)
        else:
            d = self.linear_relu_des(des)
            t = self.linear_relu_tweet(tweet)
            n = self.linear_relu_num_prop(num_prop)
            c = self.linear_relu_cat_prop(cat_prop)
            x = torch.cat((d, t, n, c), dim=1)
        
        x = self.linear_relu_input(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
            
        xs = []
        pre_alpha = None
        for i in range(self.num_layers):
            x_in = self.norms[i](x) if self.use_norm else x
            h, alpha = self.convs[i](x_in, edge_index, edge_type, pre_alpha=pre_alpha)
            pre_alpha = alpha
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h if self.use_residual else h
            xs.append(x)
            
        if self.jk_mode is not None:
            x = self.jk(xs)
        else:
            x = xs[-1]
        
        x = self.linear_relu_output1(x)
        out = self.linear_output2(x)
            
        return out
    

class RGT(nn.Module):
    def __init__(self, embedding_dimension=128, dropout=0.3, heads=4, num_rel=2, num_layers=2, 
                 des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3, feat_size=788, 
                 use_mgtab=False, use_norm=True, use_residual=True, jk='cat'):
        super(RGT, self).__init__()
        self.dropout = dropout
        self.use_mgtab = use_mgtab
        self.num_rel = num_rel
        self.use_norm = use_norm
        self.use_residual = use_residual
        self.num_layers = num_layers
        self.jk_mode = jk
        
        trans_heads = heads
        semantic_heads = heads
        
        if self.use_mgtab:
            self.linear_relu_feat = nn.Sequential(nn.Linear(feat_size, embedding_dimension), nn.LeakyReLU())
        else:
            dim_quarter = embedding_dimension // 4
            self.linear_relu_des = nn.Sequential(nn.Linear(des_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_tweet = nn.Sequential(nn.Linear(tweet_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_num_prop = nn.Sequential(nn.Linear(num_prop_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_cat_prop = nn.Sequential(nn.Linear(cat_prop_size, dim_quarter), nn.LeakyReLU())
        
        self.linear_relu_input = nn.Sequential(nn.Linear(embedding_dimension, embedding_dimension), nn.LeakyReLU())
        
        self.convs = nn.ModuleList()
        if self.use_norm:
            self.norms = nn.ModuleList()
            
        for _ in range(num_layers):
            self.convs.append(RGTLayer(num_edge_type=num_rel, in_channel=embedding_dimension, out_channel=embedding_dimension, 
                                       trans_heads=trans_heads, semantic_head=semantic_heads, dropout=dropout))
            if self.use_norm:
                self.norms.append(nn.LayerNorm(embedding_dimension))
                
        if self.jk_mode is not None:
            self.jk = JumpingKnowledge(self.jk_mode, channels=embedding_dimension, num_layers=num_layers)
            
        out_dim = embedding_dimension * num_layers if self.jk_mode == 'cat' else embedding_dimension
        
        self.linear_relu_output1 = nn.Sequential(nn.Linear(out_dim, embedding_dimension), nn.LeakyReLU())
        self.linear_output2 = nn.Linear(embedding_dimension, 2)
        
    def forward(self, edge_index, edge_type=None, des=None, tweet=None, num_prop=None, cat_prop=None, feat=None):
        if self.use_mgtab:
            x = self.linear_relu_feat(feat)
        else:
            d = self.linear_relu_des(des)
            t = self.linear_relu_tweet(tweet)
            n = self.linear_relu_num_prop(num_prop)
            c = self.linear_relu_cat_prop(cat_prop)
            x = torch.cat((d, t, n, c), dim=1)
        
        x = self.linear_relu_input(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
            
        xs = []
        for i in range(self.num_layers):
            x_in = self.norms[i](x) if self.use_norm else x
            h = self.convs[i](x_in, edge_index, edge_type)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h if self.use_residual else h
            xs.append(x)
            
        if self.jk_mode is not None:
            x = self.jk(xs)
        else:
            x = xs[-1]
        
        x = self.linear_relu_output1(x)
        out = self.linear_output2(x)
            
        return out

class SRGAT(nn.Module):
    def __init__(self, embedding_dimension=128, dropout=0.3, heads=4, num_rel=2, num_layers=2, 
                 des_size=768, tweet_size=768, num_prop_size=5, cat_prop_size=3, feat_size=788, 
                 use_mgtab=False, use_norm=True, use_residual=True, jk='cat'):
        super(SRGAT, self).__init__()
        self.dropout = dropout
        self.use_mgtab = use_mgtab
        self.num_rel = num_rel
        self.use_norm = use_norm
        self.use_residual = use_residual
        self.num_layers = num_layers
        self.jk_mode = jk
        
        trans_heads = heads
        semantic_heads = heads
        
        if self.use_mgtab:
            self.linear_relu_feat = nn.Sequential(nn.Linear(feat_size, embedding_dimension), nn.LeakyReLU())
        else:
            dim_quarter = embedding_dimension // 4
            self.linear_relu_des = nn.Sequential(nn.Linear(des_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_tweet = nn.Sequential(nn.Linear(tweet_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_num_prop = nn.Sequential(nn.Linear(num_prop_size, dim_quarter), nn.LeakyReLU())
            self.linear_relu_cat_prop = nn.Sequential(nn.Linear(cat_prop_size, dim_quarter), nn.LeakyReLU())
        
        self.linear_relu_input = nn.Sequential(nn.Linear(embedding_dimension, embedding_dimension), nn.LeakyReLU())
        
        self.convs = nn.ModuleList()
        if self.use_norm:
            self.norms = nn.ModuleList()
            
        for _ in range(num_layers):
            self.convs.append(SRGATLayer(num_edge_type=num_rel, in_channel=embedding_dimension, out_channel=embedding_dimension, 
                                       trans_heads=trans_heads, semantic_head=semantic_heads, dropout=dropout))
            if self.use_norm:
                self.norms.append(nn.LayerNorm(embedding_dimension))
                
        if self.jk_mode is not None:
            self.jk = JumpingKnowledge(self.jk_mode, channels=embedding_dimension, num_layers=num_layers)
            
        out_dim = embedding_dimension * num_layers if self.jk_mode == 'cat' else embedding_dimension
        
        self.linear_relu_output1 = nn.Sequential(nn.Linear(out_dim, embedding_dimension), nn.LeakyReLU())
        self.linear_output2 = nn.Linear(embedding_dimension, 2)
        
    def forward(self, edge_index, edge_type=None, des=None, tweet=None, num_prop=None, cat_prop=None, feat=None):
        if self.use_mgtab:
            x = self.linear_relu_feat(feat)
        else:
            d = self.linear_relu_des(des)
            t = self.linear_relu_tweet(tweet)
            n = self.linear_relu_num_prop(num_prop)
            c = self.linear_relu_cat_prop(cat_prop)
            x = torch.cat((d, t, n, c), dim=1)
        
        x = self.linear_relu_input(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if edge_type is None:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
            
        xs = []
        for i in range(self.num_layers):
            x_in = self.norms[i](x) if self.use_norm else x
            h = self.convs[i](x_in, edge_index, edge_type)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h if self.use_residual else h
            xs.append(x)
            
        if self.jk_mode is not None:
            x = self.jk(xs)
        else:
            x = xs[-1]
        
        x = self.linear_relu_output1(x)
        out = self.linear_output2(x)
            
        return out