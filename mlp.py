import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

# Baseline MLP on global graph features 
def global_graph_features(batch, num_relations=10):
    x = batch.x
    b = batch.batch
    B = int(batch.num_graphs)

    n_nodes = torch.bincount(b, minlength=B).float().unsqueeze(1)
    node_sum = global_add_pool(x, b)
    node_mean = node_sum / n_nodes.clamp(min=1.0)

    src = batch.edge_index[0]
    edge_batch = b[src]
    et = batch.edge_type
    oh = F.one_hot(et, num_classes=num_relations).float()
    edge_hist = torch.zeros((B, num_relations), device=x.device)
    edge_hist.index_add_(0, edge_batch, oh)
    n_edges = torch.bincount(edge_batch, minlength=B).float().unsqueeze(1)

    return torch.cat([n_nodes, n_edges, node_sum, node_mean, edge_hist], dim=1)

class GlobalFeatMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128, depth=3, dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, feats):
        return self.net(feats)