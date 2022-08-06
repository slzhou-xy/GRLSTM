from torch_geometric.nn import GATConv
from torch import nn


class GATLayer(nn.Module):
    def __init__(self, dimension):
        super(GATLayer, self).__init__()
        self.gat = GATConv(in_channels=dimension, out_channels=16,
                           heads=8, dropout=0.1, concat=True)

    def forward(self, x, edge_index):
        return self.gat(x, edge_index)
