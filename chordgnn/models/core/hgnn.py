import torch.nn as nn
from torch.nn import functional as F
import torch
from .gnn import SageConvScatter as SageConv, ResGatedGraphConv, JumpingKnowledge


class HeteroAttention(nn.Module):
    def __init__(self, n_hidden, n_layers):
        super(HeteroAttention, self).__init__()
        self.lstm = nn.LSTM(n_hidden, (n_layers*n_hidden)//2, bidirectional=True, batch_first=True)
        self.att = nn.Linear(2 * ((n_layers*n_hidden)//2), 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        nn.init.xavier_uniform_(self.att.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, x):
        alpha, _ = self.lstm(x)
        alpha = self.att(alpha).squeeze(-1)  # [num_nodes, num_layers]
        alpha = torch.softmax(alpha, dim=-1)
        return (x * alpha.unsqueeze(-1)).sum(dim=0)


class HeteroResGatedGraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, etypes, bias=True, reduction='mean'):
        super(HeteroResGatedGraphConvLayer, self).__init__()
        self.out_features = out_features
        self.etypes = etypes
        if reduction == 'mean':
            self.reduction = lambda x: x.mean(dim=0)
        elif reduction == 'sum':
            self.reduction = lambda x: x.sum(dim=0)
        elif reduction == 'max':
            self.reduction = lambda x: x.max(dim=0)
        elif reduction == 'min':
            self.reduction = lambda x: x.min(dim=0)
        elif reduction == 'concat':
            self.reduction = lambda x: torch.cat(x, dim=0)
        elif reduction == 'lstm':
            self.reduction = HeteroAttention(out_features, len(etypes.keys()))
        elif reduction == 'none':
            self.reduction = lambda x: x
        else:
            raise NotImplementedError

        conv_dict = dict()
        for etype in etypes.keys():
            conv_dict[etype] = ResGatedGraphConv(in_features, out_features, bias=bias)
        self.conv = nn.ModuleDict(conv_dict)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.conv.values():
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_type):
        out = torch.zeros((len(self.conv.keys()), x.shape[0], self.out_features))
        for idx, (ekey, evalue) in enumerate(self.etypes.items()):
            mask = edge_type == evalue
            out[idx] = self.conv[ekey](x, edge_index[:, mask])
        return self.reduction(out).to(x.device)


class HeteroSageConvLayer(nn.Module):
    def __init__(self, in_features, out_features, etypes, bias=True, reduction='mean'):
        super(HeteroSageConvLayer, self).__init__()
        self.out_features = out_features
        self.etypes = etypes
        if reduction == 'mean':
            self.reduction = lambda x: x.mean(dim=0)
        elif reduction == 'sum':
            self.reduction = lambda x: x.sum(dim=0)
        elif reduction == 'max':
            self.reduction = lambda x: x.max(dim=0)
        elif reduction == 'min':
            self.reduction = lambda x: x.min(dim=0)
        elif reduction == 'concat':
            self.reduction = lambda x: torch.cat(x, dim=0)
        elif reduction == 'lstm':
            self.reduction = HeteroAttention(out_features, len(etypes.keys()))
        else:
            raise NotImplementedError

        conv_dict = dict()
        for etype in etypes.keys():
            conv_dict[etype] = SageConv(in_features, out_features, bias=bias)
        self.conv = nn.ModuleDict(conv_dict)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.conv.values():
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_type):
        out = torch.zeros((len(self.conv.keys()), x.shape[0], self.out_features))
        for idx, (ekey, evalue) in enumerate(self.etypes.items()):
            mask = edge_type == evalue
            out[idx] = self.conv[ekey](x, edge_index[:, mask])
        return self.reduction(out).to(x.device)



class HGCN(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers, etypes={"onset":0, "consecutive":1, "during":2, "rests":3, "consecutive_rev":4, "during_rev":5, "rests_rev":6}, activation=F.relu, dropout=0.5, jk=False):
        super(HGCN, self).__init__()
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        self.normalize = F.normalize
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers.append(HeteroSageConvLayer(in_feats, n_hidden, etypes=etypes))
        for i in range(n_layers - 1):
            self.layers.append(HeteroSageConvLayer(n_hidden, n_hidden, etypes=etypes))
        if jk:
            self.use_knowledge = True
            self.jk = JumpingKnowledge(n_hidden=n_hidden, n_layers=n_layers)
        else:
            self.use_knowledge = False
        self.layers.append(HeteroSageConvLayer(n_hidden, out_feats, etypes=etypes))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.layers:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_type):
        h = x
        hs = []
        for conv in self.layers[:-1]:
            h = conv(h, edge_index, edge_type)
            h = self.activation(h)
            h = self.normalize(h)
            h = self.dropout(h)
            hs.append(h)
        if self.use_knowledge:
            h = self.jk(hs)
        h = self.layers[-1](h, edge_index, edge_type)
        return h


class HResGatedConv(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers, etypes={"onset":0, "consecutive":1, "during":2, "rests":3, "consecutive_rev":4, "during_rev":5, "rests_rev":6}, activation=F.relu, dropout=0.5, jk=False):
        super(HResGatedConv, self).__init__()
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        self.normalize = F.normalize
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers.append(HeteroResGatedGraphConvLayer(in_feats, n_hidden, etypes=etypes))
        for i in range(n_layers - 1):
            self.layers.append(HeteroResGatedGraphConvLayer(n_hidden, n_hidden, etypes=etypes))
        if jk:
            self.use_knowledge = True
            self.jk = JumpingKnowledge(n_hidden=n_hidden, n_layers=n_layers)
        else:
            self.use_knowledge = False
        self.layers.append(HeteroResGatedGraphConvLayer(n_hidden, out_feats, etypes=etypes))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.layers:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_type):
        h = x
        hs = []
        for conv in self.layers:
            h = conv(h, edge_index, edge_type)
            h = self.activation(h)
            h = self.normalize(h)
            h = self.dropout(h)
            hs.append(h)
        if self.use_knowledge:
            h = self.jk(hs)
        # h = self.layers[-1](h, edge_index, edge_type)
        return h


class HGPSLayer(nn.Module):
    def __init__(
            self, in_features, out_features, num_heads,
            etypes={"onset":0, "consecutive":1, "during":2, "rests":3, "consecutive_rev":4, "during_rev":5, "rests_rev":6},
            activation=F.relu, dropout=0.2, bias=True):
        """
        General Powerful Scalable Graph Transformers Convolutional Layer

        Parameters
        ----------
        in_features: int
            Number of input features
        out_features: int
            Number of output features
        num_heads: int
            Number of attention heads
        etypes: dict
            Edge types
        activation: nn.Module
            Activation function
        dropout: float
            Dropout rate
        bias: bool
            Whether to use bias
        """
        super(HGPSLayer, self).__init__()
        self.embedding = nn.Linear(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.activation = activation
        self.normalize_local = nn.LayerNorm(out_features)
        self.normalize_attn = nn.LayerNorm(out_features)
        self.dropout_ff = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_local = nn.Dropout(dropout)
        self.ff1 = nn.Linear(out_features, out_features*2, bias=bias)
        self.ff2 = nn.Linear(out_features*2, out_features, bias=bias)
        self.attn = nn.MultiheadAttention(out_features, num_heads, dropout=dropout, bias=bias, batch_first=True)
        self.local = HeteroResGatedGraphConvLayer(out_features, out_features, bias=bias, etypes=etypes)

    def forward(self, x, edge_index, edge_type):

        h_init = self.embedding(x)
        # Local embeddings
        local_out = self.local(h_init, edge_index, edge_type)
        local_out = self.activation(local_out)
        local_out = self.normalize_local(local_out)
        local_out = self.dropout_local(local_out)
        local_out = local_out + h_init

        # Global embeddings
        h = h_init.unsqueeze(0)
        attn_out, _ = self.attn(h, h, h)
        attn_out = self.activation(attn_out)
        attn_out = self.normalize_attn(attn_out)
        attn_out = self.dropout_attn(attn_out)
        attn_out = attn_out + h_init

        # Combine
        out = local_out + attn_out.squeeze()
        h = self.ff1(out)
        h = self.activation(h)
        h = self.dropout_ff(h)
        h = self.ff2(h)
        h - self.dropout_ff(h)
        out = F.normalize(out + h)
        return out


class HGPS(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers, etypes={"onset":0, "consecutive":1, "during":2, "rests":3, "consecutive_rev":4, "during_rev":5, "rests_rev":6}, activation=F.relu, dropout=0.5, jk=False):
        super(HGPS, self).__init__()
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        self.normalize = F.normalize
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers.append(HGPSLayer(in_feats, n_hidden, etypes=etypes, num_heads=4))
        for i in range(n_layers - 1):
            self.layers.append(HGPSLayer(n_hidden, n_hidden, etypes=etypes, num_heads=4))
        if jk:
            self.use_knowledge = True
            self.jk = JumpingKnowledge(n_hidden=n_hidden, n_layers=n_layers)
        else:
            self.use_knowledge = False
        self.layers.append(HGPSLayer(n_hidden, out_feats, etypes=etypes, num_heads=4))

    def forward(self, x, edge_index, edge_type):
        h = x
        hs = []
        for conv in self.layers:
            h = conv(h, edge_index, edge_type)
            h = self.activation(h)
            h = self.normalize(h)
            h = self.dropout(h)
            hs.append(h)
        if self.use_knowledge:
            h = self.jk(hs)
        # h = self.layers[-1](h, edge_index, edge_type)
        return h
