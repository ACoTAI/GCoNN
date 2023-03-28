import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from layer import GraphConvolution, GraphAttentionLayer
from dgl.nn import SAGEConv
import dgl

class GNNq(nn.Module):
    def __init__(self, opt, adj):
        super(GNNq, self).__init__()
        self.opt = opt
        self.adj = adj

        opt_ = dict([('in', opt['num_feature']), ('out', opt['hidden_dim'])])
        self.m1 = GraphConvolution(opt_, adj)

        opt_ = dict([('in', opt['hidden_dim']), ('out', opt['num_class'])])
        self.m2 = GraphConvolution(opt_, adj)

        if opt['cuda']:
            self.cuda()

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)
        x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.m2(x)
        return x


class GSq(nn.Module):
    def __init__(self,
                 opt, 
                 adj,
                 aggregator_type='mean'):
        super(GSq, self).__init__()
        self.opt = opt
        self.layers = nn.ModuleList()
        edge = (adj.indices()[0], adj.indices()[1])
        weights = adj.values()
        self.g = dgl.graph(edge)
        self.g.edata['w'] = weights
        # input layer
        self.m1 = SAGEConv(opt['num_feature'], 
                           opt['hidden_dim'], 
                           aggregator_type=aggregator_type,
                           feat_drop=opt['input_dropout'], 
                           activation=F.relu)
        # output layer
        self.m2 = SAGEConv(opt['hidden_dim'], 
                                    opt['num_class'], 
                                    aggregator_type,
                                    feat_drop=opt['dropout'], 
                                    activation=None) # activation None
        if opt['cuda']:
            self.cuda()

    def forward(self, x):
        x = self.m1(self.g, x)
        x = self.m2(self.g, x)
        return x


class GATq(nn.Module):
    def __init__(self, opt, adj, alpha=0.1, nheads=4):
        """Dense version of GAT."""
        super(GATq, self).__init__()
        self.opt = opt
        self.adj = adj
        edge = (adj.indices()[0], adj.indices()[1])
        weights = adj.values()
        self.g = dgl.graph(edge)
        self.g.edata['w'] = weights
        dropout = self.opt['dropout']
        nfeat = opt['num_feature']
        nhid = opt['hidden_dim']
        nclass = opt['num_class']
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

        if opt['cuda']:
            self.cuda()

    def forward(self, x):
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = torch.cat([att(x, self.adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = F.elu(self.out_att(x, self.adj))
        return F.log_softmax(x, dim=1)

class GNNpa(nn.Module):
    def __init__(self, opt, adj):
        super(GNNpa, self).__init__()
        self.opt = opt
        self.adj = adj

        opt_ = dict([('in', opt['num_feature']), ('out', opt['hidden_dim'])])
        self.m1 = GraphConvolution(opt_, adj)

        opt_ = dict([('in', opt['hidden_dim']), ('out', opt['num_class'])])
        self.m2 = GraphConvolution(opt_, adj)

        if opt['cuda']:
            self.cuda()

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)
        x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.m2(x)
        return x


class GATpa(nn.Module):
    def __init__(self, opt, adj, alpha=0.1, nheads=4):
        """Dense version of GAT."""
        super(GATpa, self).__init__()
        self.opt = opt
        self.adj = adj
        edge = (adj.indices()[0], adj.indices()[1])
        weights = adj.values()
        self.g = dgl.graph(edge)
        self.g.edata['w'] = weights
        dropout = self.opt['dropout']
        nfeat = opt['num_feature']
        nhid = opt['hidden_dim']
        nclass = opt['num_class']
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)
        if opt['cuda']:
            self.cuda()

    def forward(self, x):
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = torch.cat([att(x, self.adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = F.elu(self.out_att(x, self.adj))
        return F.log_softmax(x, dim=1)
        return x

class GSpa(nn.Module):
    def __init__(self,
                 opt, 
                 adj,
                 aggregator_type='mean'):
        super(GSpa, self).__init__()
        self.opt = opt
        self.layers = nn.ModuleList()
        edge = (adj.indices()[0], adj.indices()[1])
        weights = adj.values()
        self.g = dgl.graph(edge)
        self.g.edata['w'] = weights
        # input layer
        self.m1 = SAGEConv(opt['num_feature'], 
                           opt['hidden_dim'], 
                           aggregator_type=aggregator_type,
                           feat_drop=opt['input_dropout'], 
                           activation=F.relu)
        # output layer
        self.m2 = SAGEConv(opt['hidden_dim'], 
                                    opt['num_class'], 
                                    aggregator_type,
                                    feat_drop=opt['dropout'], 
                                    activation=None) # activation None
        if opt['cuda']:
            self.cuda()

    def forward(self, x):
        x = self.m1(self.g, x)
        x = self.m2(self.g, x)
        return x