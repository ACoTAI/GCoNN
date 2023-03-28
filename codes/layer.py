import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import random

class GraphConvolution(nn.Module):

    def __init__(self, opt, adj):
        super(GraphConvolution, self).__init__()
        self.opt = opt
        self.in_size = opt['in']
        self.out_size = opt['out']
        self.adj = adj
        self.weight = Parameter(torch.Tensor(self.in_size, self.out_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_size)
        self.weight.data.uniform_(-stdv, stdv)
    
    def forward(self, x):
        m = torch.mm(x, self.weight)
        m = torch.spmm(self.adj, m)
        return m


class GraphAttentionLayer(nn.Module):
	"""
	Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
	"""
	def __init__(self, in_features, out_features, dropout, alpha, concat=True):
		super(GraphAttentionLayer, self).__init__()
		self.dropout = dropout
		self.in_features = in_features
		self.out_features = out_features
		self.alpha = alpha
		self.concat = concat

		self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
		nn.init.xavier_uniform_(self.W.data, gain=1.414)
		self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
		nn.init.xavier_uniform_(self.a.data, gain=1.414)

		self.leakyrelu = nn.LeakyReLU(self.alpha)

	def forward(self, h, adj):
		Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
		e = self._prepare_attentional_mechanism_input(Wh)
		zero_vec = -9e15*torch.ones_like(e)
		adj_ = adj.to_dense()
		attention = torch.where(adj_ > 0, e, zero_vec)
		attention = F.softmax(attention, dim=1)
		attention = F.dropout(attention, self.dropout, training=self.training)
		h_prime = torch.matmul(attention, Wh)

		if self.concat:
			return F.elu(h_prime)
		else:
			return h_prime

	def _prepare_attentional_mechanism_input(self, Wh):
		# Wh.shape (N, out_feature)
		# self.a.shape (2 * out_feature, 1)
		# Wh1&2.shape (N, 1)
		# e.shape (N, N)
		Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
		Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
		# broadcast add
		e = Wh1 + Wh2.T
		return self.leakyrelu(e)

	def __repr__(self):
		return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime


class MeanAggregator(nn.Module):
	"""
	Aggregates a node's embeddings using mean of neighbors' embeddings
	"""
	def __init__(self, features, cuda=False, gcn=False): 
		"""
		Initializes the aggregator for a specific graph.
		features -- function mapping LongTensor of node ids to FloatTensor of feature values.
		cuda -- whether to use GPU
		gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
		"""

		super(MeanAggregator, self).__init__()

		self.features = features
		self.cuda = cuda
		self.gcn = gcn

	def forward(self, nodes, to_neighs, num_sample=10):
		"""
		nodes --- list of nodes in a batch
		to_neighs --- list of sets, each set is the set of neighbors for node in batch
		num_sample --- number of neighbors to sample. No sampling if None.
		"""
		# Local pointers to functions (speed hack)
		_set = set
		if not num_sample is None:
			_sample = random.sample
			samp_neighs = [_set(_sample(to_neigh, 
							num_sample,
							)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
		else:
			samp_neighs = to_neighs

		if self.gcn:
			samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
		unique_nodes_list = list(set.union(*samp_neighs))
		#  print ("\n unl's size=",len(unique_nodes_list))
		unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
		mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
		row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1
		if self.cuda:
			mask = mask.cuda()
		num_neigh = mask.sum(1, keepdim=True)
		mask = mask.div(num_neigh)
		if self.cuda:
			embed_matrix = self.features[torch.LongTensor(unique_nodes_list).cuda()]
		else:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
		to_feats = mask.mm(embed_matrix)
		return to_feats


class Encoder(nn.Module):
	"""
	Encodes a node's using 'convolutional' GraphSage approach
	"""
	def __init__(self, features, feature_dim, 
		embed_dim, adj_lists, num_sample=10,
		base_model=None, gcn=False, cuda=False): 
		super(Encoder, self).__init__()

		self.features = features
		self.feat_dim = feature_dim
		self.adj_lists = adj_lists
		self.aggregator = MeanAggregator(features, cuda, gcn)
		self.num_sample = num_sample
		if base_model != None:
			self.base_model = base_model

		self.gcn = gcn
		self.embed_dim = embed_dim
		self.cuda = cuda
		self.aggregator.cuda = cuda
		self.weight = nn.Parameter(
				torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))

		init.xavier_uniform(self.weight)

	def forward(self, nodes):
		"""
		Generates embeddings for a batch of nodes.
		nodes     -- list of nodes
		"""
		neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes], 
				self.num_sample)
		if not self.gcn:
			if self.cuda:
				self_feats = self.features[nodes].cuda()
			else:
				self_feats = self.features[nodes]
			combined = torch.cat([self_feats, neigh_feats], dim=1)
		else:
			combined = neigh_feats
		combined = F.relu(self.weight.mm(combined.t()))
		return combined