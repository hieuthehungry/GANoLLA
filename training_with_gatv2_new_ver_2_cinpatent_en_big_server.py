import argparse
import os
import random
import time

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerFullNeighborSampler, MultiLayerNeighborSampler
# from dgl.dataloading.pytorch import NodeDataLoader
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
# from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from torch import nn

# from dgl.dataloading.pytorch  import NodeDataLoader, GraphDataLoader
from tqdm import tqdm
torch.distributed.init_process_group(backend='gloo', rank = 0, world_size = 1)



def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)

import torch
from torch import nn
from torch.nn import functional as F


class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded', 
                 dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
            
        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                    conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    bn(self.in_channels)
                )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                    nn.ReLU()
                )
            
    def forward(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        batch_size = x.size(0)
        
        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
            
            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))
        
        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1) # number of position in x
            f_div_C = f / N
        
        y = torch.matmul(f_div_C, g_x)
        
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z


import torch as th
from torch import nn

from dgl import function as fn
from dgl.nn.functional import edge_softmax
from dgl.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair

class GAT2Conv(nn.Module):
    r"""
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature. Defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight. Defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    residual : bool, optional
        If True, use residual connection. Defaults: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.

    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=True,
                 share_weights=False,
                 **kwargs):
        super(GAT2Conv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=bias)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=bias)
        else:
        	self.fc_src = nn.Linear(
	            self._in_src_feats, out_feats * num_heads, bias=bias)
        	if share_weights:
        		self.fc_dst = self.fc_src
        	else:
	            self.fc_dst = nn.Linear(
	                self._in_src_feats, out_feats * num_heads, bias=bias)
        self.attn = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=bias)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, get_attention=False):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_dst[:graph.number_of_dst_nodes()] 
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
            graph.srcdata.update({'el': feat_src})# (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({'er': feat_dst})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))# (num_src_edge, num_heads, out_dim)
            e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)# (num_edge, num_heads, 1)
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e)) # (num_edge, num_heads)
            # message passing
            graph.update_all(fn.u_mul_e('el', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst

import math

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.conv.gatconv import GATConv
from models.gat2 import GAT2Conv
from models.dpgat import DPGATConv

class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.init_weights()
    
    def init_weights(self):
        stdv = 1./math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x, adj_matrix):
        sp = torch.matmul(x, self.weights)
        output = torch.matmul(adj_matrix, sp)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GAT(nn.Module):
    def __init__(
        self,
        node_feats,
        n_classes,
        n_layers,
        n_heads,
        n_hidden,
        n_nodes,
        activation,
        non_local_drop,
        dropout,
        input_drop,
        attn_drop,
        type,
        non_local_mode = "embedded",
        device = "cpu",
        label_features = None,
        label_edges = None,
        nn_bn_layer = True
    ):
        super().__init__()

        if type == 'GAT':
            base_layer = GATConv
        elif type == 'GAT2':
            base_layer = GAT2Conv
        elif type == 'DPGAT':
            base_layer = DPGATConv

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_nodes = n_nodes

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.node_encoder = nn.Linear(node_feats, n_hidden)
        self.conv1d = nn.Conv1d(n_hidden, n_hidden, n_nodes)
        # 'gaussian', 'embedded', 'dot', 'concatenate'
        self.non_local = NLBlockND(in_channels=n_hidden, mode= non_local_mode, dimension=1, bn_layer=nn_bn_layer)
        self.device = device
        for i in range(n_layers):
            in_hidden = n_hidden
            out_hidden = n_hidden
            # bias = i == n_layers - 1
            if type == 'GAT':
            	layer = base_layer(
                    in_hidden,
                    out_hidden // n_heads,
                    num_heads=n_heads,
                    attn_drop=attn_drop
                )
            else:
            	layer = base_layer(
                    in_hidden,
                    out_hidden // n_heads,
                    num_heads=n_heads,
                    attn_drop=attn_drop,
                    bias=False,
                 	share_weights=True,
                  allow_zero_in_degree = True
                )           	
            self.convs.append(layer)
            self.norms.append(nn.BatchNorm1d(out_hidden))

        # self.pred_linear = nn.Linear(out_hidden, out_hidden)

        self.non_local_drop =  nn.Dropout(non_local_drop)
        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.label_features = label_features.to(device)
        self.label_edges = label_edges.to(device)

        self.gcn = GCNLayer(in_features = self.label_features.shape[1], out_features = out_hidden, bias=True).to(device)
        
        self.classifier = nn.Linear(out_hidden, self.label_features.shape[0])

    def forward(self, g, in_network_chosen_nodes):
        if not isinstance(g, list):
            subgraphs = [g] * self.n_layers
        else:
            subgraphs = g
        h = g[0].srcdata["feat"].to(self.device)
        h = self.node_encoder(h)
        h = F.relu(h, inplace=True)
        h = self.input_drop(h)

        h_last = None
        for i in range(self.n_layers):
            h = self.convs[i](g[i], h).flatten(1, -1)
            if h_last is not None:
                h += h_last[: h.shape[0], :]

            h_last = h

            h = self.norms[i](h) # temp
            h = self.activation(h, inplace=True)
            h = self.dropout(h)
        

        h = torch.cat([h, torch.zeros((1, h.shape[1]) , device = self.device)])


        embedding_mat =  h[torch.LongTensor(in_network_chosen_nodes)]


        x = embedding_mat.transpose(1,2)
        x = self.non_local(x)

        x= torch.nn.MaxPool1d(self.n_nodes)(x).squeeze()
        # x = F.relu(x)

        x = self.non_local_drop(x)
        # output = self.classifier(x)

        
        label_embed = self.gcn(self.label_features, self.label_edges)
        label_embed = F.relu(label_embed)

        
        
        # h = torch.matmul(h, label_embed.T)
        
        output = torch.zeros((x.size(0), label_embed.size(0)), device=self.device)
        for j in range(label_embed.size(0)):
          output[:, j] = self.classifier((x + label_embed[j, :].unsqueeze(0)))[:, j]
        

        return output

import scipy
import pickle as pkl
import numpy as np
import pandas as pd
from dgl.data import DGLDataset
import torch
import os
import pandas as pd
# from dgl.dataloading.pytorch import NodeDataLoader
import joblib

data_path =  "data/cinpatent/intermediate_data_en"
adj_matrix = scipy.sparse.load_npz(f"{data_path}/ind.cinpatent.BCD.npz")
labels_binary = np.load(f"{data_path}/labels_binary.npy")
feat_data = np.load(f"{data_path}/feat_data.npy")

mlb = joblib.load(f"{data_path}/mlb.pkl")

data = pd.read_parquet(f"{data_path}/processed_data.parquet")

import json
with open("data/cinpatent/label_def.json", "r") as f:
    label_def = json.load(f)

from tqdm import tqdm

import wikipediaapi
wiki_wiki = wikipediaapi.Wikipedia('en')
from nltk.tokenize import sent_tokenize
import re
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('punkt')

sbert = SentenceTransformer('paraphrase-distilroberta-base-v2', device='cpu')

def clean_string(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'`.]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.lower().strip()

def get_text_from_wiki(text, n_sents=2):
    text = text.replace('-', ' ')
    page_py = wiki_wiki.page(text)
    paragraph = sent_tokenize(page_py.summary)
    if len(paragraph) == 0:
        return text
    elif len(paragraph) <= n_sents:
        return " ".join(paragraph)
    else:
        return " ".join(paragraph[:n_sents])
    
def normalizeAdjacency(W):
    assert W.size(0) == W.size(1)
    d = torch.sum(W, dim = 1)
    d = 1/torch.sqrt(d)
    D = torch.diag(d)
    return D @ W @ D 

def get_embedding_from_wiki(sbert, text, n_sent=1):
    text = get_text_from_wiki(text, n_sent)
    embedding = sbert.encode(text, convert_to_tensor=True)
    return embedding

label2id = {v: k for k, v in enumerate(mlb.classes_)}
edges = torch.zeros((len(label2id), len(label2id)))
for label in data[data["is_train"] == 1]["labels"]:
    if len(label) >= 2:
        for i in range(len(label) - 1):
            for j in range(i + 1, len(label)):
                src, tgt = label2id[label[i]], label2id[label[j]]
                edges[src][tgt] += 1
                edges[tgt][src] += 1

marginal_edges = torch.zeros((len(label2id)))

for label in data[data["is_train"] == 1]["labels"]:
    for i in range(len(label)):
        marginal_edges[label2id[label[i]]] += 1

for i in range(edges.size(0)):
    for j in range(edges.size(1)):
        if edges[i][j] != 0:
            edges[i][j] = math.log((edges[i][j] * len(data[data["is_train"] == 1]["labels"]))/(marginal_edges[i] * marginal_edges[j]))
            #filtering and reweighting
            if edges[i][j] <= 0.05:
                edges[i][j] = 0
            else:                 
                edges[i][j] = 1/(1 + math.exp((-13)*edges[i][j] + 7.0))
            

edges = normalizeAdjacency(edges + torch.diag(torch.ones(len(label2id))))

# Get embeddings from wikipedia
features = torch.zeros(len((label2id)), 768)
for label, id in tqdm(label2id.items()):
    label_decoded = label_def[label]
    features[id] = sbert.encode(label_decoded, convert_to_tensor=True)

class GraphDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='graph_dataset')

    def process(self):
        n_nodes = adj_matrix.shape[0]
        label = torch.zeros((n_nodes, labels_binary.shape[1])).type(torch.LongTensor)
        label[:labels_binary.shape[0], :] = torch.LongTensor(labels_binary)
        
        self.graph = dgl.from_scipy(adj_matrix, eweight_name = "weight")
        self.graph.ndata['feat'] = torch.Tensor(feat_data)
        self.graph.ndata['label'] = label
        self.graph.edata['weight'] = torch.Tensor(adj_matrix.data)

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = adj_matrix.shape[0]
        # train_ids_orig = data[data["is_train"] == 1].index.tolist()
        


        train_ids = data[data["is_train"] == 1].index.tolist()
        val_ids = data[data["is_dev"] == 1].index.tolist()
        test_ids = data[data["is_test"] == 1].index.tolist()

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        train_mask[train_ids] = True
        val_mask[val_ids] = True
        test_mask[test_ids] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self):
        return self.graph

    def __len__(self):
        return 1

dataset = GraphDataset()
# dataset.graph = dgl.add_self_loop(dataset.graph)
print(dataset.graph)



def vec_translate(a, my_dict):    
  return np.vectorize(my_dict.__getitem__)(a)

def get_block_and_nodes(graph, batch, num_chosen_nodes = 100, sampler = None, get_main_doc_nodes = True):
  chosen_nodes = []
  for ind in batch:
    if get_main_doc_nodes:
      _neighbors, _ = dgl.sampling.select_topk(graph, num_chosen_nodes - 1, 'weight', ind).edges()
      _neighbors = _neighbors.numpy().tolist()
      _neighbors.append(ind)
    else:
      _neighbors, _ = dgl.sampling.select_topk(graph, num_chosen_nodes, 'weight', ind).edges()
      _neighbors = _neighbors.numpy().tolist()
    chosen_nodes.append(_neighbors)

  neighborhood = []
  for _neighbors in chosen_nodes:
    neighborhood += _neighbors
  neighborhood = sorted(list(set(neighborhood)))
  block = sampler.sample_blocks(dataset.graph, neighborhood)[2][0]
  
  in_network_mapping = {neighborhood[i]: i for i in range(len(neighborhood))}

  in_network_chosen_nodes = []
  for i in range(len(chosen_nodes)):
    row_index  = [in_network_mapping[e] for e in chosen_nodes[i] if e in in_network_mapping.keys()]
    if len(row_index) < num_chosen_nodes:
      row_index += [len(neighborhood)]*(num_chosen_nodes - len(row_index))
    in_network_chosen_nodes.append(row_index)

  return block, in_network_mapping, in_network_chosen_nodes

from sklearn.metrics import accuracy_score, f1_score, ndcg_score
from torchmetrics import Precision

def train(model, sample_ind, labels, criterion, optimizer, scheduler, batch_size = 16, num_chosen_nodes = 100, sampler = None, epoch = None, device = "cuda"):
    model.train()

    loss_sum, total = 0, 0
    i = 0
    j = 0
    with tqdm(total = len(sample_ind), desc=f"Training epoch {epoch}") as pbar:
      while i < len(sample_ind):
        j = i + batch_size
        if j <= len(sample_ind):
          in_batch_sample = sample_ind[i:j]
        else:
          in_batch_sample = sample_ind[i:]
        
        block, in_network_mapping, in_network_chosen_nodes = get_block_and_nodes(dataset.graph, in_batch_sample, 
                                                                              num_chosen_nodes = num_chosen_nodes, sampler = sampler)
        pred = model([block.to(device)], in_network_chosen_nodes)
        loss = criterion(pred, labels[in_batch_sample].float().to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        count = len(in_batch_sample)
        loss_sum += loss.item() * count
        total += count

        torch.cuda.empty_cache()

        i = j
        pbar.update(batch_size)
    
    return loss_sum / total

@torch.no_grad()
def evaluate(model, sample_ind, labels, train_idx, val_idx, test_idx, batch_size = 16, num_chosen_nodes = 100, sampler = None, epoch = None, device = "cuda"):
    model.eval()

    preds = torch.zeros(labels.shape).to(device)
    # print(preds.shape)
    # Due to the limitation of memory capacity, we calculate the average of logits 'eval_times' times.
    eval_times = 1

    for _ in range(eval_times):
        i = 0
        j = 0
        with tqdm(total = len(sample_ind), desc=f"Evaluating epoch {epoch}") as pbar:

          while i < len(sample_ind):
            j = i + batch_size
            if j <= len(sample_ind):
              in_batch_sample = sample_ind[i:j]
            else:
              in_batch_sample = sample_ind[i:]
            
            block, in_network_mapping, in_network_chosen_nodes = get_block_and_nodes(dataset.graph, in_batch_sample, 
                                                                                  num_chosen_nodes = num_chosen_nodes, sampler = sampler)
            pred = model([block.to(device)], in_network_chosen_nodes)
            preds[in_batch_sample] += pred

            # print(preds[train_idx].device)
            # print(labels[train_idx].float().device)
            torch.cuda.empty_cache()
            i = j
            pbar.update(batch_size)

    preds /= eval_times

    train_loss = criterion(preds[train_idx], labels[train_idx].float()).item()
    val_loss = criterion(preds[val_idx], labels[val_idx].float()).item()
    test_loss = criterion(preds[test_idx], labels[test_idx].float()).item()
    
    labels_train = labels[train_idx].to("cpu").detach().numpy()
    labels_val = labels[val_idx].to("cpu").detach().numpy()
    labels_test = labels[test_idx].to("cpu").detach().numpy()

    preds_train = torch.sigmoid(preds[train_idx]).to("cpu").detach().numpy()
    preds_val = torch.sigmoid(preds[val_idx]).to("cpu").detach().numpy()
    preds_test = torch.sigmoid(preds[test_idx]).to("cpu").detach().numpy()

    train_acc = accuracy_score(labels_train, preds_train.round())
    val_acc = accuracy_score(labels_val, preds_val.round())
    test_acc = accuracy_score(labels_test, preds_test.round())

    n_classes = labels.shape[1]

    micro_f1 = f1_score(labels_test, preds_test.round(), average='micro')
    macro_f1 = f1_score(labels_test, preds_test.round(), average='macro')
        
    ndcg1 = ndcg_score(labels_test, preds_test, k=1)
    ndcg3 = ndcg_score(labels_test, preds_test, k=3)
    ndcg5 = ndcg_score(labels_test, preds_test, k=5)

    p1 = Precision(num_classes=n_classes, top_k=1).to(device)(torch.sigmoid(preds[test_idx]), labels[test_idx]).item()
    p3 = Precision(num_classes=n_classes, top_k=3).to(device)(torch.sigmoid(preds[test_idx]), labels[test_idx]).item()
    p5 = Precision(num_classes=n_classes, top_k=5).to(device)(torch.sigmoid(preds[test_idx]), labels[test_idx]).item()

    test_scores = {"micro_f1": micro_f1, "macro_f1": macro_f1, 
                   "ndcg1": ndcg1, "ndcg3": ndcg3, "ndcg5": ndcg5, 
                   "p1": p1, "p3": p3, "p5": p5}

    return (
        train_acc,
        val_acc,
        test_acc,
        train_loss,
        val_loss,
        test_loss,
        test_scores,
        preds,
    )

train_idx = torch.where(dataset.graph.ndata["train_mask"])[0].numpy().tolist()
val_idx = torch.where(dataset.graph.ndata["val_mask"])[0].numpy().tolist()
test_idx = torch.where(dataset.graph.ndata["test_mask"])[0].numpy().tolist()

# train_idx = dgl.distributed.node_split(g.ndata['train_mask'], pb, force_even=True)
# val_idx = 
# test_idx = torch.where(dataset.graph.ndata["test_mask"])[0].numpy().tolist()

train_batch_size = 32
device = "cuda"
num_chosen_nodes = 100
sampler = MultiLayerFullNeighborSampler(1)
# num_sample = 1000
# sampler = dgl.dataloading.NeighborSampler([num_sample])

torch.cuda.empty_cache()

gat2 = GAT(node_feats = dataset.graph.ndata["feat"].shape[1],
           n_classes = dataset.graph.ndata["label"].shape[1], 
           n_layers =1, 
           n_heads = 8, 
           n_hidden = 768,
           n_nodes = num_chosen_nodes, 
           activation = F.relu,
           non_local_drop = 0,
          dropout = 0.25, 
           input_drop = 0.25, 
           attn_drop = 0.25, type = "GAT2",
           non_local_mode = "embedded",          #  'gaussian', 'embedded', 'dot', 'concatenate'
           device = device,
          label_features = features,
          label_edges = edges).to(device)
gat2 = torch.nn.parallel.DistributedDataParallel(gat2)
criterion = nn.BCEWithLogitsLoss()


optimizer = optim.AdamW(gat2.parameters(), lr=0.0005, weight_decay=0.001)

from transformers import AdamW, get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_training_steps=100*int(len(train_idx)/train_batch_size),
                                                num_warmup_steps=0)

epochs = 100
best_micro_f1 = 0
best_scores = {}
metric_history = []
for e in range(1, epochs + 1):
  try:
    train_idx_random = random.sample(train_idx, k = len(train_idx))
    loss = train(gat2, train_idx_random, dataset.graph.ndata["label"], criterion, optimizer, scheduler, batch_size = train_batch_size, num_chosen_nodes = num_chosen_nodes, sampler = sampler, epoch = e, device = device)
    train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, test_scores, _  = evaluate(gat2, train_idx  + val_idx + test_idx , dataset.graph.ndata["label"].to(device), train_idx, val_idx, test_idx,
                                                                batch_size = train_batch_size, num_chosen_nodes = num_chosen_nodes, sampler = sampler, epoch = e, device = "cuda")
    print(f"Epoch: {e}")
    print(loss)
    print(f"Train/val/test accuracy: {train_acc}, {val_acc}, {test_acc}") 
    print(f"Train/val/test loss: {train_loss}, {val_loss}, {test_loss}")
    print(f"Scores on test")
    test_scores["epoch"] = e
    print(test_scores)
    if test_scores["micro_f1"] > best_micro_f1:
      best_micro_f1 =  test_scores["micro_f1"]
      best_scores = test_scores
    print(f"best_micro_f1: {best_micro_f1}")
      
    test_scores["epoch"] = e
    test_scores["train_acc"] = train_acc
    test_scores["val_acc"] = val_acc
    test_scores["test_acc"] = test_acc
    test_scores["train_loss"] = train_loss
    test_scores["val_loss"] = val_loss
    test_scores["test_loss"] = test_loss
    metric_history.append(test_scores)

  except KeyboardInterrupt:
    with open("metric_history.json", "w") as f:
        json.dump(metric_history, f) 
    torch.save(gat2.state_dict(), "gat2_state_dict.pt")
    print(best_scores)
    break
with open("metric_history.json", "w") as f:
        json.dump(metric_history, f)

print("best scores")
print(best_scores)

