from torch.nn import Parameter
import torch
import torch.nn as nn
import math
from transformers import BertModel, BertTokenizer
import numpy as np

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    # _adj = _adj / _nums
    # import ipdb;ipdb.set_trace()
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 2.0 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

def move_to_cuda(obj):
    return {key:obj[key].cuda() for key in obj}
class LabelGCN(nn.Module):
    def __init__(self, num_classes, in_channel=300, t=0, adj_file=None):
        super(LabelGCN, self).__init__()
        self.num_classes = num_classes
        # self.pooling = nn.MaxPool2d(14, 14)
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.lang_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)


        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())

    def forward(self, word_embeddings):
        # inp list
        # with torch.no_grad():
        #     word_tokens = [move_to_cuda(self.tokenizer(x, return_tensors="pt")) for x in inp]      # list [[tok]] *c
        #     word_embeddings = [self.lang_model(**word_token)[0][0][0:1] for word_token in word_tokens]  # list [d] * c
        adj = gen_adj(self.A).detach()
        word_embeddings = torch.from_numpy(word_embeddings).cuda()
        word_embeddings = word_embeddings.float()
        x = self.gc1(word_embeddings, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.transpose(0, 1)
        # x = torch.matmul(feature, x)
        return x


class LabelTransformer(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_in, nhead=8)
        self.trm_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, word_embedding):
        word_embedding = torch.from_numpy(word_embedding).cuda()
        word_embedding = word_embedding.float()
        x = word_embedding.unsqueeze(1)     # [#c, 1, d]
        out = self.trm_encoder(x)
        out = out.squeeze(1)                # [#c, d]
        return out.transpose(0,1)