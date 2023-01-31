import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as func
from typing import Tuple, Optional
from torch import Tensor
import torch.nn.functional as F
from scipy import spatial
import pickle as pickle
import numpy as np
# from transformers import BertModel,BertTokenizer
# from transformers import AutoTokenizer
from transformers import DistilBertTokenizer, DistilBertModel
from prediction import PredictionLayer
from fusion import FusionLayer
import random
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = func.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src
        attns = []

        for mod in self.layers:
            output, attn = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attns.append(attn)
        attns = torch.stack(attns)

        if self.norm is not None:
            output = self.norm(output)

        return output, attns


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return func.relu
    elif activation == "gelu":
        return func.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # d_model: word embedding size
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len,) -> (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)
        '''
        probably to prevent from rounding error
        e^(idx * (-log 10000 / d_model)) -> (e^(log 10000))^(- idx / d_model) -> 10000^(- idx / d_model) -> 1/(10000^(idx / d_model))
        since idx is an even number, it is equal to that in the formula
        '''
        pe[:, 0::2] = torch.sin(position * div_term)  # even number index, (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)  # odd number index
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, d_model) -> (1, max_len, d_model) -> (max_len, 1, d_model)
        self.register_buffer('pe', pe)  # will not be updated by back-propagation, can be called via its name

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MLP(nn.Module):
    def __init__(self, emsize=384):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(emsize, emsize)
        self.linear2 = nn.Linear(emsize, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()

    def forward(self, hidden):  # (batch_size, emsize)
        mlp_vector = self.sigmoid(self.linear1(hidden))  # (batch_size, emsize)
        rating = torch.squeeze(self.linear2(mlp_vector))  # (batch_size,)
        return rating


def generate_square_subsequent_mask(total_len):
    mask = torch.tril(torch.ones(total_len, total_len))  # (total_len, total_len), lower triangle -> 1.; others 0.
    mask = mask == 0  # lower -> False; others True
    return mask


def generate_peter_mask(src_len, tgt_len):
    total_len = src_len + tgt_len
    mask = generate_square_subsequent_mask(total_len)
    mask[0, 1] = False  # allow to attend for user and item
    mask[0, 2] = False
    mask[0, 3] = False
    mask[1, 2] = False
    mask[1, 3] = False
    mask[2, 3] = False
    return mask


class MLP1(nn.Module):
    def __init__(self, emsize=384):
        super(MLP1, self).__init__()
        self.linear3 = nn.Linear(emsize*2, emsize)
        self.linear4 = nn.Linear(emsize, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear3.weight.data.uniform_(-initrange, initrange)
        self.linear4.weight.data.uniform_(-initrange, initrange)
        self.linear3.bias.data.zero_()
        self.linear4.bias.data.zero_()

    def forward(self, hidden):  # (batch_size, emsize)
        mlp_vector = self.sigmoid(self.linear3(hidden))  # (batch_size, emsize)
        rating = torch.squeeze(self.linear4(mlp_vector))  # (batch_size,)
        return rating, mlp_vector

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class ContrastLoss(nn.Module):
    def __init__(self, emsize,hidden_dim, word2idx, idx2word, device):
        super(ContrastLoss, self).__init__()
        # 1、对rating-vetor和bert处理过的向量的线性变化
        hidden_dim = int(hidden_dim)
        self.device = device
        self.linear_r = nn.Linear(emsize*4, 384)
        self.linear_t = nn.Linear(768, 384)
        self.linear = nn.Linear(384,192)
        self.relu = nn.ReLU(inplace=True)
        # 2、bert模型处理句子向量
        self.bert_model = DistilBertModel.from_pretrained("/home/wangshuo/weights/distilbert/")
        self.bert_tokenizer = DistilBertTokenizer.from_pretrained("/home/wangshuo/weights/distilbert/")
        self.sim = Similarity(temp=0.05)# todo 这个值暂定
        self.loss_fct = nn.CrossEntropyLoss()

        self.word2idx = word2idx
        self.idx2word = idx2word
        self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.linear_r.weight.data.uniform_(-initrange, initrange)
        self.linear_t.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear_r.bias.data.zero_()
        self.linear_t.bias.data.zero_()
        self.linear.bias.data.zero_()

    # 将传入的文本向量映射到词然后，使用berttokenizer进行处理
    def map_bert(self,word_prob):
        shape = word_prob.shape
        word_prob = word_prob.permute(1,0,2) # bs tgt_len ntoken
        word_prob = word_prob.exp()
        word_idx = torch.argmax(word_prob,-1) # 这里维度会消失一个 bs tgt_len ntoken
        word_idx = word_idx.tolist()
        tokens = [self.ids2tokens(ids) for ids in word_idx]
        tokens = [' '.join(token) for token in tokens]
        encoded_input = self.bert_tokenizer.batch_encode_plus(tokens, padding=True, truncation=True,  return_tensors="pt",max_length=30)
        encoded_input = encoded_input.to(self.device)
        output = self.bert_model(**encoded_input)
        features = output[0][:,0,:]
        return features
        
    def ids2tokens(self, ids):
        eos = self.word2idx['<eos>']
        tokens = []
        for i in ids:
            if i == eos:
                break
            token = self.idx2word[i]
            final = token[-1]
            if not final.isalpha():
                token = token[0:-1]
            tokens.append(token)
        return tokens

    def forward(self,rating_vec,word_prob):
        # 将他们映射到同一空间
        rating = self.linear_r(rating_vec)# bs*hidden
        # rating = rating_vec
        seq = self.map_bert(word_prob) # bs * words * emsize
        text = self.linear_t(seq) #bs*hidden
        # 使用同一个linear，进行缩小
        rating = self.linear(self.relu(rating))
        text = self.linear(self.relu(text))
        # 开始计算对比损失

        # 先计算两个相似度
        # 维度的补充
        sim_t_r = self.sim(rating.unsqueeze(1), text.unsqueeze(0)) #(bs bs) 
        lables = torch.arange(sim_t_r.size(0)).to(self.device)
        cl_closs = self.loss_fct(sim_t_r,lables)
        return cl_closs
class Sigle_contrastLoss(nn.Module):
    def __init__(self) -> None:
        super(Sigle_contrastLoss,self).__init__()
        self.T = 0.05

    def aug_data(self,data):
        pass

    def contrastive_loss(self, x, x_aug, T):
        """
        :param x: the hidden vectors of original data
        :param x_aug: the positive vector of the auged data
        :param T: temperature
        :return: loss
        """
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
    
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        # 索引正好是 1,1 2,2 这样就是正样本对
        pos_sim = sim_matrix[range(batch_size), range(batch_size)] 
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

    def forward(self,user,item):
        user_loss = self.contrastive_loss(user,user,self.T)
        item_loss = self.contrastive_loss(item,item,self.T)
        return (user_loss + item_loss)/2
        

class SERMON(nn.Module):
    def __init__(self, peter_mask, src_len, tgt_len, pad_idx, nuser, nitem, ntoken, emsize, nhead, nhid, nlayers, word2idx, idx2word, device, output, dropout=0.5):
        super(SERMON, self).__init__()
        self.pos_encoder = PositionalEncoding(emsize, dropout)  # emsize: word embedding size
        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid, dropout)  # nhid: dim_feedforward, one basic layer, including multi-head attention and FFN
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)  # loop over the one above
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)

        self.word_embeddings = nn.Embedding(ntoken, emsize)
        self.hidden2token = nn.Linear(emsize, ntoken)

        feature_dim = 4*emsize
        self.predict_net = PredictionLayer(user_num=nuser,item_num=nitem,feature_dim=feature_dim,output=output)
        self.recommender1 = MLP1(emsize)

        self.first_layer = nn.Linear(emsize+60, emsize)
        self.last_layer = nn.Linear(emsize, 1)
        layer = nn.Linear(emsize, emsize)
        self.layers = _get_clones(layer, 3)
        self.sigmoid = nn.Sigmoid()
        self.ui_len = 2
        self.src_len = src_len+2
        self.pad_idx = pad_idx
        self.emsize = emsize
        if peter_mask:
            self.attn_mask = generate_peter_mask(src_len+2, tgt_len)
        else:
            self.attn_mask = generate_square_subsequent_mask(src_len + tgt_len)
        self.contrast_loss = ContrastLoss(emsize=emsize,hidden_dim=emsize/2, word2idx=word2idx, idx2word=idx2word, device=device)
        self.sigel_loss = Sigle_contrastLoss()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.weight.data.uniform_(-initrange, initrange)
        self.hidden2token.bias.data.zero_()

    def predict_context(self, hidden):
        context_prob = self.hidden2token(hidden[1])  # (batch_size, ntoken)
        log_context_dis = func.log_softmax(context_prob, dim=-1)
        return log_context_dis


    def predict_seq(self, hidden):
        word_prob = self.hidden2token(hidden[self.src_len:])  # (tgt_len, batch_size, ntoken)
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob

    def generate_token(self, hidden):
        word_prob = self.hidden2token(hidden[-1])  # (batch_size, ntoken)
        log_word_prob = func.log_softmax(word_prob, dim=-1)
        return log_word_prob

    def forward(self, user, item, text, seq_prediction=True, context_prediction=True, rating_prediction=True,user_retrive=None,item_retrive=None,user_retrive_global=None,item_retrive_global=None):
        '''
        :param user: (batch_size,), torch.int64
        :param item: (batch_size,), torch.int64
        :param text: (total_len - ui_len, batch_size), torch.int64
        :param seq_prediction: bool
        :param context_prediction: bool
        :param rating_prediction: bool
        :return log_word_prob: target tokens (tgt_len, batch_size, ntoken) if seq_prediction=True; the last token (batch_size, ntoken) otherwise.
        :return log_context_dis: (batch_size, ntoken) if context_prediction=True; None otherwise.
        :return rating: (batch_size,) if rating_prediction=True; None otherwise.
        :return attns: (nlayers, batch_size, total_len, total_len)
        '''
        device = user.device
        batch_size = user.size(0)
        total_len = self.ui_len + text.size(0)+2  # deal with generation when total_len != src_len + tgt_len
        # see nn.MultiheadAttention for attn_mask and key_padding_mask
        attn_mask = self.attn_mask[:total_len, :total_len].to(device)  # (total_len, total_len)
        left = torch.zeros(batch_size, self.ui_len+2).bool().to(device)  # (batch_size, ui_len)
        right = text.t() == self.pad_idx  # replace pad_idx with True and others with False, (batch_size, total_len - ui_len)
        key_padding_mask = torch.cat([left, right], 1)  # (batch_size, total_len)
        u_src = self.user_embeddings(user)  # (1, batch_size, emsize)
        i_src = self.item_embeddings(item)  # (1, batch_size, emsize)
        w_src = self.word_embeddings(text)  # (total_len - ui_len, batch_size, emsize)
        user_retrive_glo = user_retrive_global[user].to(device)
        item_retrive_glo = item_retrive_global[item].to(device)

        src = torch.cat([u_src.unsqueeze(0),i_src.unsqueeze(0),w_src[0:2],user_retrive_glo.unsqueeze(0),item_retrive_glo.unsqueeze(0),w_src[2:]], 0)  # (total_len, batch_size, emsize)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)

        hidden, attns = self.transformer_encoder(src, attn_mask, key_padding_mask)  # (total_len, batch_size, emsize) vs. (nlayers, batch_size, total_len_tgt, total_len_src)
        hal = None
        rating_vec = None
        if rating_prediction:

            hal = torch.cat([hidden[0],hidden[1],u_src,i_src], dim=1)
            rating = self.predict_net(hal,user,item).squeeze(1) 
        else:
            rating = None
        if context_prediction:
            log_context_dis = self.predict_context(hidden)  # (batch_size, ntoken)
        else:
            log_context_dis = None
        if seq_prediction:
            log_word_prob = self.predict_seq(hidden)  # (tgt_len, batch_size, ntoken)
        else:
            log_word_prob = self.generate_token(hidden)  # (batch_size, ntoken)
        return log_word_prob, log_context_dis, rating, attns, hal, (hidden[0],hidden[1])
