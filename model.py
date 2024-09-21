import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return torch.tensor(pos_encoding, dtype=torch.float32)

def create_padding_mask(seq):
    seq = (seq == 0)
    return seq.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0  # Upper triangular matrix of 0s and 1s

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    dk = q.size(-1)
    scaled_attention_logits = matmul_qk / np.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask):
        batch_size = q.size(0)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        output = self.dense(scaled_attention)
        return output, attention_weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, len_q):
        super(PositionalEncoding, self).__init__()
        self.len_q = len_q
        self.d_model = d_model
        pe = torch.zeros(len_q, d_model)
        position = torch.arange(0, len_q, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return  self.pe[:, :self.len_q, :].to(x.device)
class DepthConv2d(nn.Module):
    def __init__(self,in_channel, out_channel,kernel_size=3,padding=1,stride=1):
        super(DepthConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channel,in_channel,kernel_size=kernel_size,padding=padding,stride=stride)
        self.pointwise = nn.Conv2d(in_channel,out_channel,kernel_size=1,padding=0,stride=1)
    def forward(self,x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
def softmax_with_bias1(x,bias):
    # x = x.cuda(1).data.cpu().numpy()
    #x = torch.tensor(x).cuda(1).data.cpu().numpy()
    x = x / bias
    exp =torch.exp(x)
    return exp /torch.sum(exp,dim=-1,keepdim=True)
class ScaledDotProductAttention1(nn.Module):  ##点乘注意力
    def __init__(self, d_k, dropout=.1):
        super(ScaledDotProductAttention1, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.ratio = nn.Parameter(torch.tensor((0.9)))
        self.bias = nn.Parameter(torch.tensor(0.8))
        self.pe = PositionalEncoding(d_model=64,len_q=49)
        self.depth = DepthConv2d(in_channel=160,out_channel=160)
        #self.pe = PosEncoding(d_mode=64, len_q=144)
    def forward(self, q, k, v, attn_mask=None):   #dec_outputs1, enc_outputs, enc_outputs
        # q: [b_size x n_heads x len_q x d_k]
        # k: [b_size x n_heads x len_k x d_k]
        # v: [b_size x n_heads x len_v x d_v] note: (len_k == len_v)

        # attn: [b_size x n_heads x len_q x len_k]
        scores = (torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor)  ##matmul函数详解https://blog.csdn.net/qsmx666/article/details/105783610/
        # print(scores.shape,0)
        #a = self.softmax(scores)
        # print(q.shape,1)
        # print(k.shape,2)
        pos = self.pe(q)
        # print(pos.shape,3)
        pos1 = self.pe(k)
        pos2 = torch.matmul(pos, pos1.transpose(-1, -2))
        #print("A",pos2.shape)
        #print("B",scores.shape)
        scores = scores+pos2
        if attn_mask is not None:
            assert attn_mask.size() == scores.size()
            scores.masked_fill_(attn_mask, -1e9)   ##masked_fill方法有两个参数，maske和value，mask是一个pytorch张量（Tensor），元素是布尔值，value是要填充的值，填充规则是mask中取值为True位置对应于self的相应位置用value填充
        #ratio = 0.90
        ratio = torch.sigmoid(self.ratio)
        top_k = int(ratio*scores.shape[-1])
        val,indices = torch.topk(scores,top_k,dim=-1)
        #print(val.shape)
        filter_value = -float('inf')
        index = scores<val[:,:,:,-1].unsqueeze(-1).repeat(1,1,1,scores.shape[-1])
        scores_ = scores.detach()
        scores_[index] = filter_value
        #b = self.softmax(scores_)
        # print(a[:,:,:,1])
        #pos =
        b = softmax_with_bias1(scores_,self.bias)
        # #print(a[:,:,:,1])
        #
        #b = torch.tensor(b)
        b = b.clone().detach()
        attn = self.dropout(b)
        context = torch.matmul(attn, v)
        #print(v.shape)
        v = v.permute(0,3,1,2)
        #print(v.shape)
        v1 = self.depth(v)
        #print(v1.shape)
        v1 = v1.permute(0,2,3,1)
        context = context+v1
        return context, attn
class MultiHeadAttention1(nn.Module):
    def __init__(self, d_model, num_heads,rate):
        super(MultiHeadAttention1, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
        self.scaled_dot = ScaledDotProductAttention1(d_model,rate)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask):#c_att,c_att,encoder_out
        batch_size = q.size(0)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = self.scaled_dot(q, k, v, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        output = self.dense(scaled_attention)
        return output, attention_weights
# Memory Multi-Head Attention
class MemoryMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_memory=40):
        super(MemoryMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.k_memory = nn.Parameter(torch.randn(1, num_memory, d_model) * (1 / np.sqrt(self.depth)))
        self.v_memory = nn.Parameter(torch.randn(1, num_memory, d_model) * (1 / np.sqrt(num_memory)))
        self.k_w = nn.Linear(d_model, d_model)
        self.v_w = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask):
        batch_size = q.size(0)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        k_memory = self.k_w(self.k_memory)
        v_memory = self.v_w(self.v_memory)
        k = torch.cat([k, k_memory.expand(k.size(0), -1, -1)], dim=1)
        v = torch.cat([v, v_memory.expand(v.size(0), -1, -1)], dim=1)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        output = self.dense(scaled_attention)
        return output, attention_weights

class PointWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dff):
        super(PointWiseFeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, dff)
        self.fc2 = nn.Linear(dff, d_model)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MemoryMultiHeadAttention(d_model, num_heads,num_memory = 40)

        self.ffn = PointWiseFeedForwardNetwork(d_model, dff)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(self, x, c_att, s_att, encoder_out1):#embeddings,  c_att, s_att, encoder_out1
        attn1, _ = self.mha1(x, x, x, None)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)


        attn2, _ = self.mha2(c_att, s_att, out1, None)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)



        attn4,_ = self.mha1(out3,out3,encoder_out1,None)#  试试效果 分类器是否需要经过memory
        attn4 = self.dropout3(attn4)
        out4 = self.layernorm3(encoder_out1+attn4)
        out5 = self.ffn(out4)
        out5 = self.layernorm3(out5+out4)
        return out3,out5


class CSA_Encoder(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(CSA_Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        self.Ws = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        self.Wc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dr = nn.Dropout(rate)
        self.dr1 = nn.Dropout(rate)
        self.dr2 = nn.Dropout(rate)

    def forward(self, x):
        x_fc = self.norm((self.fc(self.dr(x))))
        # Channel-attention
        c_avg = torch.mean(x_fc, dim=1, keepdim=True)
        #print("cavg",c_avg.shape)#([16, 1, 2048])
        c_att = x_fc * torch.sigmoid(self.Wc(self.dr1(c_avg)))
        #print("catt",c_att.shape)#([16, 196, 2048])

        # Spatial-attention
        s_avg = torch.mean(x_fc, dim=-1, keepdims=True)
        #print("savg",s_avg.shape)#([16, 196, 1])
        s_avg = s_avg.expand_as(x_fc)
        s_att = x_fc * torch.sigmoid(self.Ws(self.dr2(s_avg)))

        return self.norm1(F.relu(c_att)), self.norm2(F.relu(s_att))

class MLP(nn.Module):
    """
    Multi Layer Perceptron for the ViT model

    Args:
    hidden_size (int): Hidden size
    intermediate_size (int): Intermediate size
    hidden_dropout_prob (float): Dropout probability for hidden layers. Defaults to 0.0.

    Returns:
    X (Tensor): Tensor containing the output of the MLP
    """
    def __init__(self, d_model,dff1, rate):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_model, dff1)
        self.GELU = nn.GELU()
        self.fc2 = nn.Linear(dff1, d_model)
        self.dropout = nn.Dropout(rate)

    def forward(self, X):
        X = self.fc1(X)
        X = self.GELU(X)
        X = self.fc2(X)
        X = self.dropout(X)
        return X


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention1(d_model, num_heads,rate)
        self.ffn = PointWiseFeedForwardNetwork(d_model, dff)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)

    def forward(self,encoder_out):

        attn1, _ = self.mha1(encoder_out,encoder_out,encoder_out, None)
        attn1 = self.dropout2(attn1)
        out1 = self.layernorm2(encoder_out + attn1)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout3(ffn_output)
        out2 = self.layernorm3(out1 + ffn_output)



        return out2



# Memory Transformer
class MemoryTransformer(nn.Module):
    def __init__(self, num_layers1, num_layers, d_model, num_heads, dff, num_classes, dff1, vocab_size, encoder_dim=2048, dropout=0.1):
        #num_layers1,num_layers, d_model, num_heads, dff, num_classes, dff1,vocab_size, dropout=dropout
        super(MemoryTransformer, self).__init__()
        self.num_layers = num_layers
        self.num_layers1 = num_layers1
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)])
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers1)])
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(1000, d_model)
        self.final_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mlp = MLP(d_model,dff1,dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        self.Eencoder = CSA_Encoder(d_model, num_heads, dff, dropout)
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        seq_len = encoded_captions.size(1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        embeddings = self.embedding(encoded_captions)
        embeddings *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        embeddings += self.pos_encoding[:, :seq_len, :].to(embeddings.device)
        embeddings = self.dropout(embeddings)

        decode_lengths = [c - 1 for c in caption_lengths]
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels)

        for i in range(self.num_layers1):
            encoder_out = self.encoder_layers[i](encoder_out)

        c_att, s_att = self.Eencoder(encoder_out)

        encoder_out1 = encoder_out.clone()

        for i in range(self.num_layers):
            embeddings, encoder_out1 = self.decoder_layers[i](embeddings,  c_att, s_att, encoder_out1)
#分类
        # 分类
        #只用最后一个维度 忽略时间信息
        '''
        mlp_output = self.mlp(encoder_out1)
        claout = encoder_out1 + mlp_output
        logits = self.classifier(claout[:, 0, :])

        # 平铺处理
        mlp_output = self.mlp(encoder_out1)
        claout = encoder_out1 + mlp_output
        print(claout.shape)# ([16, 196, 2048])torch.Size([16, 401408])
        claout = torch.flatten(claout, 1)
        print(claout.shape)#torch.Size([16, 401408])
        logits = self.classifier(claout)
        print(logits.shape)'''

        #全局池化处理
        #print(enc_outputs1.shape)
        pooled_features = F.avg_pool1d(encoder_out1.permute(0, 2, 1), kernel_size=encoder_out1.size(1))#.squeeze(2)
        #print(pooled_features.shape)                                                       ````

        pooled_features = torch.flatten(pooled_features,1)
        mlp_output = self.mlp(pooled_features)
        claout = mlp_output + pooled_features
        #print(pooled_features.shape)
        logits = self.classifier(claout)

        ''''#swin 做法
        pooled_features = F.avg_pool1d(encoder_out1.permute(0, 2, 1), kernel_size=encoder_out1.size(1))  # .squeeze(2)
        # print(pooled_features.shape)                                                       ````

        pooled_features = torch.flatten(pooled_features, 1)
        #mlp_output = self.mlp(pooled_features)
        #claout = pooled_features + mlp_output
        # print(pooled_features.shape)
        logits = self.classifier(claout)'''


        #logits = encoder_out1
        #print(logits.shape)
        #logits = self.classifier(logits[:, 0, :])
        #print(logits.shape)
        '''print(encoder_out1.shape)
        spatia_size = encoder_out1.size(0) * encoder_out1.size(1)
        encoder_out2 = encoder_out1.view(spatia_size,-1)
        print(encoder_out2.shape)
        mlp_output = self.mlp(encoder_out2)
        mlp_output = mlp_output.view(batch_size, encoder_out1.size(1), -1)
        claout = encoder_out1 + mlp_output
        logits = self.classifier(claout[:, 0, :])
        print(logits.shape)'''

#描述
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            preds = self.final_layer(self.dropout(embeddings[:batch_size_t, t, :]))
            # preds = self.final_layer(self.dropout(embeddings[:, t, :]))
            # print("preds.shape: ", preds.shape)
            # print(len(predictions[:batch_size_t, t, :]))
            predictions[:batch_size_t, t, :] = preds
            # alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths,logits,encoder_out



# 模拟输入数据
# encoder_out = torch.randn(8, 512)
# encoded_captions = torch.randint(1, 1000, (8, 13))  # 假设词汇表大小为1000
# caption_lengths = [13, 13, 10, 10, 10, 3, 3, 3]

# # 实例化模型
# num_layers = 2
# d_model = 512
# num_heads = 8
# dff = 512
# vocab_size = 1000  # 假设词汇表大小为1000
# dropout = 0.1

# model = MemoryTransformer(num_layers, d_model, num_heads, dff, vocab_size, dropout=dropout)

# # 运行模型
# with torch.no_grad():
#     model.eval()
#     predictions, _, _ = model(encoder_out, encoded_captions, caption_lengths)

# # 打印输出
# print("Predictions shape:", predictions.shape)