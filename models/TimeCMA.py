import torch.nn as nn
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal

class Dual(nn.Module):
    def __init__(
        self,
        device = "cuda:7",
        channel = 32,
        num_nodes = 7,
        seq_len = 96,
        pred_len = 96,
        dropout_n = 0.1,
        d_llm = 768,
        e_layer = 1,
        d_layer = 1,
        d_ff=32,
        head =8
    ):
        super().__init__()

        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout_n= dropout_n
        self.d_llm = d_llm
        self.e_layer = e_layer
        self.d_layer = d_layer
        self.d_ff = d_ff
        self.head = head

        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)

        # Time Series Encoder
        # 使用较低的特征维度(channel=32)而非原始序列长度(seq_len=96)是Transformer架构的通用设计思想：
        # 1. 训练稳定性：较低维度减少参数量，降低梯度爆炸/消失风险，提高训练稳定性
        # 2. 计算效率：注意力机制复杂度O(n²d)，降低d可显著减少计算量
        # 3. 表达能力：通过多层Transformer和注意力机制，低维特征也能学习复杂模式
        # 4. 通用实践：ViT、BERT等模型都采用类似策略（patch embedding + 固定d_model）
        # 5. 平衡设计：在模型复杂度、训练稳定性和表达能力之间取得平衡
        self.ts_encoder_layer = nn.TransformerEncoderLayer(d_model = self.channel, nhead = self.head, batch_first=True, 
                                                           dropout = self.dropout_n).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(self.ts_encoder_layer, num_layers = self.e_layer).to(self.device)

        # Prompt Encoder
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_llm, nhead = self.head, batch_first=True, 
                                                               dropout = self.dropout_n).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(self.prompt_encoder_layer, num_layers = self.e_layer).to(self.device)

        # Cross-modality alignment
        # self.cross_layer = nn.TransformerDecoderLayer(d_model = self.num_nodes, nhead = 1, batch_first=True, norm_first = True,dropout = self.dropout_n).to(self.device)
        # self.cross = nn.TransformerDecoder(self.cross_layer, num_layers = 1).to(self.device)
        self.cross = CrossModal(d_model= self.num_nodes, n_heads= 1, d_ff=self.d_ff, norm='LayerNorm', attn_dropout=self.dropout_n, 
                                dropout=self.dropout_n, pre_norm=True, activation="gelu", res_attention=True, n_layers=1, store_attn=False).to(self.device)

        # Transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = self.channel, nhead = self.head, batch_first=True, dropout = self.dropout_n).to(self.device)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = self.d_layer).to(self.device)

        # Projection
        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_data, input_data_mark, embeddings):
        # 输入维度: 
        # - input_data: [B, L, N] = [16, 96, 7] (时间序列数据)
        # - input_data_mark: [16, 96, 6] (时间戳特征)
        # - embeddings: [16, 768, 7, 1] (GPT-2生成的文本embedding，由时间序列转换成的文本prompt经过GPT-2处理得到)
        input_data = input_data.float()
        input_data_mark = input_data_mark.float()
        embeddings = embeddings.float()

        # RevIN
        # 输出维度: [B, L, N] = [16, 96, 7]
        input_data = self.normalize_layers(input_data, 'norm')

        # 输出维度: [B, N, L] = [16, 7, 96]
        input_data = input_data.permute(0,2,1) # [B, N, L]
        
        # 将时间序列的长度维度(L)压缩成特征维度(C)，实现序列到特征的转换
        # 为什么需要这一步？
        # 1. TransformerEncoder要求输入特征维度=d_model=channel(32)，但原始数据是[B,N,L]=[B,7,96]
        #    特征维度是96，不匹配！必须压缩到32
        # 2. 架构设计：将每个节点作为一个序列元素，在节点之间做注意力学习空间关系
        #    序列长度=节点数N=7，特征维度=channel=32
        # 3. 如果省略：需要将时间步L作为序列长度，节点N作为特征维度
        #    这样会学习时间步关系而非节点关系，与设计目标不符
        # 4. 便于后续与embeddings进行跨模态对齐（两者都在节点维度上对齐）
        # 
        # 注意：理论上可以将d_model从32改成96来省略这一步，但会带来：
        # - 参数量大幅增加（96² vs 32²，约9倍）
        # - 计算量和内存占用显著增加
        # - 可能影响模型性能和训练稳定性
        # 输出维度: [B, N, C] = [16, 7, 32]
        input_data = self.length_to_feature(input_data) # [B, N, C]

        embeddings = embeddings.float()
        # embeddings来源：由时间序列数据转换成的文本prompt，经过GPT-2模型处理得到的文本embedding
        # 生成过程：时间序列数据 → 文本prompt（如"From [t1] to [t2], the values were ..."） 
        #          → GPT-2 tokenizer → tokens → GPT-2模型 → last_hidden_state → 最后一个token的embedding
        # 这不是简单的token，而是经过GPT-2 Transformer层深度处理后的语义embedding（d_llm=768）
        # 输出维度: [B, E, N] = [16, 768, 7] (E=768是GPT-2的embedding维度，N=7是节点数)
        embeddings = embeddings.squeeze(-1) # [B, E, N]
        # 输出维度: [B, N, E] = [16, 7, 768]
        embeddings = embeddings.permute(0,2,1) # [B, N, E]

        # Encoder
        # 输出维度: [B, N, C] = [16, 7, 32]
        enc_out = self.ts_encoder(input_data) # [B, N, C]
        # 输出维度: [B, C, N] = [16, 32, 7]
        enc_out = enc_out.permute(0,2,1) # [B, C, N]
        # 输出维度: [B, N, E] = [16, 7, 768]
        embeddings = self.prompt_encoder(embeddings) # [B, N, E]
        # 输出维度: [B, E, N] = [16, 768, 7]
        embeddings = embeddings.permute(0,2,1) # [B, E, N]

        # Cross
        # 跨模态对齐操作（Cross-Modal Attention）：
        # 1. 输入：Q=时间序列特征[B,C,N]=[16,32,7]，K=V=文本embeddings[B,E,N]=[16,768,7]
        # 2. 多头注意力机制（n_heads=1）：
        #    - 计算Q与K的相似度分数：attention_scores = Q·K^T / √d_k
        #    - 对分数进行softmax归一化得到注意力权重
        #    - 使用注意力权重对V进行加权求和：output = attention_weights · V
        # 3. 残差连接：output = Q + dropout(attention_output)（在注意力层）
        # 4. 前馈网络（FFN）：经过两层线性变换和GELU激活
        # 5. 残差连接：output = output + dropout(ffn_output)（在前馈层）
        # 6. LayerNorm：在每个子层前后进行归一化（pre_norm=True）
        # 作用：让时间序列特征（Q）从文本embeddings（KV）中学习相关信息，实现跨模态对齐
        # 输出维度: [B, C, N] = [16, 32, 7] (Q: [16,32,7] X KV: [16,768,7])
        cross_out = self.cross(enc_out, embeddings, embeddings) # Q X KV  [B, C, N]X[B, E, N] = [B, C, N]
        # 输出维度: [B, N, C] = [16, 7, 32]
        cross_out = cross_out.permute(0,2,1) # [B, N, C]

        # Decoder
        # 输出维度: [B, N, C] = [16, 7, 32]
        dec_out = self.decoder(cross_out, cross_out) # [B, N, C]

        # Projection
        # 输出维度: [B, N, L] = [16, 7, 96]
        dec_out = self.c_to_length(dec_out) # [B, N, L]
        # 输出维度: [B, L, N] = [16, 96, 7]
        dec_out = dec_out.permute(0,2,1) # [B, L, N]

        # denorm
        # 输出维度: [B, L, N] = [16, 96, 7]
        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out