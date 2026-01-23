import torch.nn as nn
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal

class DualTemporal(nn.Module):
    """
    在时间步维度使用嵌入的TimeCMA模型
    嵌入形状: [B, L, E] (L=时间步数, E=嵌入维度)
    """
    def __init__(
        self,
        device = "cuda:0",
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
        
        # 将节点维度压缩到特征维度
        self.nodes_to_feature = nn.Linear(self.num_nodes, self.channel).to(self.device)
        
        # 投影回节点数
        self.feature_to_nodes = nn.Linear(self.channel, self.num_nodes).to(self.device)

        # Time Series Encoder - 在时间步维度上做注意力
        self.ts_encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.channel, 
            nhead = self.head, 
            batch_first=True, 
            norm_first = True,
            dropout = self.dropout_n
        ).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(self.ts_encoder_layer, num_layers = self.e_layer).to(self.device)

        # Prompt Encoder - 对时间步嵌入进行编码
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.d_llm, 
            nhead = self.head, 
            batch_first=True, 
            norm_first = True,
            dropout = self.dropout_n
        ).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(self.prompt_encoder_layer, num_layers = self.e_layer).to(self.device)

        # Cross-modality alignment - 在时间步维度对齐
        # 使用标准的TransformerDecoderLayer进行跨模态对齐
        self.cross_layer = nn.TransformerDecoderLayer(
            d_model = self.channel,  # Q的维度
            nhead = 1, 
            batch_first=True, 
            norm_first = True,
            dropout = self.dropout_n
        ).to(self.device)
        self.cross = nn.TransformerDecoder(self.cross_layer, num_layers = 1).to(self.device)
        
        # 将文本嵌入维度E投影到时间序列特征维度C
        self.embed_proj = nn.Linear(self.d_llm, self.channel).to(self.device)

        # Transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model = self.channel, 
            nhead = self.head, 
            batch_first=True, 
            norm_first = True, 
            dropout = self.dropout_n
        ).to(self.device)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = self.d_layer).to(self.device)

        # Projection: 特征维度 -> 预测长度
        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_data, input_data_mark, embeddings):
        """
        Args:
            input_data: [B, L, N] 时间序列数据
            input_data_mark: [B, L, M] 时间戳特征
            embeddings: [B, L, E] 时间步维度的文本嵌入
        Returns:
            dec_out: [B, L_pred, N] 预测结果
        """
        input_data = input_data.float()
        input_data_mark = input_data_mark.float()
        embeddings = embeddings.float()

        # RevIN归一化
        input_data = self.normalize_layers(input_data, 'norm')  # [B, L, N]

        # 将节点维度压缩到特征维度: [B, L, N] -> [B, L, C]
        input_data = self.nodes_to_feature(input_data)  # [B, L, C]

        # Time Series Encoder - 在时间步之间做注意力
        enc_out = self.ts_encoder(input_data)  # [B, L, C]

        # Prompt Encoder - 对时间步嵌入编码
        embeddings = self.prompt_encoder(embeddings)  # [B, L, E]
        
        # 将文本嵌入投影到时间序列特征维度
        embeddings_proj = self.embed_proj(embeddings)  # [B, L, E] -> [B, L, C]

        # Cross-modality alignment
        # TransformerDecoder: tgt=enc_out [B, L, C], memory=embeddings_proj [B, L, C]
        cross_out = self.cross(enc_out, embeddings_proj)  # [B, L, C]

        # Decoder
        dec_out = self.decoder(cross_out, cross_out)  # [B, L, C]

        # Projection: 将特征维度投影到预测长度
        # 我们需要 [B, pred_len, N]，所以先转置再投影
        dec_out = dec_out.permute(0, 2, 1)  # [B, C, L]
        dec_out = self.c_to_length(dec_out)  # [B, C, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [B, pred_len, C]
        
        # 投影回节点数: [B, pred_len, C] -> [B, pred_len, N]
        dec_out = self.feature_to_nodes(dec_out)  # [B, pred_len, N]

        # 反归一化
        dec_out = self.normalize_layers(dec_out, 'denorm')  # [B, pred_len, N]

        return dec_out
