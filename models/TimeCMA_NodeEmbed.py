import torch.nn as nn
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal

class Dual_NodeEmbed(nn.Module):
    """
    新模型：在节点数维度上做嵌入，而不是时间步维度
    主要变化：
    - 原模型：permute后 [B, N, L] -> Linear(L, C) -> [B, N, C]，序列长度=N，特征=C
    - 新模型：保持 [B, L, N] -> Linear(N, C) -> [B, L, C]，序列长度=L，特征=C
    """
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
        # 关键变化：在节点数维度上做嵌入，而不是时间步维度
        self.node_to_feature = nn.Linear(self.num_nodes, self.channel).to(self.device)

        # Time Series Encoder
        # 现在序列长度是L（时间步），特征维度是C（channel）
        self.ts_encoder_layer = nn.TransformerEncoderLayer(d_model = self.channel, nhead = self.head, batch_first=True, 
                                                           dropout = self.dropout_n).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(self.ts_encoder_layer, num_layers = self.e_layer).to(self.device)

        # Prompt Encoder
        # embeddings需要调整：从 [B, E, N] -> [B, L, E] 以便与时间序列对齐
        # 但embeddings原本是每个节点一个embedding，需要扩展到每个时间步
        # 这里我们使用一个投影层将节点维度的embedding扩展到时间步维度
        self.embedding_expand = nn.Linear(self.num_nodes, self.seq_len).to(self.device)
        # 将embeddings从E维投影到C维，以便与时间序列特征对齐
        self.embedding_proj = nn.Linear(self.d_llm, self.channel).to(self.device)
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(d_model = self.channel, nhead = self.head, batch_first=True, 
                                                               dropout = self.dropout_n).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(self.prompt_encoder_layer, num_layers = self.e_layer).to(self.device)

        # Cross-modality alignment
        # 现在两个模态都是 [B, L, C] 格式，需要在时间步维度上对齐
        # 但CrossModal期望的格式是 [bs * nvars x (text_num) x d_model]
        # 为了兼容，我们需要reshape: [B, L, C] -> [B*C, L, 1] 或类似
        # 实际上，我们可以使用一个简单的cross-attention层
        # 但为了保持一致性，我们仍然使用CrossModal，但需要调整输入格式
        # 简化方案：使用标准的TransformerDecoderLayer做cross-attention
        self.cross_layer = nn.TransformerDecoderLayer(d_model = self.channel, nhead = self.head, batch_first=True, 
                                                      dropout = self.dropout_n).to(self.device)

        # Transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = self.channel, nhead = self.head, batch_first=True, dropout = self.dropout_n).to(self.device)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = self.d_layer).to(self.device)

        # Projection
        # 从 [B, L, C] -> [B, pred_len, N]
        # 方法1：先映射到节点数，再扩展时间步
        self.c_to_nodes = nn.Linear(self.channel, self.num_nodes, bias=True).to(self.device)
        # 将时间步从L扩展到pred_len
        self.length_expand = nn.Linear(self.seq_len, self.pred_len, bias=True).to(self.device)

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_data, input_data_mark, embeddings, verbose=False):
        # 输入维度: 
        # - input_data: [B, L, N] = [16, 96, 7] (时间序列数据)
        # - input_data_mark: [16, 96, 6] (时间戳特征)
        # - embeddings: [16, 768, 7, 1] (GPT-2生成的文本embedding)
        if verbose:
            print(f"[输入] input_data: {input_data.shape}, input_data_mark: {input_data_mark.shape}, embeddings: {embeddings.shape}")
        input_data = input_data.float()
        input_data_mark = input_data_mark.float()
        embeddings = embeddings.float()

        # RevIN
        # 输出维度: [B, L, N] = [16, 96, 7]
        input_data = self.normalize_layers(input_data, 'norm')
        if verbose:
            print(f"[RevIN归一化后] input_data: {input_data.shape}")

        # 关键变化：不在节点和时间步之间permute，直接在节点维度上做嵌入
        # 输出维度: [B, L, C] = [16, 96, 32]
        # 将节点数维度(N)压缩成特征维度(C)
        input_data = self.node_to_feature(input_data) # [B, L, C]
        if verbose:
            print(f"[node_to_feature后] input_data: {input_data.shape}")

        # 处理embeddings
        embeddings = embeddings.float()
        # embeddings: [B, E, N, 1] -> [B, E, N]
        embeddings = embeddings.squeeze(-1) # [B, E, N] = [4, 768, 7]
        if verbose:
            print(f"[squeeze后] embeddings: {embeddings.shape}")
        
        # 将节点维度的embedding扩展到时间步维度
        # 目标：从 [B, E, N] = [4, 768, 7] 到 [B, L, E] = [4, 96, 768]
        # 方法：对每个embedding维度E，将节点数N扩展到时间步数L
        # embeddings: [B, E, N] -> reshape -> [B*E, N] -> Linear(N, L) -> [B*E, L] -> reshape -> [B, E, L] -> permute -> [B, L, E]
        B, E, N = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1).contiguous() # [B, N, E] = [4, 7, 768]
        embeddings = embeddings.view(B * E, N) # [B*E, N] = [4*768, 7] = [3072, 7]
        embeddings = self.embedding_expand(embeddings) # [B*E, L] = [3072, 96]
        embeddings = embeddings.view(B, E, self.seq_len) # [B, E, L] = [4, 768, 96]
        embeddings = embeddings.permute(0, 2, 1) # [B, L, E] = [4, 96, 768]
        # 投影到与时间序列特征相同的维度
        embeddings = self.embedding_proj(embeddings) # [B, L, C] = [4, 96, 32]
        if verbose:
            print(f"[embedding处理后] embeddings: {embeddings.shape}")

        # Encoder
        # 输出维度: [B, L, C] = [16, 96, 32]
        enc_out = self.ts_encoder(input_data) # [B, L, C]
        if verbose:
            print(f"[时间序列编码器后] enc_out: {enc_out.shape}")
        
        # 输出维度: [B, L, C] = [16, 96, 32]
        embeddings = self.prompt_encoder(embeddings) # [B, L, C]
        if verbose:
            print(f"[提示编码器后] embeddings: {embeddings.shape}")

        # Cross-modality alignment
        # 使用TransformerDecoderLayer做cross-attention
        # tgt=enc_out作为query，memory=embeddings作为key和value
        # 输出维度: [B, L, C] = [16, 96, 32]
        cross_out = self.cross_layer(tgt=enc_out, memory=embeddings) # [B, L, C]
        if verbose:
            print(f"[跨模态对齐后] cross_out: {cross_out.shape}")

        # Decoder
        # 输出维度: [B, L, C] = [16, 96, 32]
        dec_out = self.decoder(cross_out, cross_out) # [B, L, C]
        if verbose:
            print(f"[解码器后] dec_out: {dec_out.shape}")

        # Projection
        # 从 [B, L, C] -> [B, L, N]
        dec_out = self.c_to_nodes(dec_out) # [B, L, N] = [16, 96, 7]
        if verbose:
            print(f"[c_to_nodes后] dec_out: {dec_out.shape}")
        
        # 从 [B, L, N] -> [B, pred_len, N]
        # 需要将时间步从L扩展到pred_len
        # 方法：permute到 [B, N, L]，然后Linear扩展到 [B, N, pred_len]，再permute回 [B, pred_len, N]
        dec_out = dec_out.permute(0, 2, 1) # [B, N, L] = [16, 7, 96]
        dec_out = self.length_expand(dec_out) # [B, N, pred_len] = [16, 7, 96]
        dec_out = dec_out.permute(0, 2, 1) # [B, pred_len, N] = [16, 96, 7]
        if verbose:
            print(f"[length_expand后] dec_out: {dec_out.shape}")

        # denorm
        # 输出维度: [B, pred_len, N] = [16, 96, 7]
        dec_out = self.normalize_layers(dec_out, 'denorm')
        if verbose:
            print(f"[反归一化后] dec_out: {dec_out.shape}")
            print("-" * 80)

        return dec_out
