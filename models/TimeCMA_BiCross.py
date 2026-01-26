"""
TimeCMA_BiCross: 双向跨模态对齐模型

改进点：
- 原模型：单向对齐，时间序列特征作为Q，文本嵌入作为KV
- 新模型：双向对齐
  - 模块1：时间序列特征作为Q，文本嵌入作为KV（TS→Text）
  - 模块2：文本嵌入作为Q，时间序列特征作为KV（Text→TS）
  - 两个模块输出做维度统一，过FFN融合，接解码模块
"""

import torch
import torch.nn as nn
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal


class Dual_BiCross(nn.Module):
    def __init__(
        self,
        device="cuda:0",
        channel=32,
        num_nodes=7,
        seq_len=96,
        pred_len=96,
        dropout_n=0.1,
        d_llm=768,
        e_layer=1,
        d_layer=1,
        d_ff=32,
        head=8
    ):
        super().__init__()

        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout_n = dropout_n
        self.d_llm = d_llm
        self.e_layer = e_layer
        self.d_layer = d_layer
        self.d_ff = d_ff
        self.head = head

        # ============= 数据预处理层 =============
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)

        # ============= 时间序列编码器 =============
        self.ts_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            dropout=self.dropout_n
        ).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(self.ts_encoder_layer, num_layers=self.e_layer).to(self.device)

        # ============= Prompt编码器 =============
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_llm, nhead=self.head, batch_first=True,
            dropout=self.dropout_n
        ).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(self.prompt_encoder_layer, num_layers=self.e_layer).to(self.device)

        # ============= 双向跨模态对齐模块 =============
        # 模块1：时间序列 → 文本（TS作为Q，Text作为KV）
        # 输入：Q=[B,C,N], KV=[B,E,N]，输出：[B,C,N]
        self.cross_ts2text = CrossModal(
            d_model=self.num_nodes, n_heads=1, d_ff=self.d_ff, norm='LayerNorm',
            attn_dropout=self.dropout_n, dropout=self.dropout_n, pre_norm=True,
            activation="gelu", res_attention=True, n_layers=1, store_attn=False
        ).to(self.device)

        # 模块2：文本 → 时间序列（Text作为Q，TS作为KV）
        # 输入：Q=[B,E,N], KV=[B,C,N]，输出：[B,E,N]
        self.cross_text2ts = CrossModal(
            d_model=self.num_nodes, n_heads=1, d_ff=self.d_ff, norm='LayerNorm',
            attn_dropout=self.dropout_n, dropout=self.dropout_n, pre_norm=True,
            activation="gelu", res_attention=True, n_layers=1, store_attn=False
        ).to(self.device)

        # ============= 维度统一与融合层 =============
        # 将模块2的输出从E维投影到C维：[B,E,N] -> [B,C,N]
        self.text2ts_proj = nn.Linear(self.d_llm, self.channel).to(self.device)
        
        # 融合FFN：将两个对齐结果融合
        # 输入：拼接后 [B, N, 2*C]，输出：[B, N, C]
        self.fusion_ffn = nn.Sequential(
            nn.Linear(self.channel * 2, self.d_ff * 4),
            nn.GELU(),
            nn.Dropout(self.dropout_n),
            nn.Linear(self.d_ff * 4, self.channel),
            nn.Dropout(self.dropout_n)
        ).to(self.device)
        
        # 融合后的LayerNorm
        self.fusion_norm = nn.LayerNorm(self.channel).to(self.device)

        # ============= 解码器 =============
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.channel, nhead=self.head, batch_first=True,
            dropout=self.dropout_n
        ).to(self.device)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.d_layer).to(self.device)

        # ============= 输出投影层 =============
        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_data, input_data_mark, embeddings, verbose=False):
        """
        前向传播
        
        Args:
            input_data: [B, L, N] 时间序列数据
            input_data_mark: [B, L, 6] 时间戳特征
            embeddings: [B, E, N, 1] GPT-2生成的文本embedding
            verbose: 是否打印调试信息
        
        Returns:
            dec_out: [B, pred_len, N] 预测结果
        """
        if verbose:
            print(f"[输入] input_data: {input_data.shape}, embeddings: {embeddings.shape}")
        
        input_data = input_data.float()
        input_data_mark = input_data_mark.float()
        embeddings = embeddings.float()

        # ============= 1. 数据预处理 =============
        # RevIN归一化
        input_data = self.normalize_layers(input_data, 'norm')  # [B, L, N]
        
        # 时间序列嵌入
        input_data = input_data.permute(0, 2, 1)  # [B, N, L]
        input_data = self.length_to_feature(input_data)  # [B, N, C]
        if verbose:
            print(f"[时间序列嵌入后] input_data: {input_data.shape}")

        # 处理embeddings
        embeddings = embeddings.squeeze(-1)  # [B, E, N]
        embeddings = embeddings.permute(0, 2, 1)  # [B, N, E]
        if verbose:
            print(f"[Embedding处理后] embeddings: {embeddings.shape}")

        # ============= 2. 编码器 =============
        # 时间序列编码
        enc_out = self.ts_encoder(input_data)  # [B, N, C]
        enc_out_cross = enc_out.permute(0, 2, 1)  # [B, C, N]
        if verbose:
            print(f"[时间序列编码后] enc_out_cross: {enc_out_cross.shape}")

        # Prompt编码
        embeddings = self.prompt_encoder(embeddings)  # [B, N, E]
        embeddings_cross = embeddings.permute(0, 2, 1)  # [B, E, N]
        if verbose:
            print(f"[Prompt编码后] embeddings_cross: {embeddings_cross.shape}")

        # ============= 3. 双向跨模态对齐 =============
        # 模块1：时间序列 → 文本
        # Q=enc_out_cross [B,C,N], KV=embeddings_cross [B,E,N]
        # 输出：[B, C, N]
        cross_ts2text = self.cross_ts2text(enc_out_cross, embeddings_cross, embeddings_cross)
        if verbose:
            print(f"[模块1 TS→Text] cross_ts2text: {cross_ts2text.shape}")

        # 模块2：文本 → 时间序列
        # Q=embeddings_cross [B,E,N], KV=enc_out_cross [B,C,N]
        # 输出：[B, E, N]
        cross_text2ts = self.cross_text2ts(embeddings_cross, enc_out_cross, enc_out_cross)
        if verbose:
            print(f"[模块2 Text→TS] cross_text2ts: {cross_text2ts.shape}")

        # ============= 4. 维度统一与融合 =============
        # 模块1输出：[B, C, N] -> permute -> [B, N, C]
        cross_ts2text = cross_ts2text.permute(0, 2, 1)  # [B, N, C]
        
        # 模块2输出：[B, E, N] -> permute -> [B, N, E] -> proj -> [B, N, C]
        cross_text2ts = cross_text2ts.permute(0, 2, 1)  # [B, N, E]
        cross_text2ts = self.text2ts_proj(cross_text2ts)  # [B, N, C]
        if verbose:
            print(f"[维度统一后] cross_ts2text: {cross_ts2text.shape}, cross_text2ts: {cross_text2ts.shape}")

        # 拼接两个对齐结果
        fused = torch.cat([cross_ts2text, cross_text2ts], dim=-1)  # [B, N, 2*C]
        if verbose:
            print(f"[拼接后] fused: {fused.shape}")

        # 过FFN融合
        fused = self.fusion_ffn(fused)  # [B, N, C]
        
        # 残差连接：加上原始时间序列编码
        fused = fused + enc_out  # [B, N, C]
        
        # LayerNorm
        fused = self.fusion_norm(fused)  # [B, N, C]
        if verbose:
            print(f"[融合后] fused: {fused.shape}")

        # ============= 5. 解码器 =============
        dec_out = self.decoder(fused, fused)  # [B, N, C]
        if verbose:
            print(f"[解码器后] dec_out: {dec_out.shape}")

        # ============= 6. 输出投影 =============
        dec_out = self.c_to_length(dec_out)  # [B, N, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [B, pred_len, N]
        if verbose:
            print(f"[投影后] dec_out: {dec_out.shape}")

        # ============= 7. 反归一化 =============
        dec_out = self.normalize_layers(dec_out, 'denorm')  # [B, pred_len, N]
        if verbose:
            print(f"[反归一化后] dec_out: {dec_out.shape}")

        return dec_out
