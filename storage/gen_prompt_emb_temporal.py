import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model

class GenPromptEmbTemporal(nn.Module):
    """
    在时间步维度生成嵌入：对每个时间步的所有节点值生成一个文本prompt，然后通过GPT-2得到嵌入
    输出形状: [B, L, E] (L=时间步数, E=嵌入维度)
    """
    def __init__(
        self,
        data_path = 'ETTh1',
        model_name = "gpt2",
        device = 'cuda:0',
        input_len = 96,
        d_model = 768,
        layer = 12,
        divide = 'train'
    ):  
        super(GenPromptEmbTemporal, self).__init__()
        self.data_path = data_path
        self.device = device
        self.input_len = input_len
        self.model_name = model_name
        self.d_model = d_model
        self.layer = layer
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name).to(self.device)

    def _prepare_prompt(self, input_template, in_data, in_data_mark, i, t):
        """
        为第i个样本的第t个时间步生成prompt
        Args:
            in_data: [B, L, N] 时间序列数据
            in_data_mark: [B, L, M] 时间戳特征
            i: 样本索引
            t: 时间步索引
        """
        # 获取该时间步所有节点的值
        values = in_data[i, t, :].flatten().tolist()
        values_str = ", ".join([str(int(value)) for value in values])
        
        # 计算该时间步的趋势（与前一时刻的差值）
        if t > 0:
            diff = in_data[i, t, :] - in_data[i, t-1, :]
            trends = torch.sum(diff).item()
        else:
            trends = 0.0
        trends_str = f"{trends:0f}"
        
        # 日期时间信息
        if self.data_path in ['FRED', 'ILI']:
            date_str = f"{int(in_data_mark[i,t,2]):02d}/{int(in_data_mark[i,t,1]):02d}/{int(in_data_mark[i,t,0]):04d}"
        elif self.data_path in ['ETTh1', 'ETTh2', 'ECL']:
            date_str = f"{int(in_data_mark[i,t,2]):02d}/{int(in_data_mark[i,t,1]):02d}/{int(in_data_mark[i,t,0]):04d} {int(in_data_mark[i,t,4]):02d}:00"
        else: # ETTm1, ETTm2, Weather
            date_str = f"{int(in_data_mark[i,t,2]):02d}/{int(in_data_mark[i,t,1]):02d}/{int(in_data_mark[i,t,0]):04d} {int(in_data_mark[i,t,4]):02d}:{int(in_data_mark[i,t,5]):02d}"
        
        # 构建prompt
        in_prompt = input_template.replace("value1, ..., valuen", values_str)
        in_prompt = in_prompt.replace("Trends", trends_str)
        in_prompt = in_prompt.replace("[t]", date_str)
        
        tokenized_prompt = self.tokenizer.encode(in_prompt, return_tensors="pt").to(self.device)
        return tokenized_prompt

    def forward(self, tokenized_prompt):
        with torch.no_grad():
            prompt_embeddings = self.model(tokenized_prompt).last_hidden_state
        return prompt_embeddings

    def generate_embeddings(self, in_data, in_data_mark):
        """
        为每个时间步生成嵌入
        Args:
            in_data: [B, L, N] 时间序列数据
            in_data_mark: [B, L, M] 时间戳特征
        Returns:
            embeddings: [B, L, E] 每个时间步的嵌入
        """
        input_templates = {
            'FRED': "At time [t], the values across all features were value1, ..., valuen. The trend change was Trends",
            'ILI': "At time [t], the values across all features were value1, ..., valuen. The trend change was Trends",
            'ETTh1': "At time [t], the values across all features were value1, ..., valuen. The trend change was Trends",
            'ETTh2': "At time [t], the values across all features were value1, ..., valuen. The trend change was Trends",
            'ECL': "At time [t], the values across all features were value1, ..., valuen. The trend change was Trends",
            'ETTm1': "At time [t], the values across all features were value1, ..., valuen. The trend change was Trends",
            'ETTm2': "At time [t], the values across all features were value1, ..., valuen. The trend change was Trends",
            'Weather': "At time [t], the values across all features were value1, ..., valuen. The trend change was Trends"
        }
        
        input_template = input_templates.get(self.data_path, input_templates['ETTh1'])
        
        B, L, N = in_data.shape
        max_token_count = 0
        tokenized_prompts = []
        
        # 为每个样本的每个时间步生成prompt
        for i in range(B):
            for t in range(L):
                tokenized_prompt = self._prepare_prompt(input_template, in_data, in_data_mark, i, t)
                max_token_count = max(max_token_count, tokenized_prompt.shape[1])
                tokenized_prompts.append((i, t, tokenized_prompt))
        
        # 生成嵌入并填充到固定长度
        embeddings_list = []
        for i, t, tokenized_prompt in tokenized_prompts:
            prompt_embeddings = self.forward(tokenized_prompt)
            padding_length = max_token_count - tokenized_prompt.shape[1]
            if padding_length > 0:
                last_token_embedding = prompt_embeddings[:, -1, :].unsqueeze(1)
                padding = last_token_embedding.repeat(1, padding_length, 1)
                prompt_embeddings_padded = torch.cat([prompt_embeddings, padding], dim=1)
            else:
                prompt_embeddings_padded = prompt_embeddings
            
            # 取最后一个token的嵌入作为该时间步的表示
            last_token_emb = prompt_embeddings_padded[:, -1, :]  # [1, E]
            embeddings_list.append((i, t, last_token_emb.squeeze(0)))  # [E]
        
        # 重新组织为 [B, L, E]
        embeddings = torch.zeros((B, L, self.d_model), dtype=torch.float32, device=self.device)
        for i, t, emb in embeddings_list:
            embeddings[i, t, :] = emb
        
        return embeddings
