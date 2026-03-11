"""
一个精简版的 GPT-2 风格 Transformer。
"""

import math
from typing import Dict, Optional, cast

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from jaxtyping import Float, Int
from torch.nn.functional import softmax
from dataclasses import dataclass
from einops import rearrange
from transformers import GPT2LMHeadModel
import huggingface_hub

from utils import state_dict_converter


# TODO: 在整个作业中加入 attention mask
# TODO: 也许可以加入 KV cache


@dataclass
class ModelConfig:
    d_model: int
    n_heads: int
    n_layers: int
    context_length: int
    vocab_size: int


class CausalAttention(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        # 按照 Attention Is All You Need 的设置计算注意力维度
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.d_attention = int(config.d_model / config.n_heads)
        self.d_model = config.d_model

        #self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)

        self.W_k = nn.Linear(config.d_model, self.d_attention * config.n_heads)
        self.W_q = nn.Linear(config.d_model, self.d_attention * config.n_heads)
        self.W_v = nn.Linear(config.d_model, self.d_attention * config.n_heads)

        self.W_o = nn.Linear(self.d_attention * config.n_heads, config.d_model)

        # 因果掩码
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length)).view(
                1, 1, config.context_length, config.context_length
            ),
            persistent=False
        )

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    , mask: Optional[Tensor] = None) -> Float[Tensor, "batch seq_len d_model"]:

        # TODO：补全实现
        batch_size, seq_len, _ = x.shape
        if mask is None:
            causal_mask = cast(Tensor, self.causal_mask)
            mask = causal_mask[:, :, :seq_len, :seq_len] == 0
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        Q = Q.view(batch_size,seq_len,self.n_heads,self.d_attention).transpose(1,2)
        K = K.view(batch_size,seq_len,self.n_heads,self.d_attention).transpose(1,2)
        V = V.view(batch_size,seq_len,self.n_heads,self.d_attention).transpose(1,2)
        attn_weights = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.d_attention)      
        scores = attn_weights.masked_fill(mask, float("-inf"))
        att = F.softmax(scores,dim = -1)
        out = att @ V
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_o(out)


        return out



class GELU(nn.Module):
    """
    GELU 激活函数实现（与 Google BERT 仓库及 OpenAI GPT 中实现一致）。
    参考论文：Gaussian Error Linear Units (GELU) https://arxiv.org/abs/1606.08415
    """

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))  # fmt: skip

class MLP(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model)
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model)
        self.gelu = GELU()

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        x = self.fc2(self.gelu(self.fc1(x)))

        # TODO：补全实现
        return x
        

class DecoderBlock(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.mlp = MLP(config)
        self.attention = CausalAttention(config)
        self.pre_layer_norm = nn.LayerNorm(config.d_model)
        self.post_layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self, x: Float[Tensor, "batch seq_len d_model"]
    ) -> Float[Tensor, "batch seq_len d_model"]:

        # TODO：补全实现
        x1 = x
        x = self.pre_layer_norm(x)
        x = self.attention(x)
        res1 = x + x1
        x2 = self.post_layer_norm(res1)
        x3 = self.mlp(x2)
        res2 = res1 + x3



        return res2


class Transformer(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.context_length, config.d_model)
        self.backbone = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

        # 初始化所有权重，并按 GPT-2 论文对残差投影层使用特殊缩放初始化
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)
                )

    def forward(
        self, x: Int[Tensor, "batch_size seq_len"]
    ) -> Float[Tensor, "batch seq_len vocab_size"]:
        batch_size,seq_len = x.shape
        token_emb = self.embeddings(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        pos_emb = self.position_embeddings(positions)
        h = token_emb + pos_emb
        for block in self.backbone:
            h = block(h)
        h = self.final_layer_norm(h)
        logits = self.lm_head(h)
        # TODO：补全实现
        return logits

    @torch.no_grad()
    def generate(
        self,
        x: Int[Tensor, "batch_size seq_len"],
        num_new_tokens: int,
    ) -> Int[Tensor, "batch_size seq_len+num_new_tokens"]:
        
        for i in range(num_new_tokens):
            logist = self.forward(x)
            next_token = logist[:,-1,:].argmax(dim = -1,keepdim =True)
            x = torch.cat([x,next_token],dim = 1)

            

        # TODO：补全实现
        return x


    def get_loss_on_batch(
        self,
        input_ids: Int[Tensor, "batch_size seq_len"], 
    ) -> Float[Tensor, ""]:
        inputs = input_ids[:,:-1]
        labels = input_ids[:,1:]

        logits = self.forward(inputs)
        Loss =F.cross_entropy(logits.reshape(-1,logits.size(-1)),labels.reshape(-1))


        # TODO：补全实现
        return Loss


    @classmethod
    def from_pretrained(cls):
        """
        这里固定加载 GPT-2 预训练模型。
        """

        # GPT-2 配置
        config = ModelConfig(
            d_model=768,
            n_heads=12,
            n_layers=12,
            context_length=1024,
            vocab_size=50257,
        )

        model = cls(config)

        # 从 HuggingFace 加载权重
        model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
        converted_state_dict: Dict[str, Tensor] = state_dict_converter(model_hf.state_dict())

        model.load_state_dict(converted_state_dict)

        return model


if __name__ == "__main__":

    # 如果你尚未登录，可取消注释这一行
    # huggingface_hub.login()
    
    model = Transformer.from_pretrained()
