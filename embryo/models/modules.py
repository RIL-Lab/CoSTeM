import os
import torch
from torch import nn
from collections import OrderedDict
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
import sys

# make the repository root importable no matter where the module is imported from
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class QuickGELU(nn.Module):
    """Fast GELU approximation used by CLIP: x * sigmoid(1.702 * x)."""

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)




class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, mlp_ratio, attn_mask: torch.Tensor = None, dropout=0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, int(d_model * mlp_ratio))),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(int(d_model * mlp_ratio), d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask


    def attention(self, x: torch.Tensor, attn_mask=None):
        return self.attn(x, x, x, need_weights=False, key_padding_mask=attn_mask)[0]


    def forward(self, x: torch.Tensor, attn_mask=None):
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
    

class MHAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, dropout=0.0):
        super().__init__()

        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.attn_mask = attn_mask

    def attention(self, q, k, v, attn_mask=None, need_weights=False):
        attn_out, attn_mat = self.attn(q, k, v, key_padding_mask=attn_mask, need_weights=need_weights, average_attn_weights=True)

        return attn_out, attn_mat
    
    def forward(self, q, k, v, attn_mask=None, need_weights=False):
        if not need_weights:
            q = q + self.attention(self.ln(q), self.ln(k), self.ln(v), attn_mask)
            return q

        else:
            temp_q, attn_mat = self.attention(self.ln(q), self.ln(k), self.ln(v), attn_mask, need_weights=need_weights)
            q = q + temp_q

            return q, attn_mat
    

class MLP(nn.Module):
    def __init__(self, d_model, mlp_ratio=2, dropout=0.0):
        super().__init__()

        self.ln = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, int(d_model * mlp_ratio))),
            ("gelu", QuickGELU()),
            ("dropout", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(int(d_model * mlp_ratio), d_model))
        ]))

    def forward(self, x):
        x = self.mlp(self.ln(x)) + x

        return x

    

class MultiframeIntegrationTransformer(nn.Module):
    def __init__(self, length=32, embed_dim=512, layers=1, mlp_ratio=2, dropout=0.0, pe=True):
        super(MultiframeIntegrationTransformer, self).__init__()
        transformer_heads = embed_dim // 64
        if pe:
            self.positional_embedding = nn.Parameter(torch.empty(1, length, embed_dim))
            trunc_normal_(self.positional_embedding, std=0.02)
        self.pe = pe
        
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(d_model=embed_dim, n_head=transformer_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(layers)])

        self.post_ln = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear,)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x, attn_mask=None):
        # x: b t c
        if self.pe:
            x = x + self.positional_embedding
        
        for resblock in self.resblocks:
            x = x.permute(1, 0, 2)
            x = resblock(x, attn_mask)
            x = x.permute(1, 0, 2)  

        x = self.post_ln(x)
        
        return x # b t c
    


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, mlp_ratio=2, dropout=0.0):
        super(TransformerDecoderLayer, self).__init__()

        self.cross_attn = MHAttention(d_model=d_model, n_head=n_head, dropout=dropout)

        self.mlp = MLP(d_model=d_model, mlp_ratio=mlp_ratio, dropout=dropout)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear,)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, q, k, v, attn_mask=None):
        
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        q, attn = self.cross_attn(q, k, v, attn_mask, need_weights=True)  # b n_query c
        q = q.permute(1, 0, 2)

        out = self.mlp(q)

        return out, attn



class AdaFeatSelection(nn.Module):
    def __init__(self, n_layers=1, n_queries=8, seq_len=32, embed_dim=768, n_head=8, mlp_ratio=2, dropout=0.0, pe=False):
        super().__init__()
        self.pe = pe
        self.queries = nn.Embedding(n_queries, embed_dim)
        nn.init.normal_(self.queries.weight, std=0.02)
        if pe:
            self.positional_embedding = nn.Parameter(torch.empty(1, seq_len, embed_dim))
            nn.init.normal_(self.positional_embedding, std=0.02)

        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model=embed_dim, n_head=n_head, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(n_layers)])

        self.post_ln = nn.LayerNorm(embed_dim)

    def _compute_diversity_loss(self):
        """Diversity regularization: penalize correlated queries / experts."""
        norm_queries = F.normalize(self.queries.weight, p=2, dim=-1)  # [n_q, d]
        
        sim_matrix = torch.mm(norm_queries, norm_queries.T)  # [n_q, n_q]
        
        # exclude the diagonal, i.e. the similarity of a query with itself
        mask = ~torch.eye(self.queries.weight.size(0), 
                        dtype=torch.bool,
                        device=sim_matrix.device)
        
        # mean off-diagonal similarity
        avg_sim = torch.abs(sim_matrix[mask]).mean()  # abs(): anti-correlated queries are penalized as well
        
        return avg_sim
 
    def forward(self, tgt_feats, attn_mask=None):
        # b t c
        B, T, C = tgt_feats.shape

        self.diversity_loss = self._compute_diversity_loss()

        if self.pe:
            tgt_feats = tgt_feats + (self.positional_embedding).repeat(B, 1, 1)
        
        out = (self.queries.weight).unsqueeze(0).repeat(B, 1, 1)
        attn_list = []
        for layer in self.layers:
            out, attn = layer(out, tgt_feats, tgt_feats, attn_mask)
            attn_list.append(attn)

        out = self.post_ln(out)

        attn = attn_list[0]

        return out, attn


class Adaptor(nn.Module):
    def __init__(self, in_chans, hidden_chans):
        super(Adaptor, self).__init__()

        self.in_chans = in_chans
        self.hidden_chans = hidden_chans

        self.mlp = nn.Sequential(
            nn.Linear(in_chans, hidden_chans, bias=False),
            nn.GELU(),
            nn.Linear(hidden_chans, in_chans, bias=False),
        )

        self.ln = nn.LayerNorm(in_chans)

    def forward(self, x):
        x = x + self.mlp(self.ln(x))

        return x


class MSFeatureModulation(nn.Module):
    def __init__(self, n_scale, embed_dim, hidden_dim, reduction=4):
        super().__init__()
        
        self.n_scale = n_scale
        self.adaptors = nn.ModuleList([Adaptor(in_chans=embed_dim, hidden_chans=hidden_dim) for _ in range(n_scale)])

        reduced_dim = int(embed_dim // reduction)
        self.ch_reduce_convs = nn.ModuleList([nn.Linear(embed_dim, reduced_dim, bias=False) for _ in range(n_scale)])
        

    def forward(self, feat_list: list):
        # each feat is BT N C
        assert len(feat_list) == self.n_scale

        out_feat_list = []
        out_cls_list = []
        for i in range(self.n_scale):
            adaptor = self.adaptors[i]
            feat = feat_list[i]
            feat = adaptor(feat)  # already has a residual connection inside the adaptor.
            chan_reducer = self.ch_reduce_convs[i]
            feat = chan_reducer(feat)
            out_feat_list.append(feat[:, 1:, :])
            out_cls_list.append(feat[:, 0, :])

        ms_feat = torch.cat(out_feat_list, dim=-1)
        ms_cls = torch.cat(out_cls_list, dim=-1)

        return ms_feat, ms_cls  # bt n c
    

class FeatureSelectionModule(nn.Module):
    def __init__(self, seq_len, input_dim, num_experts, num_heads=8, dropout=0.3):
        super().__init__()
        self.seq_len = seq_len
        self.num_experts = num_experts
        
        # routing network: predicts the weight of every expert
        self.linear = nn.Sequential(
                                    nn.Linear(input_dim, input_dim // 4, bias=False),
                                    nn.GELU(),
                                    nn.Linear(input_dim // 4, num_experts, bias=False)
                                    )
        
        # learnable expert tokens (1, 1, M, C)
        self.experts = nn.Parameter(torch.randn(1, 1, num_experts, input_dim))
        nn.init.normal_(self.experts, std=0.02)
        
        # cross attention between the expert tokens and the frame tokens
        self.cross_attn = MHAttention(d_model=input_dim, n_head=num_heads, dropout=dropout)
        self.ln = nn.LayerNorm(input_dim)


    def _compute_diversity_loss(self):
        """Diversity regularization: penalize correlated queries / experts."""
        norm_queries = F.normalize(self.experts.squeeze(), p=2, dim=-1)  # [n_q, d]
        
        sim_matrix = torch.mm(norm_queries, norm_queries.T)  # [n_q, n_q]
        
        # exclude the diagonal, i.e. the similarity of a query with itself
        mask = ~torch.eye(self.experts.size(2), 
                        dtype=torch.bool,
                        device=sim_matrix.device)
        
        # mean off-diagonal similarity
        avg_sim = torch.abs(sim_matrix[mask]).mean()  # abs(): anti-correlated queries are penalized as well
        
        return avg_sim

    def forward(self, x):
        """x has shape (BT, N, C)."""
        BT, N, C = x.shape
        B = BT // self.seq_len
        T = self.seq_len

        x = x.view(B, T, N, C)

        # 1. average over the spatial tokens -> (B, T, C)
        x_avg = torch.mean(x, dim=2)

        # 2. predict the routing weight of every expert -> (B, T, M)
        weights = self.linear(self.ln(x_avg))
        weights = torch.softmax(weights, dim=-1)
        self.attn_weights = weights

        # 3. cross attention: expert queries (B*T, M, C) over the frame tokens (B*T, N, C)
        q = self.experts.expand(B, T, -1, -1).reshape(B*T, self.num_experts, C)
        k = v = x.view(B*T, N, C)

        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        attn_output, attn = self.cross_attn(q, k, v, need_weights=True)
        attn_output = attn_output.permute(1, 0, 2).view(B, T, self.num_experts, C)

        # 4. weighted sum over the experts -> (B, T, C)
        output = torch.einsum('btm, btmc -> btc', weights, attn_output)

        return output, attn



