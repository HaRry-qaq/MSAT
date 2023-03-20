import argparse
import logging
import pathlib
import pprint
import sys
import numpy as np
import copy

import torch.nn as nn

import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn
from x_transformers.autoregressive_wrapper import (
    ENTMAX_ALPHA,
    entmax,
    exists,
    top_a,
    top_k,
    top_p,
)
from x_transformers.x_transformers import (
    AbsolutePositionalEmbedding,
    AttentionLayers,
    Decoder,
    TokenEmbedding,
    always,
    default,
    exists,
)
sys.path.append(r'/work100/weixp/MSMM-main/baseline/MTMT-bar')
import representation
import utils

def freeze(model,*args):
    for name, param in model.named_parameters():
        for n in args:
            if n in name:
                param.requires_grad = False

def clones(module,N):
    '''
    uitilizing the clones function to initialize them together into a network layer list object.
    module: network layer which will be cloned.
    N: the number of cloning.
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MusicTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        atten_strategy,
        encoding,
        max_seq_len,
        attn_layers,
        emb_dim=None,
        max_beat=None,
        max_mem_len=0.0,
        shift_mem_down=0,
        emb_dropout=0.0,
        num_memory_tokens=None,
        tie_embedding=False,
        use_abs_pos_emb=True,
        l2norm_embed=False,
    ):
        super().__init__()
        assert isinstance(
            attn_layers, AttentionLayers
        ), "attention layers must be one of Encoder or Decoder"
        
        
        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)
        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down
        self.atten_strategy = atten_strategy
        '''
        self.sigmoid = nn.Sigmoid()'''
        n_tokens = encoding["n_tokens"]
        if max_beat is not None:
            beat_dim = encoding["dimensions"].index("beat")
            n_tokens[beat_dim] = max_beat + 1

        self.l2norm_embed = l2norm_embed

        self.token_emb = clones(nn.ModuleList(
            [
                TokenEmbedding(emb_dim, n, l2norm_embed=l2norm_embed)
                for n in n_tokens
            ]
        ),3)

        self.pos_emb=clones((
            AbsolutePositionalEmbedding(
                emb_dim, max_seq_len, l2norm_embed=l2norm_embed
            )
            if (use_abs_pos_emb and not attn_layers.has_pos_emb)
            else always(0)
        ),3)

    
        self.project_emb = clones((
            nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        ),3)


        self.emb_dropout = nn.Dropout(emb_dropout)
        # self.attn_layers is list, Using index to get attn_layers
        # eg. self.attn_layers[0]
        self.attn_layers = clones(attn_layers,3)
        self.norm = nn.LayerNorm(dim)

        self.init_()

        if atten_strategy == 'local': 
            self.self_attention_vector1 = nn.Parameter(torch.ones(5,1,3))
            self.self_attention_vector2 = nn.Parameter(torch.ones(1025,1,3))
            self.self_attention_vector3 = nn.Parameter(torch.ones(13,1,3))
            self.self_attention_vector4 = nn.Parameter(torch.ones(129,1,3)) 
            self.self_attention_vector5 = nn.Parameter(torch.ones(33,1,3)) 
            self.self_attention_vector6 = nn.Parameter(torch.ones(65,1,3))    
        elif atten_strategy == 'global':
            self.self_attention_vector1 = nn.Parameter(torch.ones(1,3))
            self.self_attention_vector2 = nn.Parameter(torch.ones(1,3))
            self.self_attention_vector3 = nn.Parameter(torch.ones(1,3))
            self.self_attention_vector4 = nn.Parameter(torch.ones(1,3)) 
            self.self_attention_vector5 = nn.Parameter(torch.ones(1,3)) 
            self.self_attention_vector6 = nn.Parameter(torch.ones(1,3)) 
                                       
                                   

        self.to_logits = (
            nn.ModuleList([nn.Linear(dim, n) for n in n_tokens])
            if not tie_embedding
            else [lambda t: t @ emb.weight.t() for emb in self.token_emb[0]]
        )

        # memory tokens (like [cls]) from Memory Transformers paper
        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(
                torch.randn(num_memory_tokens, dim)
            )

    def init_(self):
        if self.l2norm_embed:
            for te in self.token_emb:
                for emb in te:
                    nn.init.normal_(emb.emb.weight, std=1e-5)
                nn.init.normal_(self.pos_emb.emb.weight, std=1e-5)
            return
        for te in self.token_emb:
            for emb in te:
                nn.init.kaiming_normal_(emb.emb.weight)
        

    def forward(
        self,
        x,  # shape : (b, n , d)
        return_embeddings=False,
        mask=None,
        return_mems=False,
        return_attn=False,
        mems=None,
        **kwargs,
    ):
        b, _, _ = x.shape
        num_mem = self.num_memory_tokens
        atten_strategy=self.atten_strategy
        
        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = (
                mems[: self.shift_mem_down],
                mems[self.shift_mem_down :],
            )
            mems = [*mems_r, *mems_l]
        
        
        def process(x,index,mask,mems):
            x = sum(
                emb(x[..., i]) for i, emb in enumerate(self.token_emb[index])
            ) + self.pos_emb[index](x)

            x = self.emb_dropout(x)

            x = self.project_emb[index](x)

            if num_mem > 0:
                mem = repeat(self.memory_tokens, "n d -> b n d", b=b)
                x = torch.cat((mem, x), dim=1)

                        # auto-handle masking after appending memory tokens
                if exists(mask):
                    mask = F.pad(mask, (num_mem, 0), value=True)

            x, intermediates = self.attn_layers[index](
            x, mask=mask, mems=mems, return_hiddens=True, **kwargs
            )
            x = self.norm(x)
            mem, x = x[:, :num_mem], x[:, num_mem:]

            out = (
                [to_logit(x) for to_logit in self.to_logits]
                if not return_embeddings
                else x
            )
            return intermediates,out

        intermediates,out1=process(x,0,mask,mems)
        _,out2=process(x,1,mask,mems)
        _,out3=process(x,2,mask,mems)
        if atten_strategy == 'local': 
            # local self-attetion
            for t in range(len(out1)):
                t_dim = out1[t].shape[2]
                # self_attention_vector1 = self.self_attention_vector[t].to(out1[t].device)
                if t == 0:
                    self_attention_vector = self.self_attention_vector1
                elif t == 1:
                    self_attention_vector = self.self_attention_vector2
                elif t == 2:
                    self_attention_vector = self.self_attention_vector3
                elif t == 3:
                    self_attention_vector = self.self_attention_vector4
                elif t == 4:
                    self_attention_vector = self.self_attention_vector5
                elif t == 5:
                    self_attention_vector = self.self_attention_vector6
                v1, v2,v3 = self_attention_vector.split([1,1,1], dim=2)
                v1 = v1.squeeze(2)
                v2 = v2.squeeze(2)
                v3 = v3.squeeze(2)

                a1 = torch.matmul(out1[t],v1)
                a2 = torch.matmul(out2[t], v2)
                a3 = torch.matmul(out3[t], v3)

                data_1 = torch.cat((a1,a2,a3), 2)

                softmax_layer_1 = nn.Softmax(dim=2)
                data_1_to_softmax = softmax_layer_1(data_1)

                b1, b2,b3 = data_1_to_softmax.split([1,1,1], dim=2)
                # 1023,1->1023,dim
                c1 = b1.repeat_interleave(t_dim, dim=2)
                c2 = b2.repeat_interleave(t_dim, dim=2)
                c3 = b3.repeat_interleave(t_dim, dim=2)

                out1[t] = torch.mul(out1[t],c1)
                out2[t] = torch.mul(out2[t],c2)
                out3[t] = torch.mul(out3[t],c3)
                out1[t] = (out1[t]+out2[t]+out3[t])
            out = out1
        elif atten_strategy == 'global': 
            # global self-attetion
            for t in range(len(out1)):
                # self_attention_vector1 = self.self_attention_vector[t].to(out1[t].device)
                if t == 0:
                    self_attention_vector = self.self_attention_vector1
                elif t == 1:
                    self_attention_vector = self.self_attention_vector2
                elif t == 2:
                    self_attention_vector = self.self_attention_vector3
                elif t == 3:
                    self_attention_vector = self.self_attention_vector4
                elif t == 4:
                    self_attention_vector = self.self_attention_vector5
                elif t == 5:
                    self_attention_vector = self.self_attention_vector6
                    
                softmax_layer_1 = nn.Softmax(dim=1)
                data_1_to_softmax = softmax_layer_1(self_attention_vector)  
                v1, v2,v3 = data_1_to_softmax.split([1,1,1], dim=1)
                v1 = v1.squeeze(1)
                v2 = v2.squeeze(1)
                v3 = v3.squeeze(1)
                out1[t] = torch.mul(out1[t],v1)
                out2[t] = torch.mul(out2[t],v2)
                out3[t] = torch.mul(out3[t],v3)
                out1[t] = (out1[t]+out2[t]+out3[t])

            out = out1
    
    

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = (
                list(
                    map(
                        lambda pair: torch.cat(pair, dim=-2),
                        zip(mems, hiddens),
                    )
                )
                if exists(mems)
                else hiddens
            )
            new_mems = list(
                map(
                    lambda t: t[..., -self.max_mem_len :, :].detach(), new_mems
                )
            )
            return out, new_mems

        if return_attn:
            attn_maps = list(
                map(
                    lambda t: t.post_softmax_attn,
                    intermediates.attn_intermediates,
                )
            )
            return out, attn_maps

        return out


def sample(logits, kind, threshold, temperature, min_p_pow, min_p_ratio):

    """Sample from the logits with a specific sampling strategy."""
    if kind == "top_k":
        probs = F.softmax(top_k(logits, thres=threshold) / temperature, dim=-1)
    elif kind == "top_p":
        probs = F.softmax(top_p(logits, thres=threshold) / temperature, dim=-1)
    elif kind == "top_a":
        probs = F.softmax(
            top_a(logits, min_p_pow=min_p_pow, min_p_ratio=min_p_ratio)
            / temperature,
            dim=-1,
        )
    elif kind == "entmax":
        probs = entmax(logits / temperature, alpha=ENTMAX_ALPHA, dim=-1)
    else:
        raise ValueError(f"Unknown sampling strategy: {kind}")

    return torch.multinomial(probs, 1)

class MusicAutoregressiveWrapper(nn.Module):
    def __init__(self,net, encoding, ignore_index=-100, pad_value=0):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len
       
        self.sos_type_code = encoding["type_code_map"]["start-of-song"]
        self.eos_type_code = encoding["type_code_map"]["end-of-song"]
        self.son_type_code = encoding["type_code_map"]["start-of-notes"]
        self.instrument_type_code = encoding["type_code_map"]["instrument"]
        self.note_type_code = encoding["type_code_map"]["note"]

        self.dimensions = {
            key: encoding["dimensions"].index(key)
            for key in (
                "type",
                "beat",
                "position",
                "pitch",
                "duration",
                "instrument",
            )
        }
        assert self.dimensions["type"] == 0

    @torch.no_grad()
    def generate(
        self,
        start_tokens,  # shape : (b, n, d)
        seq_len,
        eos_token=None,
        temperature=1.0,  # int or list of int
        filter_logits_fn="top_k",  # str or list of str
        filter_thres=0.9,  # int or list of int
        min_p_pow=2.0,
        min_p_ratio=0.02,
        monotonicity_dim=None,
        return_attn=False,
        **kwargs,
    ):
        _, t, dim = start_tokens.shape
      
        if isinstance(temperature, (float, int)):
            temperature = [temperature] * dim
        elif len(temperature) == 1:
            temperature = temperature * dim
        else:
            assert (
                len(temperature) == dim
            ), f"`temperature` must be of length {dim}"

        if isinstance(filter_logits_fn, str):
            filter_logits_fn = [filter_logits_fn] * dim
        elif len(filter_logits_fn) == 1:
            filter_logits_fn = filter_logits_fn * dim
        else:
            assert (
                len(filter_logits_fn) == dim
            ), f"`filter_logits_fn` must be of length {dim}"

        if isinstance(filter_thres, (float, int)):
            filter_thres = [filter_thres] * dim
        elif len(filter_thres) == 1:
            filter_thres = filter_thres * dim
        else:
            assert (
                len(filter_thres) == dim
            ), f"`filter_thres` must be of length {dim}"

        if isinstance(monotonicity_dim, str):
            monotonicity_dim = [self.dimensions[monotonicity_dim]]
        else:
            monotonicity_dim = [self.dimensions[d] for d in monotonicity_dim]

        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 2:
            start_tokens = start_tokens[None, :, :]

        self.net.eval()
        out = start_tokens
        mask = kwargs.pop("mask", None)

        if mask is None:
            mask = torch.ones(
                (out.shape[0], out.shape[1]),
                dtype=torch.bool,
                device=out.device,
            )

        if monotonicity_dim is not None:
            current_values = {
                d: torch.max(start_tokens[:, :, d], 1)[0]
                for d in monotonicity_dim
            }
        else:
            current_values = None

        instrument_dim = self.dimensions["instrument"]
        type_dim = self.dimensions["type"]
        for _ in range(seq_len):
            x = out[:, -self.max_seq_len :]
            mask = mask[:, -self.max_seq_len :]

            if return_attn:
                logits, attn = self.net(
                    x, mask=mask, return_attn=True, **kwargs
                )
                logits = [l[:, -1, :] for l in logits]
            else:
                logits = [
                    l[:, -1, :] for l in self.net(x, mask=mask, **kwargs)
                ]

            # Enforce monotonicity
            if monotonicity_dim is not None and 0 in monotonicity_dim:
                for i, v in enumerate(current_values[0]):
                    logits[0][i, :v] = -float("inf")

            # Filter out sos token
            logits[0][type_dim, 0] = -float("inf")

            # Sample from the logits
            sample_type = sample(
                logits[0],
                filter_logits_fn[0],
                filter_thres[0],
                temperature[0],
                min_p_pow,
                min_p_ratio,
            )

            # Update current values
            if monotonicity_dim is not None and 0 in monotonicity_dim:
                current_values[0] = torch.maximum(
                    current_values[0], sample_type.reshape(-1)
                )

            # Iterate after each sample
            samples = [[s_type] for s_type in sample_type]
            for idx, s_type in enumerate(sample_type):
                # A start-of-song, end-of-song or start-of-notes code
                if s_type in (
                    self.sos_type_code,
                    self.eos_type_code,
                    self.son_type_code,
                ):
                    samples[idx] += [torch.zeros_like(s_type)] * (
                        len(logits) - 1
                    )
                # An instrument code
                elif s_type == self.instrument_type_code:
                    samples[idx] += [torch.zeros_like(s_type)] * (
                        len(logits) - 2
                    )
                    logits[instrument_dim][:, 0] = -float("inf")  # avoid none
                    sampled = sample(
                        logits[instrument_dim][idx : idx + 1],
                        filter_logits_fn[instrument_dim],
                        filter_thres[instrument_dim],
                        temperature[instrument_dim],
                        min_p_pow,
                        min_p_ratio,
                    )[0]
                    samples[idx].append(sampled)
                # A note code
                elif s_type == self.note_type_code:
                    for d in range(1, dim):
                        # Enforce monotonicity
                        if (
                            monotonicity_dim is not None
                            and d in monotonicity_dim
                        ):
                            logits[d][idx, : current_values[d][idx]] = -float(
                                "inf"
                            )

                        # Sample from the logits
                        logits[d][:, 0] = -float("inf")  # avoid none
                        sampled = sample(
                            logits[d][idx : idx + 1],
                            filter_logits_fn[d],
                            filter_thres[d],
                            temperature[d],
                            min_p_pow,
                            min_p_ratio,
                        )[0]
                        samples[idx].append(sampled)

                        # Update current values
                        if (
                            monotonicity_dim is not None
                            and d in monotonicity_dim
                        ):
                            current_values[d][idx] = torch.max(
                                current_values[d][idx], sampled
                            )[0]
                else:
                    raise ValueError(f"Unknown event type code: {s_type}")

            stacked = torch.stack(
                [torch.cat(s).expand(1, -1) for s in samples], 0
            )
            out = torch.cat((out, stacked), dim=1)
            mask = F.pad(mask, (0, 1), value=True)

            if exists(eos_token):
                is_eos_tokens = out[..., 0] == eos_token

                # Mask out everything after the eos tokens
                if is_eos_tokens.any(dim=1).all():
                    for i, is_eos_token in enumerate(is_eos_tokens):
                        idx = torch.argmax(is_eos_token.byte())
                        out[i, idx + 1 :] = self.pad_value
                    break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)

        if return_attn:
            return out, attn

        return out

    def forward(self, x, return_list=False, **kwargs):
        xi = x[:, :-1]
        xo = x[:, 1:]
        # help auto-solve a frequent area of confusion around input masks in auto-regressive
        # if user supplies a mask that is only off by one from the source sequence, resolve it for them
        mask = kwargs.get("mask", None)
        if mask is not None and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
            kwargs["mask"] = mask


        out = self.net(xi, **kwargs)
        losses = [
            F.cross_entropy(
                out[i].transpose(1, 2),
                xo[..., i],
                ignore_index=self.ignore_index,
            )
            for i in range(len(out))
        ]
        loss = sum(losses)

        if return_list:
            return loss, losses
        return loss

class MusicXTransformer(nn.Module):
    def __init__(self, *, atten_strategy,dim, encoding, **kwargs):
        super().__init__()
        assert "dim" not in kwargs, "dimension must be set with `dim` keyword"
        transformer_kwargs = {
            "max_seq_len": kwargs.pop("max_seq_len"),
            "max_beat": kwargs.pop("max_beat"),
            "emb_dropout": kwargs.pop("emb_dropout", 0),
            "use_abs_pos_emb": kwargs.pop("use_abs_pos_emb", True),
        }
        
        self.decoder = MusicTransformerWrapper(
            atten_strategy = atten_strategy,
            encoding=encoding,
            attn_layers=Decoder(dim=dim, **kwargs),
            **transformer_kwargs,
        )

        freeze(self.decoder,
            "pos_emb.1",
            "pos_emb.2",
            "attn_layers.1",
            "attn_layers.2",
            "token_emb.1",
            "token_emb.2")

        self.decoder = MusicAutoregressiveWrapper(
            self.decoder, encoding=encoding
        )
        
    @torch.no_grad()
    def generate(self, seq_in, seq_len, **kwargs):
        return self.decoder.generate(seq_in, seq_len, **kwargs)

    def forward(self, seq, mask=None, **kwargs):
        return self.decoder(seq, mask=mask, **kwargs)


