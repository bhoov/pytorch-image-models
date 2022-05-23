""" Energy Transformer (ET) in PyTorch

A PyTorch implement of Energy Transformers (for vision) as described in:

TODO
"""
import math
import logging
from functools import partial
from collections import OrderedDict
from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange


from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from .helpers import build_model_with_cfg, resolve_pretrained_cfg, named_apply, adapt_input_conv, checkpoint_seq
from .layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from .registry import register_model

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # experimental (may be removed)
    'et_base_patch32_224': _cfg(url='', input_size=(3, 224, 224)),
}

class ClassicalHopfield(nn.Module):
    """ Shared weight classical hopfield network (analogous to MLP) to be substituted into Vision Transformer Blocks
    """
    def __init__(self, in_features, hidden_features, act_layer=nn.ReLU, bias=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        hidden_features = hidden_features

        self.weight = nn.Parameter(torch.empty((hidden_features, in_features), **factory_kwargs))
        if bias:
            self.bias_hid = nn.Parameter(torch.empty(hidden_features, **factory_kwargs))
            self.bias_vis = nn.Parameter(torch.empty(in_features, **factory_kwargs))
        else:
            self.register_parameter("bias_hid", None)
            self.register_parameter("bias_vis", None)

        self.act = act_layer()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias_hid is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_hid, -bound, bound)

        if self.bias_vis is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight.T)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_vis, -bound, bound)

    def forward(self, x):
        x = F.linear(x, self.weight, self.bias_hid)
        x = self.act(x)
        x = F.linear(x, self.weight.T, self.bias_vis)
        return -x
    
    def energy(self, x):
        hid = F.linear(x, self.weight, self.bias_hid)
        return -0.5 * self.act(hid).pow(2).sum(-1)

class HopfieldMLP(nn.Module):
    """ A lite wrapper around `SymmetricMLP` to make it compatible with existing ViT code"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, bias=False):
        super().__init__()
        self.mlp = ClassicalHopfield(in_features, hidden_features=hidden_features, act_layer=act_layer, bias=bias)
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

    def forward(self, x):
        x = self.mlp(x)
        return x
    
    def energy(self, x):
        return self.mlp.energy(x)
    
class KQAlignedAttention(nn.Module):
    """Attention modified s.t. the vectors are particles, whose query and key are trying to align    

    Given the "headmix" projection and the correct type of layer norm, this is perfectly energy aligned
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, train_betas=False, head_dim:Optional[int]=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads if head_dim is None else head_dim
        self.betas = nn.Parameter(torch.ones(self.num_heads) * (self.head_dim ** -0.5), requires_grad=train_betas)
        zspace_dim = int(self.head_dim * num_heads)
        
        self.q_proj = nn.Linear(dim, zspace_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, zspace_dim, bias=qkv_bias)
        
    def forward(self, x):
        """`x` is going to serve as self attention to itself"""
        Q = self.q_proj(x) # B Nq Z
        K = self.k_proj(x) # B Nk Z
        Q = rearrange(Q, "b q (h z) -> b h q z", h=self.num_heads, z=self.head_dim)
        K = rearrange(K, "b k (h z) -> b h k z", h=self.num_heads, z=self.head_dim)
        
        Wq = rearrange(self.q_proj.weight, "(h z) d -> h z d", h=self.num_heads, z=self.head_dim)
        Wk = rearrange(self.k_proj.weight, "(h z) d -> h z d", h=self.num_heads, z=self.head_dim)

        F1 = torch.einsum("hzd,bhkz->bkhd", Wq, K)
        F2 = torch.einsum("hzd,bhqz->bqhd", Wk, Q)

        # Calculate the attention
        attn = (torch.einsum("bhqk,h->bhqk", Q @ K.transpose(-2, -1), self.betas)).softmax(dim=-1) # (B, H, Nq, Nk)
        T1 = -torch.einsum("bkhd,bhqk->bqd", F1, attn)
        T2 = -torch.einsum("bqhd,bhqk->bkd", F2, attn)
        return T1+T2
    
    def energy(self, x):
        Q = self.q_proj(x) # B Nq Z
        K = self.k_proj(x) # B Nk Z
        Q = rearrange(Q, "b q (h z) -> b q h z", h=self.num_heads, z=self.head_dim)
        K = rearrange(K, "b k (h z) -> b k h z", h=self.num_heads, z=self.head_dim)
        
        preatt = torch.einsum("h,bqhz,bkhz->bhqk",self.betas, Q, K)
        postatt = torch.logsumexp(preatt, dim=-1).sum(-1) # batch, heads
        return torch.einsum("bh,h->b", postatt, -1/self.betas)
    
class EnergyLayerNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], eps: float=1e-5, train_scale=True, train_bias=True, device=None, dtype=None):
        """An energy plausible version of layernorm, assumes the dimension of interest is dim=-1"""
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.train_scale = train_scale
        self.train_bias = train_bias
        self.normalized_shape=normalized_shape
        
        if self.train_scale:
            self.weight = nn.Parameter(torch.empty(1, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            
        if self.train_bias:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        if self.train_scale:
            nn.init.ones_(self.weight)
        if self.train_bias:
            nn.init.zeros_(self.bias)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight if self.train_scale else 1.
        bias = self.bias if self.train_bias else 0.
                
        std, mean = torch.std_mean(x, -1, keepdim=True)
        return weight * (x - mean) / std + bias 
    
class KQEnergyBlock(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, init_values=None, act_layer=nn.ReLU):
        super().__init__()
        self.attn = KQAlignedAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.chn = HopfieldMLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward(self, g, alpha=0.1):
        return self.attn(g) + self.chn(g)

    def energy(self, x):
        return self.attn.energy(x) + self.chn.energy(x).sum(-1)


class EnergyVisionTransformer(nn.Module):
    """ Energy Vision Transformer

    Shared blocks that lead to a fixed point
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
            class_token=True, weight_init='', fc_norm=True,
            embed_layer=PatchEmbed, norm_layer=EnergyLayerNorm, act_layer=None, block_fn=KQEnergyBlock):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        norm_layer = norm_layer or partial(EnergyLayerNorm, eps=1e-6)
        act_layer = act_layer or nn.ReLU
        self.depth = depth
        
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1 if class_token else 0
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if self.num_tokens > 0 else None
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + self.num_tokens, embed_dim) * .02)

        self.block = block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values, act_layer=act_layer)
        self.norm = norm_layer(embed_dim)

        # Classifier Head
        self.fc_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
    def step_block(self, x, alpha:float=0.1):
        g = self.norm(x)
        x = x - alpha * self.block(g)
        return x
        
    def descend_block(self, x, depth:int=12, alpha:float=0.1):
        for i in range(depth):
            self.step_block(x, alpha=alpha)
        return x

    def forward_features(self, x, alpha:float=0.1):
        x = self.patch_embed(x)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq([partial(self.step_block, alpha=alpha) for i in range(self.depth)], x)
        else:
            x = self.descend_block(x, depth=self.depth, alpha=alpha)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, alpha=0.1):
        x = self.forward_features(x, alpha=alpha)
        x = self.forward_head(x)
        return x


def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str = ''):
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def get_init_weights_vit(mode='jax', head_bias: float = 0.):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


# @torch.no_grad()
# def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
#     """ Load weights from .npz checkpoints for official Google Brain Flax implementation
#     """
#     import numpy as np

#     def _n2p(w, t=True):
#         if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
#             w = w.flatten()
#         if t:
#             if w.ndim == 4:
#                 w = w.transpose([3, 2, 0, 1])
#             elif w.ndim == 3:
#                 w = w.transpose([2, 0, 1])
#             elif w.ndim == 2:
#                 w = w.transpose([1, 0])
#         return torch.from_numpy(w)

#     w = np.load(checkpoint_path)
#     if not prefix and 'opt/target/embedding/kernel' in w:
#         prefix = 'opt/target/'

#     if hasattr(model.patch_embed, 'backbone'):
#         # hybrid
#         backbone = model.patch_embed.backbone
#         stem_only = not hasattr(backbone, 'stem')
#         stem = backbone if stem_only else backbone.stem
#         stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
#         stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
#         stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
#         if not stem_only:
#             for i, stage in enumerate(backbone.stages):
#                 for j, block in enumerate(stage.blocks):
#                     bp = f'{prefix}block{i + 1}/unit{j + 1}/'
#                     for r in range(3):
#                         getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
#                         getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
#                         getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
#                     if block.downsample is not None:
#                         block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
#                         block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
#                         block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
#         embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
#     else:
#         embed_conv_w = adapt_input_conv(
#             model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
#     model.patch_embed.proj.weight.copy_(embed_conv_w)
#     model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
#     model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
#     pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
#     if pos_embed_w.shape != model.pos_embed.shape:
#         pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
#             pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
#     model.pos_embed.copy_(pos_embed_w)
#     model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
#     model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
#     if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
#         model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
#         model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
#     # NOTE representation layer has been removed, not used in latest 21k/1k pretrained weights
#     # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
#     #     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
#     #     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
#     for i, block in enumerate(model.blocks.children()):
#         block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
#         mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
#         block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
#         block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
#         block.attn.qkv.weight.copy_(torch.cat([
#             _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
#         block.attn.qkv.bias.copy_(torch.cat([
#             _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
#         block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
#         block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
#         for r in range(2):
#             getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
#             getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
#         block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
#         block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


# def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
#     # Rescale the grid of position embeddings when loading from state_dict. Adapted from
#     # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
#     _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
#     ntok_new = posemb_new.shape[1]
#     if num_tokens:
#         posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
#         ntok_new -= num_tokens
#     else:
#         posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
#     gs_old = int(math.sqrt(len(posemb_grid)))
#     if not len(gs_new):  # backwards compatibility
#         gs_new = [int(math.sqrt(ntok_new))] * 2
#     assert len(gs_new) >= 2
#     _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
#     posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
#     posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
#     posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
#     posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
#     return posemb


# def checkpoint_filter_fn(state_dict, model):
#     """ convert patch embedding weight from manual patchify + linear proj to conv"""
#     out_dict = {}
#     if 'model' in state_dict:
#         # For deit models
#         state_dict = state_dict['model']
#     for k, v in state_dict.items():
#         if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
#             # For old models that I trained prior to conv based patchification
#             O, I, H, W = model.patch_embed.proj.weight.shape
#             v = v.reshape(O, -1, H, W)
#         elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
#             # To resize pos embedding when using model at different size from pretrained weights
#             v = resize_pos_embed(
#                 v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
#         elif 'pre_logits' in k:
#             # NOTE representation layer removed as not used in latest 21k/1k pretrained weights
#             continue
#         out_dict[k] = v
#     return out_dict


# def _create_vision_transformer(variant, pretrained=False, **kwargs):
#     if kwargs.get('features_only', None):
#         raise RuntimeError('features_only not implemented for Vision Transformer models.')

#     pretrained_cfg = resolve_pretrained_cfg(variant, kwargs=kwargs)
#     model = build_model_with_cfg(
#         VisionTransformer, variant, pretrained,
#         pretrained_cfg=pretrained_cfg,
#         pretrained_filter_fn=checkpoint_filter_fn,
#         pretrained_custom_load='npz' in pretrained_cfg['url'],
#         **kwargs)
#     return model


# @register_model
# def vit_tiny_patch16_224(pretrained=False, **kwargs):
#     """ ViT-Tiny (Vit-Ti/16)
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
#     model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_tiny_patch16_384(pretrained=False, **kwargs):
#     """ ViT-Tiny (Vit-Ti/16) @ 384x384.
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
#     model = _create_vision_transformer('vit_tiny_patch16_384', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_small_patch32_224(pretrained=False, **kwargs):
#     """ ViT-Small (ViT-S/32)
#     """
#     model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
#     model = _create_vision_transformer('vit_small_patch32_224', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_small_patch32_384(pretrained=False, **kwargs):
#     """ ViT-Small (ViT-S/32) at 384x384.
#     """
#     model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
#     model = _create_vision_transformer('vit_small_patch32_384', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_small_patch16_224(pretrained=False, **kwargs):
#     """ ViT-Small (ViT-S/16)
#     NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
#     model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_small_patch16_384(pretrained=False, **kwargs):
#     """ ViT-Small (ViT-S/16)
#     NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
#     model = _create_vision_transformer('vit_small_patch16_384', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_base_patch32_224(pretrained=False, **kwargs):
#     """ ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-1k weights fine-tuned from in21k, source https://github.com/google-research/vision_transformer.
#     """
#     model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer('vit_base_patch32_224', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_base_patch32_384(pretrained=False, **kwargs):
#     """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
#     """
#     model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer('vit_base_patch32_384', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_base_patch16_224(pretrained=False, **kwargs):
#     """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_base_patch16_384(pretrained=False, **kwargs):
#     """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer('vit_base_patch16_384', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_base_patch8_224(pretrained=False, **kwargs):
#     """ ViT-Base (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
#     """
#     model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer('vit_base_patch8_224', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_large_patch32_224(pretrained=False, **kwargs):
#     """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
#     """
#     model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
#     model = _create_vision_transformer('vit_large_patch32_224', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_large_patch32_384(pretrained=False, **kwargs):
#     """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
#     """
#     model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
#     model = _create_vision_transformer('vit_large_patch32_384', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_large_patch16_224(pretrained=False, **kwargs):
#     """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
#     model = _create_vision_transformer('vit_large_patch16_224', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_large_patch16_384(pretrained=False, **kwargs):
#     """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
#     model = _create_vision_transformer('vit_large_patch16_384', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_large_patch14_224(pretrained=False, **kwargs):
#     """ ViT-Large model (ViT-L/14)
#     """
#     model_kwargs = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, **kwargs)
#     model = _create_vision_transformer('vit_large_patch14_224', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_huge_patch14_224(pretrained=False, **kwargs):
#     """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
#     """
#     model_kwargs = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)
#     model = _create_vision_transformer('vit_huge_patch14_224', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_giant_patch14_224(pretrained=False, **kwargs):
#     """ ViT-Giant model (ViT-g/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
#     """
#     model_kwargs = dict(patch_size=14, embed_dim=1408, mlp_ratio=48/11, depth=40, num_heads=16, **kwargs)
#     model = _create_vision_transformer('vit_giant_patch14_224', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_gigantic_patch14_224(pretrained=False, **kwargs):
#     """ ViT-Gigantic model (ViT-G/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
#     """
#     model_kwargs = dict(patch_size=14, embed_dim=1664, mlp_ratio=64/13, depth=48, num_heads=16, **kwargs)
#     model = _create_vision_transformer('vit_gigantic_patch14_224', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_tiny_patch16_224_in21k(pretrained=False, **kwargs):
#     """ ViT-Tiny (Vit-Ti/16).
#     ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
#     NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
#     model = _create_vision_transformer('vit_tiny_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_small_patch32_224_in21k(pretrained=False, **kwargs):
#     """ ViT-Small (ViT-S/16)
#     ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
#     NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
#     """
#     model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
#     model = _create_vision_transformer('vit_small_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_small_patch16_224_in21k(pretrained=False, **kwargs):
#     """ ViT-Small (ViT-S/16)
#     ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
#     NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
#     model = _create_vision_transformer('vit_small_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_base_patch32_224_in21k(pretrained=False, **kwargs):
#     """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
#     NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
#     """
#     model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer('vit_base_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_base_patch16_224_in21k(pretrained=False, **kwargs):
#     """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
#     NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_base_patch8_224_in21k(pretrained=False, **kwargs):
#     """ ViT-Base model (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
#     NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
#     """
#     model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer('vit_base_patch8_224_in21k', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_large_patch32_224_in21k(pretrained=False, **kwargs):
#     """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
#     NOTE: this model has a representation layer but the 21k classifier head is zero'd out in original weights
#     """
#     model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
#     model = _create_vision_transformer('vit_large_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_large_patch16_224_in21k(pretrained=False, **kwargs):
#     """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
#     NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
#     model = _create_vision_transformer('vit_large_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_huge_patch14_224_in21k(pretrained=False, **kwargs):
#     """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
#     NOTE: this model has a representation layer but the 21k classifier head is zero'd out in original weights
#     """
#     model_kwargs = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)
#     model = _create_vision_transformer('vit_huge_patch14_224_in21k', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_base_patch16_224_sam(pretrained=False, **kwargs):
#     """ ViT-Base (ViT-B/16) w/ SAM pretrained weights. Paper: https://arxiv.org/abs/2106.01548
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer('vit_base_patch16_224_sam', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_base_patch32_224_sam(pretrained=False, **kwargs):
#     """ ViT-Base (ViT-B/32) w/ SAM pretrained weights. Paper: https://arxiv.org/abs/2106.01548
#     """
#     model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer('vit_base_patch32_224_sam', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_small_patch16_224_dino(pretrained=False, **kwargs):
#     """ ViT-Small (ViT-S/16) w/ DINO pretrained weights (no head) - https://arxiv.org/abs/2104.14294
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
#     model = _create_vision_transformer('vit_small_patch16_224_dino', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_small_patch8_224_dino(pretrained=False, **kwargs):
#     """ ViT-Small (ViT-S/8) w/ DINO pretrained weights (no head) - https://arxiv.org/abs/2104.14294
#     """
#     model_kwargs = dict(patch_size=8, embed_dim=384, depth=12, num_heads=6, **kwargs)
#     model = _create_vision_transformer('vit_small_patch8_224_dino', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_base_patch16_224_dino(pretrained=False, **kwargs):
#     """ ViT-Base (ViT-B/16) /w DINO pretrained weights (no head) - https://arxiv.org/abs/2104.14294
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer('vit_base_patch16_224_dino', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_base_patch8_224_dino(pretrained=False, **kwargs):
#     """ ViT-Base (ViT-B/8) w/ DINO pretrained weights (no head) - https://arxiv.org/abs/2104.14294
#     """
#     model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer('vit_base_patch8_224_dino', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_base_patch16_224_miil_in21k(pretrained=False, **kwargs):
#     """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
#     Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs)
#     model = _create_vision_transformer('vit_base_patch16_224_miil_in21k', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_base_patch16_224_miil(pretrained=False, **kwargs):
#     """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
#     Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs)
#     model = _create_vision_transformer('vit_base_patch16_224_miil', pretrained=pretrained, **model_kwargs)
#     return model


# # Experimental models below

# @register_model
# def vit_base_patch32_plus_256(pretrained=False, **kwargs):
#     """ ViT-Base (ViT-B/32+)
#     """
#     model_kwargs = dict(patch_size=32, embed_dim=896, depth=12, num_heads=14, init_values=1e-5, **kwargs)
#     model = _create_vision_transformer('vit_base_patch32_plus_256', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_base_patch16_plus_240(pretrained=False, **kwargs):
#     """ ViT-Base (ViT-B/16+)
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=896, depth=12, num_heads=14, init_values=1e-5, **kwargs)
#     model = _create_vision_transformer('vit_base_patch16_plus_240', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_base_patch16_rpn_224(pretrained=False, **kwargs):
#     """ ViT-Base (ViT-B/16) w/ residual post-norm
#     """
#     model_kwargs = dict(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, init_values=1e-5, class_token=False,
#         block_fn=ResPostBlock, global_pool=kwargs.pop('global_pool', 'avg'), **kwargs)
#     model = _create_vision_transformer('vit_base_patch16_rpn_224', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_small_patch16_36x1_224(pretrained=False, **kwargs):
#     """ ViT-Base w/ LayerScale + 36 x 1 (36 block serial) config. Experimental, may remove.
#     Based on `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
#     Paper focuses on 24x2 + 48x1 for 'Small' width but those are extremely slow.
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=384, depth=36, num_heads=6, init_values=1e-5, **kwargs)
#     model = _create_vision_transformer('vit_small_patch16_36x1_224', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_small_patch16_18x2_224(pretrained=False, **kwargs):
#     """ ViT-Small w/ LayerScale + 18 x 2 (36 block parallel) config. Experimental, may remove.
#     Based on `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
#     Paper focuses on 24x2 + 48x1 for 'Small' width but those are extremely slow.
#     """
#     model_kwargs = dict(
#         patch_size=16, embed_dim=384, depth=18, num_heads=6, init_values=1e-5, block_fn=ParallelBlock, **kwargs)
#     model = _create_vision_transformer('vit_small_patch16_18x2_224', pretrained=pretrained, **model_kwargs)
#     return model


# @register_model
# def vit_base_patch16_18x2_224(pretrained=False, **kwargs):
#     """ ViT-Base w/ LayerScale + 18 x 2 (36 block parallel) config. Experimental, may remove.
#     Based on `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
#     """
#     model_kwargs = dict(
#         patch_size=16, embed_dim=768, depth=18, num_heads=12, init_values=1e-5, block_fn=ParallelBlock, **kwargs)
#     model = _create_vision_transformer('vit_base_patch16_18x2_224', pretrained=pretrained, **model_kwargs)
#     return model
