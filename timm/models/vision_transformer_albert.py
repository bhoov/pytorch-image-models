""" Vision Transformer (ViT) in PyTorch with shared weights
"""
import math
import logging
from functools import partial
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from .helpers import build_model_with_cfg, resolve_pretrained_cfg, named_apply, adapt_input_conv, checkpoint_seq
from .layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from .registry import register_model
from .vision_transformer import VisionTransformer, get_init_weights_vit, LayerScale
from .energy_transformer import HopfieldMLP, KQAlignedAttention, EnergyLayerNorm, KQEnergyBlock

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
    'avt_base_patch16_224': _cfg(url='', input_size=(3, 224, 224)),
    'tavt_noproj': _cfg(url='', input_size=(3, 224, 224)),
    'tavt_nobias': _cfg(url='', input_size=(3, 224, 224)),
    'tavt_relu': _cfg(url='', input_size=(3, 224, 224)),
    'tavt_symmlp': _cfg(url='', input_size=(3, 224, 224)),
    'tavt_parallel': _cfg(url='', input_size=(3, 224, 224)),
    'tavt_elnorm': _cfg(url='', input_size=(3, 224, 224)),
    'tavt_newatt': _cfg(url='', input_size=(3, 224, 224)),
    'tavt_base0': _cfg(url='', input_size=(3, 224, 224)),
    'tavt_base0_nobias': _cfg(url='', input_size=(3, 224, 224)),
    'tavt_base0_alpha': _cfg(url='', input_size=(3, 224, 224)),
    'tavt_base0_alpha_nobias': _cfg(url='', input_size=(3, 224, 224)),
}


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., proj_bias=True, use_proj=True):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.use_proj = use_proj
        if use_proj:
            self.proj = nn.Linear(dim, dim, bias=proj_bias)
            self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.use_proj:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mlp_bias=True, proj_bias=True, mlp_fn=Mlp, attn_fn=Attention, use_proj=True, alpha=1.):
        super().__init__()
        self.alpha = alpha
        self.norm1 = norm_layer(dim)
        self.attn = attn_fn(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, proj_bias=proj_bias, use_proj=use_proj)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_fn(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop, bias=mlp_bias)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.alpha * self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.alpha * self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

# Cell
class ParallelBlock(Block):
    def __init__(self, *args, alpha=1., **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, "norm2"):
            delattr(self, "norm2")
        self.alpha = alpha
            
    def forward(self, x):
        g = self.norm1(x)
        attn_sig = self.drop_path1(self.ls1(self.attn(g)))
        mlp_sig = self.drop_path2(self.ls2(self.mlp(g)))
        x = x + self.alpha * (attn_sig + mlp_sig)
        return x

class RepeatedSequential(nn.Module):
    """A plug-in for the nn.Sequential module that simulates a sequential operation, but repeats a single operation N times (requires a homomorphic operation)"""
    def __init__(self, block, depth:int):
        super().__init__()
        self.block = block
        self.depth = depth
        
    def forward(self, x, depth:int=None):
        """Repeat the block operation `depth` times. 
        
        If a new depth is provided, override the default
        """
        N = depth if depth is not None else self.depth
        for _ in range(N):
            x = self.block(x)
            
        return x
    
    def __len__(self):
        return self.depth

class AlbertVisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, mlp_bias=True, proj_bias=True, init_values=None,
            class_token=True, fc_norm=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='',
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, mlp_fn=Mlp, attn_fn=Attention, alpha=1., use_proj=True):
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
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

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
        self.pos_drop = nn.Dropout(p=drop_rate)

        block = block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, mlp_bias=mlp_bias, proj_bias=proj_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer, act_layer=act_layer, mlp_fn=mlp_fn, attn_fn=attn_fn, alpha=alpha, use_proj=use_proj)
        self.blocks = RepeatedSequential(block, depth)
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
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

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

            
def _create_albert_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    pretrained_cfg = resolve_pretrained_cfg(variant, kwargs=kwargs)
    model = build_model_with_cfg(
        AlbertVisionTransformer, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        # pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model
        
@register_model
def avt_base_patch16_224(pretrained=False, **kwargs):
    """ AVT-Base, new for this paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_albert_vision_transformer('avt_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def tavt_nobias(pretrained=False, **kwargs):
    """ No biases in any of the major linear operations. LayerNorm stuff can still have biases"""
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, mlp_bias=False, proj_bias=False, **kwargs)
    model = _create_albert_vision_transformer('tavt_nobias', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def tavt_noproj(pretrained=False, **kwargs):
    """ No biases in any of the major linear operations. LayerNorm stuff can still have biases"""
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, use_proj=False, **kwargs)
    model = _create_albert_vision_transformer('tavt_noproj', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def tavt_relu(pretrained=False, **kwargs):
    """ AVT-Base, new for this paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=True, act_layer=nn.ReLU, **kwargs)
    model = _create_albert_vision_transformer('tavt_relu', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def tavt_symmlp(pretrained=False, **kwargs):
    """ AVT-Base, new for this paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=True, mlp_fn=HopfieldMLP, **kwargs)
    model = _create_albert_vision_transformer('tavt_symmlp', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def tavt_parallel(pretrained=False, **kwargs):
    """ AVT-Base, new for this paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=True, block_fn=ParallelBlock, **kwargs)
    model = _create_albert_vision_transformer('tavt_parallel', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def tavt_elnorm(pretrained=False, **kwargs):
    """ Use the EnergyLayerNorm
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=True, norm_layer=EnergyLayerNorm, **kwargs)
    model = _create_albert_vision_transformer('tavt_elnorm', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def tavt_newatt(pretrained=False, **kwargs):
    """ Use our new Attention only
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=True, attn_fn=KQAlignedAttention, **kwargs)
    model = _create_albert_vision_transformer('tavt_newatt', pretrained=pretrained, **model_kwargs)
    return model

## BASE0 models, which all have relu, elnorm and are parallel
@register_model
def tavt_base0(pretrained=False, **kwargs):
    """ Use our new Attention with symmetric MLP, parallel blocks, the whole shebang without specifying step size
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=True, act_layer=nn.ReLU, norm_layer=EnergyLayerNorm, mlp_fn=HopfieldMLP, attn_fn=KQAlignedAttention, block_fn=ParallelBlock, **kwargs)
    model = _create_albert_vision_transformer('tavt_base0', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def tavt_base0_nobias(pretrained=False, **kwargs):
    """ Use our new Attention with symmetric MLP, parallel blocks, the whole shebang without specifying step size
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, mlp_bias=False, proj_bias=False, act_layer=nn.ReLU, norm_layer=EnergyLayerNorm, mlp_fn=HopfieldMLP, attn_fn=KQAlignedAttention, block_fn=ParallelBlock, **kwargs)
    model = _create_albert_vision_transformer('tavt_base0_nobias', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def tavt_base0_alpha(pretrained=False, **kwargs):
    """ Use our new Attention with symmetric MLP but this time specifying step size.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=True, act_layer=nn.ReLU, norm_layer=EnergyLayerNorm, mlp_fn=HopfieldMLP, attn_fn=KQAlignedAttention, alpha=0.1, block_fn=ParallelBlock, **kwargs)
    model = _create_albert_vision_transformer('tavt_base0_alpha', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def tavt_base0_alpha_nobias(pretrained=False, **kwargs):
    """ Use our new Attention with symmetric MLP, parallel blocks, the whole shebang without specifying step size
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, mlp_bias=False, proj_bias=False, act_layer=nn.ReLU, norm_layer=EnergyLayerNorm, mlp_fn=HopfieldMLP, attn_fn=KQAlignedAttention, block_fn=ParallelBlock, alpha=0.1, **kwargs)
    model = _create_albert_vision_transformer('tavt_base0_nobias', pretrained=pretrained, **model_kwargs)
    return model