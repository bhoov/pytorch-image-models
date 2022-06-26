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
from .vision_transformer import VisionTransformer

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
}

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
        


class AlbertVisionTransformer(VisionTransformer):
    """ Vision Transformer with shared weights (ALBERT style)

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(self, *args, **kwargs):
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
        super().__init__(*args, **kwargs)
        new_blocks = RepeatedSequential(self.blocks[0], len(self.blocks))
        self.blocks = new_blocks
        
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