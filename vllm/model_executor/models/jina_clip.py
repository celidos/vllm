# coding=utf-8
#
# Code mainly copied from:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py
# and adjusted for Jina CLIP

import os
import re
from copy import deepcopy
import base64
import importlib.util
import collections
import warnings
from functools import partial
from io import BytesIO
from typing import Any, List, Dict, Optional, Tuple, Union

from math import pi

from einops import rearrange, repeat

import numpy as np
import requests
import torch
import torch
import torch.nn.functional as f
import torch.utils.checkpoint
from PIL import Image
from torch import nn
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    BatchEncoding,
    BatchFeature,
    PreTrainedModel,
    PretrainedConfig,
    logging,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.models.clip.modeling_clip import (
    CLIPOutput,
    CLIPTextModelOutput,
    CLIPVisionModelOutput,
    clip_loss,
)



from vllm.config import VllmConfig

logger = logging.get_logger(__name__)

# CONFIGURATION_CLIP --------------------------------------------------

class JinaCLIPTextConfig(PretrainedConfig):  # MEX: something to review
    model_type = 'jina_clip_text'

    def __init__(
        self,
        embed_dim: int = 768,
        hf_model_name_or_path: str = 'jinaai/jina-bert-flash-implementation',
        hf_model_config_kwargs: Optional[Dict[str, Any]] = None,
        default_instruction_task: Optional[str] = None,
        default_lora_task: Optional[str] = None,
        pooler_type: Optional[str] = None,
        proj_type: Optional[str] = None,
        proj_bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.hf_model_name_or_path = hf_model_name_or_path
        self.hf_model_config_kwargs = hf_model_config_kwargs or {}
        self.default_instruction_task = default_instruction_task
        self.default_lora_task = default_lora_task
        self.pooler_type = pooler_type
        self.proj_type = proj_type
        self.proj_bias = proj_bias

    @classmethod
    def from_pretrained(    # ME: should I keep it?
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> 'PretrainedConfig':
        cls._set_token_in_kwargs(kwargs)

        configdict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )
        # get the text config dict if we are loading from JinaCLIPConfig
        if configdict.get('model_type') == 'jina_clip':
            configdict = configdict['text_config']
        if (
            'model_type' in configdict
            and hasattr(cls, 'model_type')
            and configdict['model_type'] != cls.model_type
        ):
            logger.warning(
                f'You are using a model of type {configdict["model_type"]} to '
                f'instantiate a model of type {cls.model_type}. This is not supported '
                'for all configurations of models and can yield errors.'
            )
        return cls.from_dict(configdict, **kwargs)


class JinaCLIPVisionConfig(PretrainedConfig):  # MEX: something to review
    model_type = 'jina_clip_vision'

    def __init__(
        self,
        embed_dim: int = 768,
        width: int = 768,
        image_size: int = 224,
        patch_size: int = 16,
        layers: int = 12,
        head_width: int = 64,
        mlp_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        patch_dropout: float = 0.0,
        qkv_bias: bool = True,
        fused_layer_norm: bool = False,
        x_attention: bool = False,
        post_norm: bool = False,
        rope_embeddings: bool = False,
        pt_hw_seq_len: int = 16,
        intp_freq: bool = False,
        naive_swiglu: bool = False,
        subln: bool = False,
        drop_path_rate: float = 0.0,
        proj_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.layers = layers
        self.embed_dim = embed_dim
        self.width = width
        self.head_width = head_width
        self.mlp_ratio = mlp_ratio
        self.image_size = image_size
        self.patch_size = patch_size
        self.ls_init_value = ls_init_value
        self.patch_dropout = patch_dropout
        self.qkv_bias = qkv_bias
        self.fused_layer_norm = fused_layer_norm
        self.x_attention = x_attention
        self.post_norm = post_norm
        self.rope_embeddings = rope_embeddings
        self.pt_hw_seq_len = pt_hw_seq_len
        self.intp_freq = intp_freq
        self.naive_swiglu = naive_swiglu
        self.subln = subln
        self.drop_path_rate = drop_path_rate
        self.proj_type = proj_type

    @classmethod
    def from_pretrained(  # ME: should i keep it?
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> 'PretrainedConfig':
        cls._set_token_in_kwargs(kwargs)

        configdict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs
        )
        # get the vision config dict if we are loading from JinaCLIPConfig
        if configdict.get('model_type') == 'jina_clip':
            configdict = configdict['vision_config']
        if (
            'model_type' in configdict
            and hasattr(cls, 'model_type')
            and configdict['model_type'] != cls.model_type
        ):
            logger.warning(
                f'You are using a model of type {configdict["model_type"]} to '
                f'instantiate a model of type {cls.model_type}. This is not supported '
                'for all configurations of models and can yield errors.'
            )
        return cls.from_dict(configdict, **kwargs)


class JinaCLIPConfig(PretrainedConfig):  # MEX: something to review
    model_type = 'jina_clip'
    is_composition = True

    def __init__(
        self,
        text_config: Optional[Dict] = None,
        vision_config: Optional[Dict] = None,
        add_projections: bool = False,
        projection_dim: int = 768,
        logit_scale_init_value: float = 2.6592,
        use_text_flash_attn: Optional[bool] = None,
        use_vision_xformers: Optional[bool] = None,
        matryoshka_dimensions: Optional[List[int]] = None,
        truncate_dim: Optional[int] = None,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs,
    ):
        # If `_config_dict` exist, we use them for the backward compatibility.
        # We pop out these 2 attributes before calling `super().__init__` to avoid
        # them being saved (which causes a lot of confusion!).

        text_config_dict: Optional[Dict] = kwargs.pop('text_config_dict', None)
        vision_config_dict: Optional[Dict] = kwargs.pop('vision_config_dict', None)
        self.use_text_flash_attn = use_text_flash_attn
        self.use_vision_xformers = use_vision_xformers
        self.matryoshka_dimensions = matryoshka_dimensions
        self.truncate_dim = truncate_dim

        super().__init__(**kwargs)

        if text_config_dict is not None:
            if text_config is None:
                text_config = {}

            # This is the complete result when using `text_config_dict`.
            _text_config_dict = JinaCLIPTextConfig(**text_config_dict).to_dict()

            # Give a warning if the values exist in both `_text_config_dict` and
            # `text_config` but being different.
            for key, value in _text_config_dict.items():
                if (
                    key in text_config
                    and value != text_config[key]
                    and key not in ['transformers_version']
                ):
                    # If specified in `text_config_dict`
                    if key in text_config_dict:
                        message = (
                            f'`{key}` is found in both `text_config_dict` and '
                            f'`text_config` but with different values. '
                            f'The value `text_config_dict["{key}"]` will be used '
                            f'instead.'
                        )
                    # If inferred from default argument values (
                    # just to be super careful)
                    else:
                        message = (
                            f'`text_config_dict` is provided which will be used to '
                            f'initialize `JinaCLIPTextConfig`. The '
                            f'value `text_config["{key}"]` will be overriden.'
                        )
                    logger.info(message)

            # Update all values in `text_config` with the ones in `_text_config_dict`.
            text_config.update(_text_config_dict)

        if vision_config_dict is not None:
            if vision_config is None:
                vision_config = {}

            # This is the complete result when using `vision_config_dict`.
            _vision_config_dict = JinaCLIPVisionConfig(**vision_config_dict).to_dict()
            # convert keys to string instead of integer
            if 'id2label' in _vision_config_dict:
                _vision_config_dict['id2label'] = {
                    str(key): value
                    for key, value in _vision_config_dict['id2label'].items()
                }

            # Give a warning if the values exist in both `_vision_config_dict`
            # and `vision_config` but being different.
            for key, value in _vision_config_dict.items():
                if (
                    key in vision_config
                    and value != vision_config[key]
                    and key not in ['transformers_version']
                ):
                    # If specified in `vision_config_dict`
                    if key in vision_config_dict:
                        message = (
                            f'`{key}` is found in both `vision_config_dict` and '
                            f'`vision_config` but with different '
                            f'values. The value `vision_config_dict["{key}"]` will '
                            f'be used instead.'
                        )
                    # If inferred from default argument values
                    # (just to be super careful)
                    else:
                        message = (
                            f'`vision_config_dict` is provided which will be used to '
                            f'initialize `JinaCLIPVisionConfig`. '
                            f'The value `vision_config["{key}"]` will be overriden.'
                        )
                    logger.info(message)

            # Update all values in `vision_config` with the ones in
            # `_vision_config_dict`.
            vision_config.update(_vision_config_dict)

        if text_config is None:
            text_config = {}
            logger.info(
                '`text_config` is `None`. Initializing the `JinaCLIPTextConfig` with '
                'default values.'
            )

        if vision_config is None:
            vision_config = {}
            logger.info(
                '`vision_config` is `None`. initializing the `JinaCLIPVisionConfig` '
                'with default values.'
            )

        self.text_config = JinaCLIPTextConfig(**text_config)
        self.vision_config = JinaCLIPVisionConfig(**vision_config)

        self.add_projections = add_projections
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

        if not self.add_projections:
            if self.text_config.embed_dim != self.vision_config.embed_dim:
                raise ValueError(
                    'When projections are disabled (`add_projections=False`), text '
                    'and vision towers need to have the same embedding dimensionality. '
                    f'Currently text embedding dim is {self.text_config.embed_dim} != '
                    f'{self.vision_config.embed_dim} of the vision tower. '
                    'Either set the same output dim for both towers, or enable '
                    'projections with `add_projections=True`.'
                )

        if (
            torch_dtype
            and hasattr(torch, torch_dtype)
            and type(getattr(torch, torch_dtype)) is torch.dtype
        ):
            self.torch_dtype = getattr(torch, torch_dtype)
        else:
            self.torch_dtype = torch_dtype

        use_text_flash_attn = (
            self.use_text_flash_attn if self.use_text_flash_attn is not None
            else self.text_config.hf_model_config_kwargs.get('use_flash_attn', False)
        )
        if not use_text_flash_attn or not torch.cuda.is_available():
            self.torch_dtype = torch.float32

    @classmethod
    def from_text_vision_configs(
        cls,
        text_config: JinaCLIPTextConfig,
        vision_config: JinaCLIPVisionConfig,
        **kwargs,
    ):
        return cls(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            projection_dim=text_config.projection_dim,
            **kwargs,
        )

    def to_dict(self):
        output = deepcopy(self.__dict__)
        output['text_config'] = self.text_config.to_dict()
        output['vision_config'] = self.vision_config.to_dict()
        output['model_type'] = self.__class__.model_type
        return output

# end of CONFIGURATION CLIP -------------------------------------------

# ROPE EMBEDDINGS -----------------------------------------------------

def broadcast(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for='lang',
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs_h = torch.einsum('..., f -> ... f', t, freqs)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r=2)

        freqs_w = torch.einsum('..., f -> ... f', t, freqs)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r=2)

        freqs = broadcast((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)

        self.register_buffer('freqs_cos', freqs.cos(), persistent=False)
        self.register_buffer('freqs_sin', freqs.sin(), persistent=False)

    def forward(self, t, start_index=0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], (
            f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in '
            f'all the positions {rot_dim}'
        )
        t_left, t, t_right = (
            t[..., :start_index],
            t[..., start_index:end_index],
            t[..., end_index:],
        )
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)

        return torch.cat((t_left, t, t_right), dim=-1)


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for='lang',
        theta=10000,
        max_freq=10,
        num_freqs=1,
        patch_dropout=0.0,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1.0 / (
                theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum('..., f -> ... f', t, freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)
        freqs = broadcast((freqs[:, None, :], freqs[None, :, :]), dim=-1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.patch_dropout = patch_dropout

        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)

    def forward(self, t, patch_indices_keep=None):
        if patch_indices_keep is not None:
            batch = t.size()[0]
            batch_indices = torch.arange(batch)
            batch_indices = batch_indices[..., None]

            freqs_cos = repeat(
                self.freqs_cos, 'i j -> n i m j', n=t.shape[0], m=t.shape[1]
            )
            freqs_sin = repeat(
                self.freqs_sin, 'i j -> n i m j', n=t.shape[0], m=t.shape[1]
            )

            freqs_cos = freqs_cos[batch_indices, patch_indices_keep]
            freqs_cos = rearrange(freqs_cos, 'n i m j -> n m i j')
            freqs_sin = freqs_sin[batch_indices, patch_indices_keep]
            freqs_sin = rearrange(freqs_sin, 'n i m j -> n m i j')

            return t * freqs_cos + rotate_half(t) * freqs_sin

        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin    

# end of ROPE EMBEDDINGS ----------------------------------------------

# TRANSFORM -----------------------------------------------------------

import random
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torchvision.transforms.functional as f
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    Compose,
    Grayscale,
    InterpolationMode,
    Normalize,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

OPENAI_DATASET_MEAN = tuple(OPENAI_CLIP_MEAN)
OPENAI_DATASET_STD = tuple(OPENAI_CLIP_STD)


def _setup_size(size, error_msg):
    if isinstance(size, int):
        return size, size
    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]
    if len(size) != 2:
        raise ValueError(error_msg)
    return size


def _center_crop_or_pad(
    img: torch.Tensor,
    output_size: Union[int, Tuple[int, ...], List[int]],
    fill: Union[int, Tuple[int]] = 0,
) -> torch.Tensor:
    """
    Center crops and/or pads the given image. If the image is torch Tensor, it is
    expected to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions. If image size is smaller than output size along any edge, image is
    padded with 0 and then center cropped.
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    _, image_height, image_width = f.get_dimensions(img)
    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = f.pad(img, padding_ltrb, fill=fill)
        _, image_height, image_width = f.get_dimensions(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return f.crop(img, crop_top, crop_left, crop_height, crop_width)


class _CenterCropOrPad(torch.nn.Module):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions. If image size is smaller than output size along any edge, image is
    padded with 0 and then center cropped.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as
            (size[0], size[0]).
    """

    def __init__(self, size, fill=0):
        super().__init__()
        self.size = _setup_size(
            size, error_msg='Please provide only two dimensions (h, w) for size.'
        )
        self.fill = fill

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.
        Returns:
            PIL Image or Tensor: Cropped image.
        """
        return _center_crop_or_pad(img, self.size, fill=self.fill)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(size={self.size})'


def _convert_to_rgb(image):
    return image.convert('RGB')


class _ResizeKeepRatio:
    """Resize while keeping ratio. Copied from timm"""

    def __init__(
        self,
        size,
        longest=0.0,
        interpolation=InterpolationMode.BICUBIC,
        random_scale_prob=0.0,
        random_scale_range=(0.85, 1.05),
        random_aspect_prob=0.0,
        random_aspect_range=(0.9, 1.11),
    ):
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        self.interpolation = interpolation
        self.longest = float(longest)  # [0, 1] where 0 == shortest edge, 1 == longest
        self.random_scale_prob = random_scale_prob
        self.random_scale_range = random_scale_range
        self.random_aspect_prob = random_aspect_prob
        self.random_aspect_range = random_aspect_range

    @staticmethod
    def get_params(
        img,
        target_size,
        longest,
        random_scale_prob=0.0,
        random_scale_range=(0.85, 1.05),
        random_aspect_prob=0.0,
        random_aspect_range=(0.9, 1.11),
    ):
        """Get parameters"""
        source_size = img.size[::-1]  # h, w
        h, w = source_size
        target_h, target_w = target_size
        ratio_h = h / target_h
        ratio_w = w / target_w
        ratio = max(ratio_h, ratio_w) * longest + min(ratio_h, ratio_w) * (
            1.0 - longest
        )
        if random_scale_prob > 0 and random.random() < random_scale_prob:
            ratio_factor = random.uniform(random_scale_range[0], random_scale_range[1])
            ratio_factor = (ratio_factor, ratio_factor)
        else:
            ratio_factor = (1.0, 1.0)
        if random_aspect_prob > 0 and random.random() < random_aspect_prob:
            aspect_factor = random.uniform(
                random_aspect_range[0], random_aspect_range[1]
            )
            ratio_factor = (
                ratio_factor[0] / aspect_factor,
                ratio_factor[1] * aspect_factor,
            )
        return [
            round(x * factor / ratio) for x, factor in zip(source_size, ratio_factor)
        ]

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.
        Returns:
            PIL Image: Resized, padded to at least target size, possibly
            cropped to exactly target size
        """
        size = self.get_params(
            img,
            self.size,
            self.longest,
            self.random_scale_prob,
            self.random_scale_range,
            self.random_aspect_prob,
            self.random_aspect_range,
        )
        img = f.resize(img, size, self.interpolation)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += f', interpolation={self.interpolation})'
        format_string += f', longest={self.longest:.3f})'
        return format_string


class _ColorJitter(object):
    """Apply color jitter to the PIL image with a specified probability"""

    def __init__(self, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0, p=0.8):
        assert 0.0 <= p <= 1.0
        self.p = p
        self.transf = ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


class _GrayScale(object):
    """Apply gray scale to the PIL image with a specified probability"""

    def __init__(self, p=0.2):
        assert 0.0 <= p <= 1.0
        self.p = p
        self.transf = Grayscale(num_output_channels=3)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


@dataclass
class AugmentationCfg:
    scale: Tuple[float, float] = (0.9, 1.0)
    ratio: Optional[Tuple[float, float]] = None
    color_jitter: Optional[
        Union[float, Tuple[float, float, float], Tuple[float, float, float, float]]
    ] = None
    re_prob: Optional[float] = None
    re_count: Optional[int] = None
    use_timm: bool = False
    color_jitter_prob: float = None
    gray_scale_prob: float = None


def image_transform(
    image_size: Union[int, Tuple[int, int]],
    is_train: bool,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    resize_mode: Optional[str] = None,
    interpolation: Optional[str] = None,
    fill_color: int = 0,
    aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    interpolation = interpolation or 'bicubic'
    assert interpolation in ['bicubic', 'bilinear', 'random']
    # NOTE random is ignored for interpolation_mode, so defaults to BICUBIC for
    # inference if set
    interpolation_mode = (
        InterpolationMode.BILINEAR
        if interpolation == 'bilinear'
        else InterpolationMode.BICUBIC
    )

    resize_mode = resize_mode or 'shortest'
    assert resize_mode in ('shortest', 'longest', 'squash')

    if isinstance(aug_cfg, dict):
        aug_cfg = AugmentationCfg(**aug_cfg)
    else:
        aug_cfg = aug_cfg or AugmentationCfg()

    normalize = Normalize(mean=mean, std=std)

    if is_train:
        aug_cfg_dict = {k: v for k, v in asdict(aug_cfg).items() if v is not None}
        use_timm = aug_cfg_dict.pop('use_timm', False)
        if use_timm:
            from timm.data import create_transform  # timm can still be optional

            if isinstance(image_size, (tuple, list)):
                assert len(image_size) >= 2
                input_size = (3,) + image_size[-2:]
            else:
                input_size = (3, image_size, image_size)

            aug_cfg_dict.setdefault('color_jitter', None)  # disable by default
            # drop extra non-timm items
            aug_cfg_dict.pop('color_jitter_prob', None)
            aug_cfg_dict.pop('gray_scale_prob', None)

            train_transform = create_transform(
                input_size=input_size,
                is_training=True,
                hflip=0.0,
                mean=mean,
                std=std,
                re_mode='pixel',
                interpolation=interpolation,
                **aug_cfg_dict,
            )
        else:
            train_transform = [
                RandomResizedCrop(
                    image_size,
                    scale=aug_cfg_dict.pop('scale'),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                _convert_to_rgb,
            ]
            if aug_cfg.color_jitter_prob:
                assert (
                    aug_cfg.color_jitter is not None and len(aug_cfg.color_jitter) == 4
                )
                train_transform.extend(
                    [_ColorJitter(*aug_cfg.color_jitter, p=aug_cfg.color_jitter_prob)]
                )
            if aug_cfg.gray_scale_prob:
                train_transform.extend([_GrayScale(aug_cfg.gray_scale_prob)])
            train_transform.extend(
                [
                    ToTensor(),
                    normalize,
                ]
            )
            train_transform = Compose(train_transform)
            if aug_cfg_dict:
                warnings.warn(
                    f'Unused augmentation cfg items, specify `use_timm` to use '
                    f'({list(aug_cfg_dict.keys())}).'
                )
        return train_transform
    else:
        if resize_mode == 'longest':
            transforms = [
                _ResizeKeepRatio(
                    image_size, interpolation=interpolation_mode, longest=1
                ),
                _CenterCropOrPad(image_size, fill=fill_color),
            ]
        elif resize_mode == 'squash':
            if isinstance(image_size, int):
                image_size = (image_size, image_size)
            transforms = [
                Resize(image_size, interpolation=interpolation_mode),
            ]
        else:
            assert resize_mode == 'shortest'
            if not isinstance(image_size, (tuple, list)):
                image_size = (image_size, image_size)
            if image_size[0] == image_size[1]:
                # simple case, use torchvision built-in Resize w/ shortest edge mode
                # (scalar size arg)
                transforms = [Resize(image_size[0], interpolation=interpolation_mode)]
            else:
                # resize shortest edge to matching target dim for non-square target
                transforms = [_ResizeKeepRatio(image_size)]
            transforms += [CenterCrop(image_size)]

        transforms.extend(
            [
                _convert_to_rgb,
                ToTensor(),
                normalize,
            ]
        )
        return Compose(transforms)    

# end of TRANFORM -----------------------------------------------------

# HF MODEL ------------------------------------------------------------
   
_HF_ARCH_DICT = {
    # https://huggingface.co/docs/transformers/model_doc/roberta#roberta
    'roberta': {
        'config_names': {
            'context_length': 'max_position_embeddings',
            'vocab_size': 'vocab_size',
            'width': 'hidden_size',
            'heads': 'num_attention_heads',
            'layers': 'num_hidden_layers',
            'layer_attr': 'layer',
            'token_embeddings_attr': 'embeddings',
        },
        'pooler': 'mean_pooler',
    },
    # https://huggingface.co/docs/transformers/model_doc/xlm-roberta#transformers.XLMRobertaConfig
    'xlm-roberta': {
        'config_names': {
            'context_length': 'max_position_embeddings',
            'vocab_size': 'vocab_size',
            'width': 'hidden_size',
            'heads': 'num_attention_heads',
            'layers': 'num_hidden_layers',
            'layer_attr': 'layer',
            'token_embeddings_attr': 'embeddings',
        },
        'pooler': 'mean_pooler',
    },
    # https://huggingface.co/docs/transformers/model_doc/bert
    'bert': {
        'config_names': {
            'context_length': 'max_position_embeddings',
            'vocab_size': 'vocab_size',
            'width': 'hidden_size',
            'heads': 'num_attention_heads',
            'layers': 'num_hidden_layers',
        },
        'pooler': 'cls_pooler',
    },
}

_POOLERS = {}


def _camel2snake(s):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()


def register_pooler(cls):
    """Decorator registering pooler class"""
    _POOLERS[_camel2snake(cls.__name__)] = cls
    return cls


@register_pooler
class MeanPooler(nn.Module):
    @staticmethod
    def forward(x: BaseModelOutput, attention_mask: torch.Tensor):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)


@register_pooler
class MaxPooler(nn.Module):
    @staticmethod
    def forward(x: BaseModelOutput, attention_mask: torch.Tensor):
        masked_output = x.last_hidden_state.masked_fill(
            attention_mask.unsqueeze(-1), -torch.inf
        )
        return masked_output.max(1).values


@register_pooler
class ClsPooler(nn.Module):
    def __init__(self, use_pooler_output: bool = True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output

    def forward(self, x: BaseModelOutput, _: torch.Tensor):
        if (
            self.use_pooler_output
            and isinstance(
                x,
                (
                    BaseModelOutputWithPooling,
                    BaseModelOutputWithPoolingAndCrossAttentions,
                ),
            )
            and (x.pooler_output is not None)
        ):
            return x.pooler_output
        return x.last_hidden_state[:, self.cls_token_position, :]


class HFTextEncoder(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
        self,
        model_name_or_path: str,
        output_dim: int,
        config: PretrainedConfig = None,
        pooler_type: str = None,
        proj_type: str = None,
        proj_bias: bool = False,
        pretrained: bool = True,
        output_tokens: bool = False,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        code_revision: Optional[str] = None,
        default_instruction_task: Optional[str] = None,
        default_lora_task: Optional[str] = None,
        model_config_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.output_dim = output_dim

        model_config_kwargs = model_config_kwargs or {}

        if config is None:
            if pretrained:
                self.transformer = AutoModel.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    add_pooling_layer=False,
                    code_revision=code_revision,
                    **model_config_kwargs,
                )
                self.config = self.transformer.config
            else:
                self.config = AutoConfig.from_pretrained(
                    model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    code_revision=code_revision,
                )
                self.config.update(model_config_kwargs)
                self.transformer = AutoModel.from_config(
                    self.config,
                    trust_remote_code=trust_remote_code,
                    add_pooling_layer=False,
                    code_revision=code_revision,
                )
            if (
                hasattr(self.config, 'is_encoder_decoder')
                and self.config.is_encoder_decoder
            ):
                self.transformer = self.transformer.encoder

        else:
            self.config = config
            self.config.update(model_config_kwargs)
            self.transformer = AutoModel.from_config(
                self.config,
                trust_remote_code=trust_remote_code,
                revision=revision,
                code_revision=code_revision,
            )
        self.vocab_size = getattr(self.config, 'vocab_size', 0)
        self.context_length = getattr(self.config, 'max_position_embeddings', 0)

        pooler_type = pooler_type or _HF_ARCH_DICT[self.config.model_type]['pooler']
        self.pooler = _POOLERS[pooler_type]()

        d_model = getattr(
            self.config, _HF_ARCH_DICT[self.config.model_type]['config_names']['width']
        )
        if (d_model == output_dim) and (proj_type is None):  # do we always need a proj?
            self.proj = nn.Identity()
        elif (d_model != output_dim) or proj_type == 'linear':
            self.proj = nn.Linear(d_model, output_dim, bias=proj_bias)
        elif proj_type == 'mlp':
            hidden_size = (d_model + output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias=proj_bias),
                nn.GELU(),
                nn.Linear(hidden_size, output_dim, bias=proj_bias),
            )

        self._task_instructions = {}
        self._lora_adaptation_map = {}
        self._supports_task_instructions = False
        self._supports_lora = False
        if (
            hasattr(self.transformer, '_adaptation_map')
            and len(self.transformer._adaptation_map) > 0
        ):
            self._lora_adaptation_map = self.transformer._adaptation_map
            self._supports_lora = True
        if (
            hasattr(self.transformer, '_task_instructions')
            and len(self.transformer._task_instructions) > 0
        ):
            self._task_instructions = self.transformer._task_instructions
            self._supports_task_instructions = True

        self._default_instruction_task = None
        self._default_lora_task = None
        self._default_instruction = None
        self._default_loraid = None

        if default_instruction_task is not None:
            self._default_instruction_task = default_instruction_task
            self._default_instruction = self.get_instruction_from_task(
                default_instruction_task
            )
        if default_lora_task is not None:
            self._default_lora_task = default_lora_task
            self._default_loraid = self.get_loraid_from_task(default_lora_task)

    @property
    def supports_task_instructions(self) -> bool:
        return self._supports_task_instructions

    @property
    def supports_lora(self) -> bool:
        return self._supports_lora

    @property
    def task_instructions(self) -> Dict[str, str]:
        return self._task_instructions

    @property
    def lora_adaptation_map(self) -> Dict[str, int]:
        return self._lora_adaptation_map

    @property
    def default_instruction(self) -> Optional[str]:
        return self._default_instruction

    @property
    def default_loraid(self) -> Optional[int]:
        return self._default_loraid

    def get_instruction_from_task(self, task: Optional[str]) -> Optional[str]:
        if self._supports_task_instructions:
            if task is None:
                return self._default_instruction
            if task not in self._task_instructions:
                raise ValueError(
                    f'Unsupported task \'{task}\'. Choose one of the following: '
                    f'{", ".join(self._task_instructions)} or set to None to disable '
                    f'task instructions completely'
                )
            return self._task_instructions[task]
        else:
            if task is not None:
                warnings.warn(
                    'Model does not support task instructions, ignoring instruction '
                    f"task '{task}'"
                )
        return None

    def get_loraid_from_task(self, task: Optional[str]) -> Optional[int]:
        if self._supports_lora:
            if task is None:
                return self._default_loraid
            if task not in self._lora_adaptation_map:
                raise ValueError(
                    f'Unsupported task \'{task}\'. Choose one of the following: '
                    f'{", ".join(self._task_instructions)} or set to None to disable '
                    f'the LoRA adapters completely'
                )
            return self._lora_adaptation_map[task]
        else:
            if task is not None:
                warnings.warn(
                    f"Model does not support LoRA adapters, ignoring LoRA task '{task}'"
                )
        return None

    @staticmethod
    def get_adapter_mask_from_loraid(
        batch_size: int, loraid: int, device: Union[str, torch.device]
    ):
        return torch.full((batch_size,), loraid, dtype=torch.int32, device=device)

    @torch.jit.ignore
    def set_grad_checkpointing(self, _=True):
        self.transformer.gradient_checkpointing_enable()

    def init_parameters(self):
        pass

    def forward(self, x: torch.Tensor, adapter_mask: Optional[torch.Tensor] = None):
        if adapter_mask is None:
            default_loraid = self.default_loraid
            if default_loraid is not None:
                adapter_mask = self.get_adapter_mask_from_loraid(
                    x.shape[0], default_loraid, x.device
                )
        else:
            if not self.supports_lora:
                warnings.warn(
                    'Model does not support LoRA adapters, setting adapter_mask to None'
                )
                adapter_mask = None

        attention_mask = (x != self.config.pad_token_id).long()
        lora_kwargs = {}
        if adapter_mask is not None:
            lora_kwargs['adapter_mask'] = adapter_mask

        out = self.transformer(
            input_ids=x, attention_mask=attention_mask, **lora_kwargs
        )
        pooled_out = self.pooler(out, attention_mask)
        projected = self.proj(pooled_out)
        seqlen = out.last_hidden_state.shape[1]
        tokens = (
            out.last_hidden_state[
                :, torch.arange(seqlen) != self.pooler.cls_token_position, :
            ]
            if isinstance(self.pooler, ClsPooler)
            else out.last_hidden_state
        )
        if self.output_tokens:
            return projected, tokens
        return projected

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        if not unlocked_layers:
            for n, p in self.transformer.named_parameters():
                p.requires_grad = (
                    (not freeze_layer_norm) if 'LayerNorm' in n.split('.') else False
                )
            return

        encoder = (
            self.transformer.encoder
            if hasattr(self.transformer, 'encoder')
            else self.transformer
        )
        layer_list = getattr(
            encoder, _HF_ARCH_DICT[self.config.model_type]['config_names']['layer_attr']
        )
        print(f'Unlocking {unlocked_layers}/{len(layer_list) + 1} layers of hf model')
        embeddings = getattr(
            self.transformer,
            _HF_ARCH_DICT[self.config.model_type]['config_names'][
                'token_embeddings_attr'
            ],
        )
        modules = [embeddings, *layer_list][:-unlocked_layers]
        # freeze layers
        for module in modules:
            for n, p in module.named_parameters():
                p.requires_grad = (
                    (not freeze_layer_norm) if 'LayerNorm' in n.split('.') else False
                )

# end of HF MODEL -----------------------------------------------------

# EVA MODEL -----------------------------------------------------------

import math
import os
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as f


def timm_drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(x, x)


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)

# if os.getenv('ENV_TYPE') == 'deepspeed':
#     try:
#         from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
#     except ImportError or ModuleNotFoundError:
#         from torch.utils.checkpoint import checkpoint
# else:
#     from torch.utils.checkpoint import checkpoint

from torch.utils.checkpoint import checkpoint

try:
    import xformers.ops as xops
except ImportError:
    xops = None


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x, patch_indices_keep


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return timm_drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        drop=0.0,
        subln=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.ffn_ln(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLU(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.SiLU,
        drop=0.0,
        norm_layer=nn.LayerNorm,
        subln=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        window_size=None,
        attn_head_dim=None,
        xattn=False,
        rope=None,
        subln=False,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.subln = subln
        if self.subln:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=False)
        else:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (
                2 * window_size[1] - 1
            ) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads)
            )  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = (
                coords_flatten[:, :, None] - coords_flatten[:, None, :]
            )  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(
                1, 2, 0
            ).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = torch.zeros(
                size=(window_size[0] * window_size[1] + 1,) * 2,
                dtype=relative_coords.dtype,
            )
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer('relative_position_index', relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.inner_attn_ln = norm_layer(all_head_dim) if subln else nn.Identity()
        # self.proj = nn.Linear(all_head_dim, all_head_dim)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.xattn = xattn
        self.xattn_drop = attn_drop

        self.rope = rope

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        b, n, _ = x.shape
        if self.subln:
            q = f.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
            k = f.linear(input=x, weight=self.k_proj.weight, bias=None)
            v = f.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)

            q = q.reshape(b, n, self.num_heads, -1).permute(
                0, 2, 1, 3
            )  # B, num_heads, N, C
            k = k.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            v = v.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
        else:
            qkv_bias = None
            if self.q_bias is not None:
                qkv_bias = torch.cat(
                    (
                        self.q_bias,
                        torch.zeros_like(self.v_bias, requires_grad=False),
                        self.v_bias,
                    )
                )

            qkv = f.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(b, n, 3, self.num_heads, -1).permute(
                2, 0, 3, 1, 4
            )  # 3, B, num_heads, N, C
            q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope:
            # slightly fast impl
            q_t = q[:, :, 1:, :]
            ro_q_t = self.rope(q_t)
            q = torch.cat((q[:, :, :1, :], ro_q_t), -2).type_as(v)

            k_t = k[:, :, 1:, :]
            ro_k_t = self.rope(k_t)
            k = torch.cat((k[:, :, :1, :], ro_k_t), -2).type_as(v)

        if self.xattn:
            if xops is None:
                raise ValueError(
                    "Can't use xattn without xformers. Please 'pip install xformers'"
                )
            q = q.permute(0, 2, 1, 3)  # B, num_heads, N, C -> B, N, num_heads, C
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            x = xops.memory_efficient_attention(
                q,
                k,
                v,
                p=self.xattn_drop,
                scale=self.scale,
            )
            x = x.reshape(b, n, -1)
            x = self.inner_attn_ln(x)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            if self.relative_position_bias_table is not None:
                relative_position_bias = self.relative_position_bias_table[
                    self.relative_position_index.view(-1)
                ].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1,
                    -1,
                )  # Wh*Ww,Wh*Ww,nH
                relative_position_bias = relative_position_bias.permute(
                    2, 0, 1
                ).contiguous()  # nH, Wh*Ww, Wh*Ww
                attn = attn + relative_position_bias.unsqueeze(0).type_as(attn)

            if rel_pos_bias is not None:
                attn = attn + rel_pos_bias.type_as(attn)

            if attn_mask is not None:
                attn_mask = attn_mask.bool()
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float('-inf'))

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(b, n, -1)
            x = self.inner_attn_ln(x)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        window_size=None,
        attn_head_dim=None,
        xattn=False,
        rope=None,
        postnorm=False,
        subln=False,
        naiveswiglu=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
            attn_head_dim=attn_head_dim,
            xattn=xattn,
            rope=rope,
            subln=subln,
            norm_layer=norm_layer,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better
        # than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if naiveswiglu:
            self.mlp = SwiGLU(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                subln=subln,
                norm_layer=norm_layer,
            )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                subln=subln,
                drop=drop,
            )

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim,)), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim,)), requires_grad=True
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.postnorm = postnorm

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        if self.gamma_1 is None:
            if self.postnorm:
                x = x + self.drop_path(
                    self.norm1(
                        self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
                    )
                )
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(
                    self.attn(
                        self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask
                    )
                )
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.postnorm:
                x = x + self.drop_path(
                    self.gamma_1
                    * self.norm1(
                        self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
                    )
                )
                x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(
                    self.gamma_1
                    * self.attn(
                        self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask
                    )
                )
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x, **_):
        target_dtype = self.proj.weight.dtype
        _, __, h, w = x.shape
        # FIXME look at relaxing size constraints
        assert h == self.img_size[0] and w == self.img_size[1], (
            f"Input image size ({h}*{w}) doesn't match model "
            f'({self.img_size[0]}*{self.img_size[1]}).'
        )
        x = self.proj(x.to(dtype=target_dtype)).flatten(2).transpose(1, 2)
        return x


class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (
            2 * window_size[1] - 1
        ) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
        )
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer('relative_position_index', relative_position_index)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] + 1,
            self.window_size[0] * self.window_size[1] + 1,
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class EVAVisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        patch_dropout=0.0,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        rope=False,
        use_mean_pooling=True,
        init_scale=0.001,
        grad_checkpointing=False,
        xattn=False,
        postnorm=False,
        pt_hw_seq_len=16,
        intp_freq=False,
        naiveswiglu=False,
        subln=False,
        proj_type=None,
    ):
        super().__init__()
        self.image_size = img_size
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_embed.patch_shape, num_heads=num_heads
            )
        else:
            self.rel_pos_bias = None

        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len if intp_freq else None,
                patch_dropout=patch_dropout,
            )
        else:
            self.rope = None

        self.naiveswiglu = naiveswiglu

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=self.patch_embed.patch_shape
                    if use_rel_pos_bias
                    else None,
                    xattn=xattn,
                    rope=self.rope,
                    postnorm=postnorm,
                    subln=subln,
                    naiveswiglu=naiveswiglu,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        if (num_classes == embed_dim) and (proj_type is None):
            self.head = nn.Identity()
        elif proj_type == 'linear':
            self.head = nn.Linear(embed_dim, num_classes, bias=qkv_bias)
        elif proj_type == 'mlp':
            hidden_size = (embed_dim + num_classes) // 2
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, hidden_size, bias=qkv_bias),
                nn.GELU(),
                nn.Linear(hidden_size, num_classes, bias=qkv_bias),
            )

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        trunc_normal_(self.cls_token, std=0.02)

        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=0.02)
            self.head.weight.data.mul_(init_scale)
            if qkv_bias:
                self.head.bias.data.mul_(init_scale)

        # setting a patch_dropout of 0. would mean it is disabled and this function
        # would be the identity fn
        self.patch_dropout = (
            PatchDropout(patch_dropout) if patch_dropout > 0.0 else nn.Identity()
        )

        self.grad_checkpointing = grad_checkpointing

    def fix_init_weight(self):
        def rescale(param, _layer_id):
            param.div_(math.sqrt(2.0 * _layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if self.naiveswiglu:
                rescale(layer.mlp.w3.weight.data, layer_id + 1)
            else:
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_cast_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    def lock(self, unlocked_groups=0, *_, **__):
        assert (
            unlocked_groups == 0
        ), 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, *_, **__):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x, return_all_features=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # a patch_dropout of 0. would mean it is disabled and this function would do
        # nothing but return what was passed in
        if self.rope is not None:
            if self.training and not isinstance(self.patch_dropout, nn.Identity):
                x, patch_indices_keep = self.patch_dropout(x)
                self.rope.forward = partial(
                    self.rope.forward, patch_indices_keep=patch_indices_keep
                )
            else:
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=None)
                x = self.patch_dropout(x)
        else:
            x = self.patch_dropout(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            if self.grad_checkpointing:
                x = checkpoint(blk, x, (rel_pos_bias,))
            else:
                x = blk(x, rel_pos_bias=rel_pos_bias)

        if not return_all_features:
            x = self.norm(x)
            if self.fc_norm is not None:
                return self.fc_norm(x.mean(1))
            else:
                return x[:, 0]
        return x

    def forward(self, x, return_all_features=False):
        if return_all_features:
            return self.forward_features(x, return_all_features)
        x = self.forward_features(x)
        x = self.head(x)
        return x                

# end of EVA MODEL ----------------------------------------------------


# MODELING CLIP -------------------------------------------------------

class LayerNorm(nn.LayerNorm):  # ME: maybe remove
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        origtype = x.dtype
        x = f.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(origtype)


def _build_text_tower(config: JinaCLIPTextConfig) -> HFTextEncoder:
    return HFTextEncoder(
        model_name_or_path=config.hf_model_name_or_path,
        output_dim=config.embed_dim,
        default_instruction_task=config.default_instruction_task,
        default_lora_task=config.default_lora_task,
        pooler_type=config.pooler_type,
        proj_type=config.proj_type,
        proj_bias=config.proj_bias,
        pretrained=False,
        output_tokens=False,
        trust_remote_code=True,
        revision=None,
        model_config_kwargs=config.hf_model_config_kwargs,
    )


def _build_vision_tower(config: JinaCLIPVisionConfig) -> EVAVisionTransformer:
    norm_layer = partial(LayerNorm, eps=1e-6)

    if config.fused_layer_norm:
        try:
            from apex.normalization import FusedLayerNorm

            norm_layer = partial(FusedLayerNorm, eps=1e-6)
        except (ModuleNotFoundError, ImportError):
            logger.warning('Please install apex to use fused layer norm, ignoring')

    return EVAVisionTransformer(
        img_size=config.image_size,
        patch_size=config.patch_size,
        num_classes=config.embed_dim,
        use_mean_pooling=False,
        init_values=config.ls_init_value,
        patch_dropout=config.patch_dropout,
        embed_dim=config.width,
        depth=config.layers,
        num_heads=config.width // config.head_width,
        mlp_ratio=config.mlp_ratio,
        qkv_bias=config.qkv_bias,
        drop_path_rate=config.drop_path_rate,
        norm_layer=norm_layer,
        xattn=config.x_attention,
        rope=config.rope_embeddings,
        postnorm=config.post_norm,
        pt_hw_seq_len=config.pt_hw_seq_len,
        intp_freq=config.intp_freq,
        naiveswiglu=config.naive_swiglu,
        subln=config.subln,
        proj_type=config.proj_type,
    )


def _resolve_attention_libs(config: JinaCLIPConfig):
    use_text_flash_attn = (
        config.use_text_flash_attn
        if config.use_text_flash_attn is not None
        else config.text_config.hf_model_config_kwargs.get('use_flash_attn', True)
    )
    use_vision_xformers = (
        config.use_vision_xformers
        if config.use_vision_xformers is not None
        else config.vision_config.x_attention
    )

    def _resolve_use_text_flash_attn() -> bool:
        if use_text_flash_attn:
            if not torch.cuda.is_available():
                warnings.warn('Flash attention requires CUDA, disabling')
                return False
            if importlib.util.find_spec('flash_attn') is None:
                warnings.warn(
                    'Flash attention is not installed. Check '
                    'https://github.com/Dao-AILab/flash-attention?'
                    'tab=readme-ov-file#installation-and-features '
                    'for installation instructions, disabling'
                )
                return False
            major, minor, *_ = torch.version.cuda.split('.')
            major, minor = int(major), int(minor)
            if major < 11 or (major == 11 and minor < 7):
                warnings.warn(
                    'Flash attention requires CUDA>=11.7. Found version '
                    f'{major}.{minor}, disabling'
                )
                return False
            capability = torch.cuda.get_device_capability()
            major, *_ = capability
            major = int(major)
            if major < 8:
                device_name = torch.cuda.get_device_properties(0).name
                warnings.warn(
                    'Flash attention requires device capability>=8.0 (NVIDIA Ampere, '
                    f'Hopper or ADA). Found device {device_name} with capability '
                    f'{capability}, disabling'
                )
                return False
            return True
        return False

    def _resolve_use_vision_xformers() -> bool:
        if use_vision_xformers:
            if not torch.cuda.is_available():
                warnings.warn('xFormers requires CUDA, disabling')
                return False
            if importlib.util.find_spec('xformers') is None:
                warnings.warn(
                    'xFormers is not installed. Check '
                    'https://github.com/facebookresearch/xformers?'
                    'tab=readme-ov-file#installing-xformers for installation '
                    'instructions, disabling'
                )
                return False
            return True
        return False

    _use_text_flash_attn = _resolve_use_text_flash_attn()
    _use_vision_xformers = _resolve_use_vision_xformers()

    config.use_text_flash_attn = _use_text_flash_attn
    config.use_vision_xformers = _use_vision_xformers
    config.text_config.hf_model_config_kwargs['use_flash_attn'] = _use_text_flash_attn
    config.vision_config.x_attention = _use_vision_xformers

    return config


class JinaCLIPPreTrainedModel(nn.Module):
    """
    An abstract class to handle weights initialization and a simple interface for
    downloading and loading pretrained models.
    """

    config_class = JinaCLIPConfig
    base_model_prefix = 'clip'
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, JinaCLIPModel):
            if isinstance(module.text_projection, nn.Linear):
                nn.init.normal_(
                    module.text_projection.weight,
                    std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
                )
            if isinstance(module.text_projection, nn.Linear):
                nn.init.normal_(
                    module.visual_projection.weight,
                    std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
                )
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        if 'torch_dtype' not in kwargs:
            kwargs['torch_dtype'] = 'auto'
        return super().from_pretrained(*args, **kwargs)


class JinaCLIPTextModel(JinaCLIPPreTrainedModel):
    config_class = JinaCLIPTextConfig

    def __init__(self, config: JinaCLIPTextConfig):
        super().__init__(config)
        self.text_model = _build_text_tower(config)
        self.post_init()

    def forward(
        self,
        input_ids: Union[None, torch.Tensor, BatchEncoding] = None,
        return_dict: Optional[bool] = None,
        *_,
        **__,
    ) -> Union[Tuple[Optional[torch.FloatTensor], ...], CLIPTextModelOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        x = input_ids.input_ids if isinstance(input_ids, BatchEncoding) else input_ids
        feats = self.text_model(x=x)
        out = CLIPTextModelOutput(text_embeds=feats)
        return out if return_dict else out.to_tuple()


class JinaCLIPVisionModel(JinaCLIPPreTrainedModel):
    config_class = JinaCLIPVisionConfig
    main_input_name = 'pixel_values'

    def __init__(self, config: JinaCLIPVisionConfig):
        super().__init__(config)
        self.vision_model = _build_vision_tower(config)
        self.post_init()

    def forward(
        self,
        pixel_values: Union[None, torch.FloatTensor, BatchFeature] = None,
        return_dict: Optional[bool] = None,
        *_,
        **__,
    ) -> Union[Tuple[Optional[torch.FloatTensor], ...], CLIPVisionModelOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        x = (
            pixel_values.pixel_values
            if isinstance(pixel_values, BatchFeature)
            else pixel_values
        )
        feats = self.vision_model(x=x)
        out = CLIPVisionModelOutput(image_embeds=feats)
        return out if return_dict else out.to_tuple()


class JinaCLIPModel(JinaCLIPPreTrainedModel):
    config_class = JinaCLIPConfig

    def __init__(
            self, 
            vllm_config: VllmConfig,
            # config: JinaCLIPConfig,
            prefix: str = ""
        ):
        config = AutoConfig.from_pretrained(vllm_config.model, trust_remote_code=vllm_config.trust_remote_code)

        super().__init__(config)

        if not isinstance(config.text_config, JinaCLIPTextConfig):
            raise ValueError(
                'Attribute config.text_config is expected to be of type '
                f'JinaCLIPTextConfig but is of type {type(config.text_config)}.'
            )

        if not isinstance(config.vision_config, JinaCLIPVisionConfig):
            raise ValueError(
                'Attribute config.vision_config is expected to be of type '
                f'JinaCLIPVisionConfig but is of type {type(config.vision_config)}.'
            )

        config = _resolve_attention_libs(config)
        text_config = config.text_config
        vision_config = config.vision_config

        self.add_projections = config.add_projections
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.embed_dim
        self.vision_embed_dim = vision_config.embed_dim
        self.text_model = _build_text_tower(text_config)
        self.vision_model = _build_vision_tower(vision_config)
        self.logit_scale = nn.Parameter(
            torch.tensor(self.config.logit_scale_init_value)
        )
        if self.add_projections:
            self.visual_projection = nn.Linear(
                self.vision_embed_dim, self.projection_dim, bias=False
            )
            self.text_projection = nn.Linear(
                self.text_embed_dim, self.projection_dim, bias=False
            )
        else:
            self.visual_projection = nn.Identity()
            self.text_projection = nn.Identity()

        self.tokenizer = None
        self.preprocess = None
        self.post_init()

    def get_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config._name_or_path, trust_remote_code=True
            )
        return self.tokenizer

    def get_preprocess(self):
        if not self.preprocess:
            self.preprocess = AutoImageProcessor.from_pretrained(
                self.config._name_or_path, trust_remote_code=True
            )
        return self.preprocess

    def get_text_features(
        self,
        input_ids: Union[None, torch.Tensor, BatchEncoding] = None,
        *_,
        **__,
    ) -> torch.FloatTensor:
        x = input_ids.input_ids if isinstance(input_ids, BatchEncoding) else input_ids
        return self.text_projection(self.text_model(x=x))

    def get_image_features(
        self,
        pixel_values: Union[None, torch.FloatTensor, BatchFeature] = None,
        *_,
        **__,
    ) -> torch.FloatTensor:
        x = (
            pixel_values.pixel_values
            if isinstance(pixel_values, BatchFeature)
            else pixel_values
        )
        return self.visual_projection(self.vision_model(x=x))

    def _truncate_embeddings(self, embeddings: torch.Tensor, truncate_dim: int):
        if not self.config.matryoshka_dimensions:
            logger.warning(
                'Model is not trained using Matryoshka Representation Learning, '
                'truncating embeddings will not work optimally.'
            )
        return embeddings[:, :truncate_dim]

    @staticmethod
    def _decode_image_data(image_data_str: str) -> Image:
        header, data = image_data_str.split(',', 1)
        image_data = base64.b64decode(data)
        return Image.open(BytesIO(image_data))

    @torch.inference_mode()
    def encode_image(
        self,
        images: Union[str, List[Union[str, 'Image.Image']]],
        batch_size: int = 32,
        show_progress_bar: Optional[bool] = None,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: Optional[torch.device] = None,
        normalize_embeddings: bool = True,
        truncate_dim: Optional[int] = None,
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:
        """
        Computes image embeddings
        Args:
            images(`str` or `List[Union[str, Image.Image]]`):
                Image paths, URLs, PIL images, or data:image/ strings to be encoded
            batch_size(`int`, *optional*, defaults to 32):
                Batch size for the computation
            show_progress_bar(`bool`, *optional*, defaults to None):
                Show a progress bar when encoding images. If set to None, progress bar
                is only shown when `logger.level == logging.INFO` or
                `logger.level == logging.DEBUG`
            convert_to_numpy(`bool`, *optional*, defaults to True):
                If true, the output is a list of numpy vectors. Else, it is a list of
                pytorch tensors
            convert_to_tensor(`bool`, *optional*, defaults to False):
                If true, you get one large tensor as return. Overwrites any setting
                from convert_to_numpy
            device(`torch.device`, *optional*, defaults to None):
                Which torch.device to use for the computation
            normalize_embeddings(`bool`, *optional*, defaults to True):
                If set to true, returned vectors will have length 1. In that case,
                the faster dot-product (util.dot_score) instead of cosine similarity
                can be used
            truncate_dim(`int`, *optional*, defaults to None):
                The dimension to truncate sentence embeddings to. If set to `None`
                no truncation is performed
        Returns:
            By default, a list of tensors is returned. If convert_to_tensor, a stacked
            tensor is returned. If convert_to_numpy, a numpy matrix is returned
        """

        _is_training = self.training
        self.eval()

        self.preprocess = self.get_preprocess()
        all_embeddings = []

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )
        if convert_to_tensor:
            convert_to_numpy = False

        _input_was_single_img = False
        if isinstance(images, str) or not hasattr(images, '__len__'):
            images = [images]
            _input_was_single_img = True

        if device is not None:
            self.to(device)

        _permutation = np.argsort([-len(str(i)) for i in images])
        _inverse_permutation = np.argsort(_permutation)
        images = [images[idx] for idx in _permutation]

        if has_tqdm:
            range_iter = trange(
                0,
                len(images),
                batch_size,
                desc='Encoding',
                disable=not show_progress_bar,
            )
        else:
            range_iter = range(0, len(images), batch_size)

        truncate_dim = truncate_dim or self.config.truncate_dim

        for i in range_iter:
            _processed_images = []
            for img in images[i: i + batch_size]:
                if isinstance(img, str):
                    if img.startswith('http'):
                        response = requests.get(img)
                        image = Image.open(BytesIO(response.content)).convert('RGB')
                    elif img.startswith('data:image/'):
                        image = self._decode_image_data(img).convert('RGB')
                    else:
                        image = Image.open(img).convert('RGB')
                elif isinstance(img, Image.Image):
                    image = img.convert('RGB')
                else:
                    raise ValueError('Unsupported image format')
                _processed_images.append(image)

            pixelvals = self.preprocess(_processed_images)
            pixelvals = pixelvals.to(self.device)
            embeddings = self.get_image_features(pixelvals)

            if truncate_dim:
                embeddings = self._truncate_embeddings(embeddings, truncate_dim)
            if normalize_embeddings:
                embeddings = f.normalize(embeddings, p=2, dim=1)
            if convert_to_numpy:
                embeddings = embeddings.cpu()

            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in _inverse_permutation]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray(
                [emb.to(torch.float32).numpy() for emb in all_embeddings]
            )

        if _input_was_single_img:
            all_embeddings = all_embeddings[0]

        self.train(_is_training)
        return all_embeddings

    @torch.inference_mode()
    def encode_text(
        self,
        sentences: Union[str, List[str]],
        task: Optional[str] = None,
        batch_size: int = 32,
        show_progress_bar: Optional[bool] = None,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: Optional[torch.device] = None,
        normalize_embeddings: bool = True,
        truncate_dim: Optional[int] = None,
        **tokenizer_kwargs,
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:
        """
        Computes text embeddings
        Args:
            sentences(`str` or `List[str]`):
                Sentence or sentences to be encoded
            task(`str`, *optional*, defaults to `None`):
                Specifies the task for which the encoding is intended. If a `task` is
                provided, a task-specific instruction is added to the beginning of each
                sentence. If `task` is not provided, no instructions are added.
            batch_size(`int`, *optional*, defaults to 32):
                Batch size for the computation
            show_progress_bar(`bool`, *optional*, defaults to None):
                Show a progress bar when encoding sentences. If set to None, progress
                bar is only shown when `logger.level == logging.INFO` or
                `logger.level == logging.DEBUG`
            convert_to_numpy(`bool`, *optional*, defaults to True):
                If true, the output is a list of numpy vectors. Else, it is a list of
                pytorch tensors
            convert_to_tensor(`bool`, *optional*, defaults to False):
                If true, you get one large tensor as return. Overwrites any setting
                from convert_to_numpy
            device(`torch.device`, *optional*, defaults to None):
                Which torch.device to use for the computation
            normalize_embeddings(`bool`, *optional*, defaults to True):
                If set to true, returned vectors will have length 1. In that case,
                the faster dot-product (util.dot_score) instead of cosine similarity
                can be used
            truncate_dim(`int`, *optional*, defaults to None):
                The dimension to truncate sentence embeddings to. If set to `None`
                no truncation is performed
            tokenizer_kwargs(`Dict[str, Any]`, *optional*, defaults to {}):
                Keyword arguments for the tokenizer
        Returns:
            By default, a list of tensors is returned. If convert_to_tensor, a stacked
            tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        _is_training = self.training
        self.eval()

        all_embeddings = []
        self.tokenizer = self.get_tokenizer()

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )
        if convert_to_tensor:
            convert_to_numpy = False

        _input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]
            _input_was_string = True

        if device is not None:
            self.to(device)

        _permutation = np.argsort([-len(i) for i in sentences])
        _inverse_permutation = np.argsort(_permutation)
        sentences = [sentences[idx] for idx in _permutation]

        tokenizer_kwargs['padding'] = tokenizer_kwargs.get('padding', True)
        tokenizer_kwargs['max_length'] = tokenizer_kwargs.get('max_length', 512)
        tokenizer_kwargs['truncation'] = tokenizer_kwargs.get('truncation', True)

        if has_tqdm:
            range_iter = trange(
                0,
                len(sentences),
                batch_size,
                desc='Encoding',
                disable=not show_progress_bar,
            )
        else:
            range_iter = range(0, len(sentences), batch_size)

        truncate_dim = truncate_dim or self.config.truncate_dim

        instruction = self.text_model.get_instruction_from_task(task)
        if instruction:
            sentences = [instruction + sentence for sentence in sentences]

        for i in range_iter:
            tokens = self.tokenizer(
                sentences[i: i + batch_size],
                return_tensors='pt',
                **tokenizer_kwargs,
            ).to(self.device)
            embeddings = self.get_text_features(input_ids=tokens)
            if truncate_dim:
                embeddings = self._truncate_embeddings(embeddings, truncate_dim)
            if normalize_embeddings:
                embeddings = f.normalize(embeddings, p=2, dim=1)
            if convert_to_numpy:
                embeddings = embeddings.cpu()
            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in _inverse_permutation]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray(
                [emb.to(torch.float32).numpy() for emb in all_embeddings]
            )
        if _input_was_string:
            all_embeddings = all_embeddings[0]

        self.train(_is_training)
        return all_embeddings

    def forward(  # ME: new forward, for now supports only text input
        self,
        input_ids: Union[None, torch.Tensor, BatchEncoding] = None,
        # pixel_values: Union[None, torch.FloatTensor, BatchFeature] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        # image_embeds = self.get_image_features(pixel_values=pixel_values) # ME: i've disabled this temporarily
        text_embeds = self.get_text_features(input_ids=input_ids)

        # normalized features
        # image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)  # ME!
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        return text_embeds

        """
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (
                logits_per_image,
                logits_per_text,
                text_embeds,
                image_embeds,
                None,
                None,
            )
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=None,
            vision_model_output=None,
        )
        """
    """
    def forward_OLD(  # ME: was forward
        self,
        input_ids: Union[None, torch.Tensor, BatchEncoding] = None,
        pixel_values: Union[None, torch.FloatTensor, BatchFeature] = None,
        return_dict: Optional[bool] = None,
        return_loss: Optional[bool] = None,
        *_,
        **__,
    ) -> Union[Tuple[Optional[torch.FloatTensor], ...], CLIPOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        image_embeds = self.get_image_features(pixel_values=pixel_values)
        text_embeds = self.get_text_features(input_ids=input_ids)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = clip_loss(logits_per_text)

        if not return_dict:
            output = (
                logits_per_image,
                logits_per_text,
                text_embeds,
                image_embeds,
                None,
                None,
            )
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=None,
            vision_model_output=None,
        )
    """
