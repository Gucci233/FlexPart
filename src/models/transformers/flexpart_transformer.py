



import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.embeddings import (
  GaussianFourierProjection,
  TimestepEmbedding,
  Timesteps,
)
import numpy as np
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import (
  AdaLayerNormContinuous,
  FP32LayerNorm,
  LayerNorm,
)
from torchvision.ops import generalized_box_iou_loss
from diffusers.utils import (
  USE_PEFT_BACKEND,
  is_torch_version,
  logging,
  scale_lora_layers,
  unscale_lora_layers,
)
from diffusers.utils.torch_utils import maybe_allow_in_graph
from typing import *


from ..attention_processor import FusedTripoSGAttnProcessor2_0, TripoSGAttnProcessor2_0, PartCrafterAttnProcessor
from .modeling_outputs import Transformer1DModelOutput

logger = logging.get_logger(__name__) # pylint: disable=invalid-name


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Helper function for AdaLN.
    x: [N, T, D]
    shift, scale: [N, D]
    """
    # We unsqueeze(1) to make shift/scale broadcastable with x
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class PointPromptEncoder(nn.Module):
  """
  Encodes 2D point prompts (x, y) into a high-dimensional embedding.
  Uses GaussianFourierProjection for positional encoding.
  """
  def __init__(self, output_dim: int, fourier_dim: int = 128, fourier_scale: float = 4.0):
    super().__init__()
    self.output_dim = output_dim

    self.point_gff_encoder = GaussianFourierProjection(embedding_size=fourier_dim, scale=fourier_scale,log=False)

    prompt_input_dim = fourier_dim * 2 * 2
    self.prompt_mlp = nn.Sequential(
      nn.Linear(prompt_input_dim, self.output_dim),
      nn.ReLU(),
      nn.Linear(self.output_dim, self.output_dim)
    )

  def forward(self, point_prompt_xy: torch.Tensor) -> torch.Tensor:
    x_prompt = point_prompt_xy[:, 0:1] # [N, 1]
    y_prompt = point_prompt_xy[:, 1:2] # [N, 1]

    x_embed = self.point_gff_encoder(x_prompt.squeeze(-1)) # [N, fourier_dim * 2]
    y_embed = self.point_gff_encoder(y_prompt.squeeze(-1)) # [N, fourier_dim * 2]

    prompt_fourier_features = torch.cat([x_embed, y_embed], dim=-1)
    point_prompt_condition = self.prompt_mlp(prompt_fourier_features)

    return point_prompt_condition


class BoxPromptEncoder(nn.Module):
    """
    Encodes 2D bounding boxes (Box) into a high-dimensional embedding.
    (Inspired by the SAM architecture)

    1. Decompose (x1, y1, x2, y2) into two corner points: top-left and bottom-right.
    2. Encode these two (x, y) coordinates using Gaussian Fourier Features (GFF).
    3. Add learnable "corner type" embeddings for "tl" (top-left) and "br" (bottom-right) respectively.
    4. Pass both through an MLP.
    5. Sum the final embeddings of the two corners to produce the box_embedding.
    """
    def __init__(self, output_dim: int, fourier_dim: int = 128, fourier_scale: float = 4.0):
        super().__init__()
        self.output_dim = output_dim

        self.gff_encoder = GaussianFourierProjection(
            embedding_size=fourier_dim, scale=fourier_scale, log=False
        )

        gff_output_dim = fourier_dim * 2 * 2

        self.corner_type_embedding = nn.Embedding(2, gff_output_dim)

        self.corner_mlp = nn.Sequential(
            nn.Linear(gff_output_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim)
        )

    def _encode_corner(self, xy_coords: torch.Tensor, corner_type: int) -> torch.Tensor:
        x_c = xy_coords[:, 0:1]
        y_c = xy_coords[:, 1:2]

        x_embed = self.gff_encoder(x_c.squeeze(-1)) # [B, fourier_dim * 2]
        y_embed = self.gff_encoder(y_c.squeeze(-1)) # [B, fourier_dim * 2]

        features = torch.cat([x_embed, y_embed], dim=-1)

        type_emb = self.corner_type_embedding(
            torch.tensor(corner_type, device=features.device)
        )

        return self.corner_mlp(features + type_emb)

    def forward(self, box_xyxy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            box_xyxy (torch.Tensor): [B, 4] 归一化坐标 (x1, y1, x2, y2)
        """
        # [B, 2]
        top_left = box_xyxy[:, :2]
        bottom_right = box_xyxy[:, 2:]

        # 编码两个角
        emb_tl = self._encode_corner(top_left, corner_type=0)     # [B, output_dim]
        emb_br = self._encode_corner(bottom_right, corner_type=1) # [B, output_dim]

        # 将两个嵌入相加，得到最终的 Box 嵌入
        box_embedding = emb_tl + emb_br
        return box_embedding


class MaskPromptEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        base_dim = 16
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, base_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(),
            nn.Conv2d(base_dim, base_dim*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_dim*2),
            nn.ReLU(),
            nn.Conv2d(base_dim*2, base_dim*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_dim*4),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(base_dim * 4, output_dim)

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        if masks.dim() == 3:
            x = masks.unsqueeze(1)
        else:
            x = masks

        B, C, H, W = x.shape

        x = F.interpolate(x, size=(512, 512), mode='nearest')

        feat = self.conv_net(x)
        feat = self.gap(feat).flatten(1)
        feat = self.proj(feat)

        return feat

class Box3DPromptEncoder(nn.Module):
    def __init__(self, output_dim: int, fourier_dim: int = 128, fourier_scale: float = 4.0):
        super().__init__()
        self.output_dim = output_dim

        self.gff_encoder = GaussianFourierProjection(
            embedding_size=fourier_dim, scale=fourier_scale,log=False
        )

        gff_output_dim = fourier_dim * 2 * 3

        self.corner_type_embedding = nn.Embedding(2, gff_output_dim)

        self.corner_mlp = nn.Sequential(
            nn.Linear(gff_output_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
            nn.Linear(self.output_dim, self.output_dim)
        )

    def _encode_corner(self, xyz_coords: torch.Tensor, corner_type: int) -> torch.Tensor:
        x = xyz_coords[:, 0]
        y = xyz_coords[:, 1]
        z = xyz_coords[:, 2]

        x_emb = self.gff_encoder(x)
        y_emb = self.gff_encoder(y)
        z_emb = self.gff_encoder(z)

        features = torch.cat([x_emb, y_emb, z_emb], dim=-1)

        type_emb = self.corner_type_embedding(
            torch.tensor(corner_type, device=features.device)
        )

        return self.corner_mlp(features + type_emb)

    def forward(self, box_3d: torch.Tensor) -> torch.Tensor:
        box_3d = box_3d
        # 分解为 Min 角和 Max 角
        min_corner = box_3d[:, 0, :] # [B, 3]
        max_corner = box_3d[:, 1, :] # [B, 3]

        # 分别编码
        emb_min = self._encode_corner(min_corner, corner_type=0)
        emb_max = self._encode_corner(max_corner, corner_type=1)

        box_embedding = emb_min + emb_max

        return box_embedding

class SinkhornWithDustbin(nn.Module):
    """
    Sinkhorn in log-space with a single learnable dustbin score.
    Supports inputs of shape [N, M] or [B, N, M] (square or rectangular).
    Assumes logits already scaled by external logit_scale.
    """
    def __init__(self, n_iters: int = 50, learnable=True, init_value: float = 1.0):
        super().__init__()
        self.n_iters = n_iters
        if learnable:
            self.bin_score = nn.Parameter(torch.tensor(init_value))
        else:
            self.register_buffer("bin_score", torch.tensor(init_value))

    def forward(self, logits, mask_row=None, mask_col=None):
        """
        logits: [N, M] or [B, N, M] (not yet augmented)
        mask_row: optional bool mask for valid rows shape [N] or [B, N]
        mask_col: optional bool mask for valid cols shape [M] or [B, M]
        returns dict with augmented P and components:
          "P_aug": [B, N+1, M+1] or [N+1, M+1]
          "P_match": [B, N, M] or [N, M]
          "P_row_bin": [B, N] or [N]
          "P_col_bin": [B, M] or [M]
          "P_bin": scalar or [B]
        """
        single = logits.dim() == 2
        if single:
            logits = logits.unsqueeze(0)  # [1, N, M]
        B, N, M = logits.shape
        device = logits.device
        b = self.bin_score

        # Build augmented logit matrix L of shape [B, N+1, M+1]
        L = logits.new_full((B, N + 1, M + 1), fill_value=-1e4)  # large negative for unused by default

        # top-left: original logits
        L[:, :N, :M] = logits

        # top-right: row -> bin (N x 1) fill with b
        L[:, :N, M] = b

        # bottom-left: bin <- col (1 x M) fill with b
        L[:, N, :M] = b

        # bottom-right: bin-bin = 0
        L[:, N, M] = 0.0

        # Optional: mask invalid rows/cols by setting to -inf so they get zero prob.
        # mask_row: if provided, shape [B, N] or [N]
        if mask_row is not None:
            mr = mask_row
            if mr.dim() == 1:
                mr = mr.unsqueeze(0).expand(B, -1)  # [B, N]
            # where invalid rows -> set entire that row (except bin) to large negative
            invalid = (~mr).to(torch.bool)
            if invalid.any():
                L[invalid, :M] = -1e9  # sets top-left entries in invalid rows
                L[invalid, M] = b      # keep the row->bin entry as b (or set to -inf if you want forbid)
        if mask_col is not None:
            mc = mask_col
            if mc.dim() == 1:
                mc = mc.unsqueeze(0).expand(B, -1)  # [B, M]
            invalidc = (~mc).to(torch.bool)
            if invalidc.any():
                L[invalidc, :, :M] = -1e9  # careful broadcasting, you may want explicit loops
                L[:, N, invalidc] = b     # keep col->bin

        # Sinkhorn in log-space
        log_alpha = L  # already logits scaled outside

        for _ in range(self.n_iters):
            # row normalize: logsumexp over columns
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
            # col normalize: logsumexp over rows
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)

        P = torch.exp(log_alpha)  # [B, N+1, M+1]

        P_match = P[:, :N, :M]
        P_row_bin = P[:, :N, M]
        P_col_bin = P[:, N, :M]
        P_bin = P[:, N, M]

        if single:
            # squeeze batch dim
            return {
                "P_aug": P[0],
                "P_match": P_match[0],
                "P_row_bin": P_row_bin[0],
                "P_col_bin": P_col_bin[0],
                "P_bin": P_bin[0]
            }
        else:
            return {
                "P_aug": P,
                "P_match": P_match,
                "P_row_bin": P_row_bin,
                "P_col_bin": P_col_bin,
                "P_bin": P_bin
            }

class PartVisualBridge(nn.Module):
    def __init__(self, part_dim=256, image_dim=768, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=part_dim, kdim=image_dim, vdim=image_dim, num_heads=num_heads, batch_first=True)

        self.norm1 = nn.LayerNorm(part_dim)
        self.norm2 = nn.LayerNorm(part_dim)
        self.ffn = nn.Sequential(
            nn.Linear(part_dim, part_dim * 4),
            nn.GELU(),
            nn.Linear(part_dim * 4, part_dim)
        )

    def forward(self, part_emb, image_emb):
        # 1. Cross Attention
        # query=part, key=value=image
        part_emb = part_emb.unsqueeze(1)
        attn_out, _ = self.cross_attn(query=self.norm1(part_emb), key=image_emb, value=image_emb)

        # Residual Connection
        x = part_emb + attn_out

        # 2. FFN
        x = x + self.ffn(self.norm2(x))
        return x.squeeze(1)

@maybe_allow_in_graph
class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        adaln_input_dim: Optional[int] = None,# <--- 必需参数
        use_self_attention: bool = True,
        self_attention_norm_type: Optional[str] = None,
        use_cross_attention: bool = True,
        cross_attention_dim: Optional[int] = None,
        cross_attention_norm_type: Optional[str] = "fp32_layer_norm",
        dropout=0.0,
        activation_fn: str = "gelu",
        norm_type: str = "fp32_layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        skip: bool = False,
        skip_concat_front: bool = False,
        skip_norm_last: bool = False,
        qk_norm: bool = True,
        qkv_bias: bool = True,
    ):
        super().__init__()

        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention
        self.skip_concat_front = skip_concat_front
        self.skip_norm_last = skip_norm_last
        self.adaln_input_dim = adaln_input_dim
        self.dim = dim

        current_norm_affine = True


        if use_self_attention:
            if (
            self_attention_norm_type == "fp32_layer_norm"
            or self_attention_norm_type is None
        ):
                self.norm1 = FP32LayerNorm(dim, norm_eps, current_norm_affine)
            else:
                raise NotImplementedError

            self.attn1 = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=dim // num_attention_heads,
                heads=num_attention_heads,
                qk_norm="rms_norm" if qk_norm else None,
                eps=1e-6,
                bias=qkv_bias,
                processor=TripoSGAttnProcessor2_0(),
            )
        else:
            self.norm1 = None
            self.attn1 = None


        if use_cross_attention:
            assert cross_attention_dim is not None
            self.norm2 = FP32LayerNorm(dim, norm_eps, current_norm_affine)
            self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            dim_head=dim // num_attention_heads,
            heads=num_attention_heads,
            qk_norm="rms_norm" if qk_norm else None,
            cross_attention_norm=cross_attention_norm_type,
            eps=1e-6,
            bias=qkv_bias,
            processor=TripoSGAttnProcessor2_0(),
        )
        else:
            self.norm2 = None
            self.attn2 = None


        self.norm3 = FP32LayerNorm(dim, norm_eps, current_norm_affine)
        self.ff = FeedForward(
        dim,
        dropout=dropout,
        activation_fn=activation_fn,
        final_dropout=final_dropout,
        inner_dim=ff_inner_dim,
        bias=ff_bias,
        )

        self.adaLN_modulation = None

        if skip:
            self.skip_norm = FP32LayerNorm(dim, norm_eps, elementwise_affine=True)
            self.skip_linear = nn.Linear(2 * dim, dim)
        else:
            self.skip_linear = None

        self._chunk_size = None
        self._chunk_dim = 0

    def set_topk(self, topk):
        self.flash_processor.topk = topk

    def set_flash_processor(self, flash_processor):
        self.flash_processor = flash_processor
        self.attn2.processor = self.flash_processor

    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim


    def add_adaln_modules(self):
        num_params = 0
        if self.use_self_attention:
            num_params += self.dim * 3
        if self.use_cross_attention:
            num_params += self.dim * 3
        num_params += self.dim * 3 # for FeedForward

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.adaln_input_dim, num_params)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        part_condition: Optional[torch.Tensor] = None, # <--- AdaLN 条件
        image_rotary_emb: Optional[torch.Tensor] = None,
        skip: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:

        attention_kwargs = attention_kwargs or {}

        assert part_condition is not None, "Gated AdaLN modulation requires part_condition"

        all_params = self.adaLN_modulation(part_condition)

        param_list = list(all_params.chunk(all_params.shape[1] // hidden_states.shape[-1], dim=1))

        if self.skip_linear is not None:
            cat = torch.cat(
            (
            [skip, hidden_states]
            if self.skip_concat_front
            else [hidden_states, skip]
            ),
            dim=-1,
        )
            if self.skip_norm_last:
                hidden_states = self.skip_linear(cat)
                hidden_states = self.skip_norm(hidden_states)
            else:
                cat = self.skip_norm(cat)
                hidden_states = self.skip_linear(cat)

        if self.use_self_attention:
            shift_attn1 = param_list.pop(0)
            scale_attn1 = param_list.pop(0)
            gate_attn1 = param_list.pop(0)

            norm_x = self.norm1(hidden_states)
            modulated_x = modulate(norm_x, shift_attn1, scale_attn1)
            attn_output = self.attn1(
            modulated_x,
            image_rotary_emb=image_rotary_emb,
            **attention_kwargs,
        )
            hidden_states = hidden_states + gate_attn1.unsqueeze(1) * attn_output

        if self.use_cross_attention:
            shift_attn2 = param_list.pop(0)
            scale_attn2 = param_list.pop(0)
            gate_attn2 = param_list.pop(0)

            norm_x = self.norm2(hidden_states)
            modulated_x = modulate(norm_x, shift_attn2, scale_attn2)
            attn_output = self.attn2(
            modulated_x,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **attention_kwargs,
        )
            hidden_states = hidden_states + gate_attn2.unsqueeze(1) * attn_output

        shift_ff = param_list.pop(0)
        scale_ff = param_list.pop(0)
        gate_ff = param_list.pop(0)

        norm_x = self.norm3(hidden_states)
        modulated_x = modulate(norm_x, shift_ff, scale_ff)
        ff_output = self.ff(modulated_x)
        hidden_states = hidden_states + gate_ff.unsqueeze(1) * ff_output

        return hidden_states



class FlexPartDiTModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    TripoSG: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88):
            The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        patch_size (`int`, *optional*):
            The size of the patch to use for the input.
        activation_fn (`str`, *optional*, defaults to `"geglu"`):
            Activation function to use in feed-forward.
        sample_size (`int`, *optional*):
            The width of the latent images. This is fixed during training since it is used to learn a number of
            position embeddings.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        cross_attention_dim (`int`, *optional*):
            The number of dimension in the clip text embedding.
        hidden_size (`int`, *optional*):
            The size of hidden layer in the conditioning embedding layers.
        num_layers (`int`, *optional*, defaults to 1):
            The number of layers of Transformer blocks to use.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            The ratio of the hidden layer size to the input size.
        learn_sigma (`bool`, *optional*, defaults to `True`):
             Whether to predict variance.
        cross_attention_dim_t5 (`int`, *optional*):
            The number dimensions in t5 text embedding.
        pooled_projection_dim (`int`, *optional*):
            The size of the pooled projection.
        text_len (`int`, *optional*):
            The length of the clip text embedding.
        text_len_t5 (`int`, *optional*):
            The length of the T5 text embedding.
        use_style_cond_and_image_meta_size (`bool`,  *optional*):
            Whether or not to use style condition and image meta size. True for version <=1.1, False for version >= 1.2
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        width: int = 2048,
        in_channels: int = 64,
        num_layers: int = 21,
        cross_attention_dim: int = 1024,
        max_num_parts: int = 32,
        enable_part_embedding=True,
        enable_local_cross_attn: bool = True,
        enable_global_cross_attn: bool = True,
        global_attn_block_ids: Optional[List[int]] = None,
        global_attn_block_id_range: Optional[List[int]] = None,

        enable_point_prompt: bool = True,
        point_prompt_fourier_dim: int = 128,
        enable_box_prompt: bool = True,
        box_prompt_fourier_dim: int = 128,
        enable_mask_prompt:bool = True,

        enable_3dbox_prompt:bool = False,

        add_module:bool = True,
    ):
        super().__init__()
        self.out_channels = in_channels
        self.num_heads = num_attention_heads
        self.inner_dim = width
        self.mlp_ratio = 4.0
        self.cross_attention_dim = cross_attention_dim
        time_embed_dim, timestep_input_dim = self._set_time_proj(
            "positional",
            inner_dim=self.inner_dim,
            flip_sin_to_cos=False,
            freq_shift=0,
            time_embedding_dim=None,
        )
        self.time_proj = TimestepEmbedding(
            timestep_input_dim, time_embed_dim, act_fn="gelu", out_dim=self.inner_dim
        )


        self.proj_in = nn.Linear(self.config.in_channels, self.inner_dim, bias=True)

        # 调制设置
        self.use_gated_adaln = enable_point_prompt | enable_box_prompt | enable_mask_prompt | enable_3dbox_prompt

        if self.use_gated_adaln:
            self.prompt_dim = 256
            self.adaln_cond_dim = self.prompt_dim

        else:
            self.prompt_dim = None
            self.null_prompt_embedding = None

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    adaln_input_dim=self.adaln_cond_dim,
                    use_self_attention=True,
                    self_attention_norm_type="fp32_layer_norm",
                    use_cross_attention=True,
                    cross_attention_dim=cross_attention_dim,
                    cross_attention_norm_type=None,
                    activation_fn="gelu",
                    norm_type="fp32_layer_norm",  # TODO
                    norm_eps=1e-5,
                    ff_inner_dim=int(self.inner_dim * self.mlp_ratio),
                    skip=layer > num_layers // 2,
                    skip_concat_front=True,
                    skip_norm_last=True,  # this is an error
                    qk_norm=True,  # See http://arxiv.org/abs/2302.05442 for details.
                    qkv_bias=False,
                )
                for layer in range(num_layers)
            ]
        )

        self.norm_out = LayerNorm(self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=True)

        self.gradient_checkpointing = False

        self.enable_local_cross_attn = enable_local_cross_attn
        self.enable_global_cross_attn = enable_global_cross_attn

        if global_attn_block_ids is None:
            global_attn_block_ids = []
            if global_attn_block_id_range is not None:
                global_attn_block_ids = list(range(global_attn_block_id_range[0], global_attn_block_id_range[1] + 1))
        self.global_attn_block_ids = global_attn_block_ids

        if len(global_attn_block_ids) > 0:
            # Override self-attention processors for global attention blocks
            attn_processor_dict = {}
            modified_attn_processor = []
            for layer_id in range(num_layers):
                for attn_id in [1, 2]:
                    if layer_id in global_attn_block_ids:
                        # apply to both self-attention and cross-attention
                        attn_processor_dict[f'blocks.{layer_id}.attn{attn_id}.processor'] = PartCrafterAttnProcessor()
                        modified_attn_processor.append(f'blocks.{layer_id}.attn{attn_id}.processor')
                    else:
                        attn_processor_dict[f'blocks.{layer_id}.attn{attn_id}.processor'] = TripoSGAttnProcessor2_0()
            self.set_attn_processor(attn_processor_dict)
            # logger.info(f"Modified {modified_attn_processor} to PartCrafterAttnProcessor")
        self.enable_part_embedding = enable_part_embedding
        self.max_num_parts = max_num_parts

        self.enable_point_prompt = enable_point_prompt
        self.point_prompt_fourier_dim = point_prompt_fourier_dim

        self.enable_box_prompt = enable_box_prompt
        self.box_prompt_fourier_dim = box_prompt_fourier_dim

        self.enable_mask_prompt = enable_mask_prompt

        self.enable_3dbox_prompt = enable_3dbox_prompt

        if add_module:
            self.add_modules()

    def add_modules(self):

        for block in self.blocks:
            if hasattr(block, "add_adaln_modules"):
                block.add_adaln_modules()

        if self.enable_part_embedding:
            self.part_embedding = nn.Embedding(self.max_num_parts, self.inner_dim)
            self.part_embedding.weight.data.normal_(mean=0.0, std=0.02)

        if self.enable_point_prompt:
            self.point_prompt_encoder = PointPromptEncoder(
                output_dim=self.prompt_dim,
                fourier_dim=self.point_prompt_fourier_dim
              )
            self.null_point_prompt_emb = nn.Embedding(1, self.prompt_dim)
            nn.init.normal_(self.null_point_prompt_emb.weight, std=0.02)

        if self.enable_box_prompt:
            self.box_prompt_encoder = BoxPromptEncoder(
                output_dim=self.prompt_dim,
                fourier_dim=self.box_prompt_fourier_dim
            )
            self.null_box_prompt_emb = nn.Embedding(1, self.prompt_dim)
            nn.init.normal_(self.null_box_prompt_emb.weight, std=0.02)

        if self.enable_mask_prompt:
            self.mask_prompt_encoder = MaskPromptEncoder(output_dim=self.prompt_dim)
            self.null_mask_prompt_emb = nn.Embedding(1, self.prompt_dim)
            nn.init.normal_(self.null_mask_prompt_emb.weight, std=0.02)

        if self.enable_3dbox_prompt:
            self.box3d_prompt_encoder = Box3DPromptEncoder(output_dim=self.prompt_dim)
            self.null_box3d_prompt_emb = nn.Embedding(1, self.prompt_dim)
            nn.init.normal_(self.null_box3d_prompt_emb.weight, std=0.02)

        self.proj_dim = 256
        prompt_nums = int(self.enable_point_prompt + self.enable_box_prompt + self.enable_mask_prompt + self.enable_3dbox_prompt)
        self.combiner_mlp = nn.Sequential(
            nn.Linear(self.prompt_dim*prompt_nums, self.adaln_cond_dim),
            nn.ReLU(),
            nn.Linear(self.adaln_cond_dim, self.adaln_cond_dim)
        )

        self.visual_bridge = PartVisualBridge(part_dim=self.prompt_dim,image_dim=self.cross_attention_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.sinkhorn = SinkhornWithDustbin(init_value=1.0)

    def _set_gradient_checkpointing(
        self,
        enable: bool = False,
        gradient_checkpointing_func: Optional[Callable] = None,
    ):
        # TODO: implement gradient checkpointing
        self.gradient_checkpointing = enable

    def _set_time_proj(
        self,
        time_embedding_type: str,
        inner_dim: int,
        flip_sin_to_cos: bool,
        freq_shift: float,
        time_embedding_dim: int,
    ) -> Tuple[int, int]:
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or inner_dim * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(
                    f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}."
                )
            self.time_embed = GaussianFourierProjection(
                time_embed_dim // 2,
                set_W_to_weight=False,
                log=False,
                flip_sin_to_cos=flip_sin_to_cos,
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or inner_dim * 4

            self.time_embed = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
            timestep_input_dim = inner_dim
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        return time_embed_dim, timestep_input_dim

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedTripoSGAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError(
                    "`fuse_qkv_projections()` is not supported for models having added KV projections."
                )

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedTripoSGAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(TripoSGAttnProcessor2_0())

    def forward(
        self,
        hidden_states: Optional[torch.Tensor], # [Total_Parts, T, D] (注意这里 batch 已经是 total parts)
        timestep: Union[int, float, torch.LongTensor],
        encoder_hidden_states: Optional[torch.Tensor] = None, # 🟢 [Total_Parts, 257, 1024] (DINO, 已扩展)
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,

        # --- Prompts ---
        point_prompt_xy: Optional[torch.Tensor] = None,
        box_prompt_xyxy: Optional[torch.Tensor] = None,
        mask_prompt: Optional[torch.Tensor] = None,

        # --- Dropout Masks ---
        keep_point: Optional[torch.Tensor] = None,
        keep_box: Optional[torch.Tensor] = None,
        keep_mask: Optional[torch.Tensor] = None,

        box3d_prompt: Optional[torch.Tensor] = None,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
            num_parts = attention_kwargs["num_parts"] # 必须有，用于 Locator 交互
        else:
            raise ValueError("attention_kwargs with 'num_parts' is required.")

        total_parts_batch = hidden_states.shape[0]
        device = hidden_states.device

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        _, T, _ = hidden_states.shape
        temb = self.time_embed(timestep).to(hidden_states.dtype)
        temb = self.time_proj(temb)
        temb = temb.unsqueeze(dim=1)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = torch.cat([temb, hidden_states], dim=1)

        if self.enable_part_embedding:
            if isinstance(num_parts, torch.Tensor):
                part_embeddings = []
                for num_part in num_parts:
                    part_embedding = self.part_embedding(torch.arange(num_part, device=device))
                    part_embeddings.append(part_embedding)
                part_emb_base = torch.cat(part_embeddings, dim=0)
            elif isinstance(num_parts, int):
                part_emb_base = self.part_embedding(torch.arange(total_parts_batch, device=device))

            # hidden_states = hidden_states + part_emb_base.unsqueeze(dim=1)
            hidden_states = torch.cat([part_emb_base.unsqueeze(1), hidden_states], dim=1)

        dino_patches = encoder_hidden_states[:, 1:, :]


        # --- Point Encoding ---
        final_point_emb = None
        if self.enable_point_prompt:
            final_point_emb = self.null_point_prompt_emb.weight.expand(total_parts_batch, -1)

            target_xy = point_prompt_xy
            if target_xy is not None:
                real_point_emb = self.point_prompt_encoder(target_xy)
                if keep_point is not None:
                    mask_p = keep_point.view(-1, 1)
                    final_point_emb = torch.where(mask_p, real_point_emb, final_point_emb)
                else:
                    final_point_emb = real_point_emb

        # --- Box Encoding ---
        final_box_emb = None
        if self.enable_box_prompt:
            final_box_emb = self.null_box_prompt_emb.weight.expand(total_parts_batch, -1)
            if box_prompt_xyxy is not None:
                real_box_emb = self.box_prompt_encoder(box_prompt_xyxy)
                mask_b = keep_box.view(-1, 1)
                final_box_emb = torch.where(mask_b, real_box_emb, final_box_emb)

        # --- Mask Encoding ---
        final_mask_emb = None
        if self.enable_mask_prompt:
            final_mask_emb = self.null_mask_prompt_emb.weight.expand(total_parts_batch, -1)
            if mask_prompt is not None:
                real_mask_emb = self.mask_prompt_encoder(mask_prompt)
                mask_m = keep_mask.view(-1, 1)
                final_mask_emb = torch.where(mask_m, real_mask_emb, final_mask_emb)

        # --- 3D Box Encoding ---
        final_box3d_emb = None
        if self.enable_3dbox_prompt:
            final_box3d_emb = self.null_box3d_prompt_emb.weight.expand(total_parts_batch, -1)
            if box3d_prompt is not None:
                real_box3d_emb = self.box3d_prompt_encoder(box3d_prompt)
                # box3d_m = keep_mask.view(-1, 1)
                final_box3d_emb = real_box3d_emb


        feature_list = []
        if self.enable_point_prompt: feature_list.append(final_point_emb)
        if self.enable_box_prompt:   feature_list.append(final_box_emb)
        if self.enable_mask_prompt:  feature_list.append(final_mask_emb)
        if self.enable_3dbox_prompt:  feature_list.append(final_box3d_emb)

        combined_feature_raw = torch.cat(feature_list, dim=-1)
        final_adaln_input = self.combiner_mlp(combined_feature_raw)

        aux_loss = torch.tensor(0.0, device=device)

        if self.training and hasattr(self, 'sinkhorn'):
            if self.enable_point_prompt and keep_point is not None:
                student_feature = final_point_emb

            # Teachers
            mask_teacher_feat = None
            box_teacher_feat = None
            if self.enable_mask_prompt and mask_prompt is not None:
                mask_teacher_feat = real_mask_emb.detach()
            if self.enable_box_prompt and box_prompt_xyxy is not None:
                box_teacher_feat = real_box_emb.detach()

            # Bridge Alignment
            if (mask_teacher_feat is not None or box_teacher_feat is not None):
                feat_student = self.visual_bridge(student_feature, dino_patches)
                feat_student = F.normalize(feat_student, dim=-1)

                # Teacher Logits
                final_logits = 0.0
                count = 0

                if box_teacher_feat is not None:
                    feat_box = self.visual_bridge(box_teacher_feat, dino_patches)
                    feat_box = F.normalize(feat_box, dim=-1)
                    final_logits += 0.5 * (self.logit_scale.exp() * feat_student @ feat_box.transpose(-2, -1))
                    count += 1

                if mask_teacher_feat is not None:
                    feat_mask = self.visual_bridge(mask_teacher_feat, dino_patches)
                    feat_mask = F.normalize(feat_mask, dim=-1)
                    final_logits += 0.5 * (self.logit_scale.exp() * feat_student @ feat_mask.transpose(-2, -1))
                    count += 1

                if count == 1: final_logits *= 2.0


                # Sinkhorn Loss
                out = self.sinkhorn(final_logits)
                log_P = torch.log(out["P_match"] + 1e-8)
                labels = torch.arange(log_P.shape[0], device=log_P.device)
                aux_loss = F.nll_loss(log_P, labels)

        # 🟢 汇总
        all_loss = aux_loss

        # prepare negative encoder_hidden_states
        negative_encoder_hidden_states = torch.zeros_like(encoder_hidden_states) if encoder_hidden_states is not None else None

        skips = []
        for layer, block in enumerate(self.blocks):
            skip = None if layer <= self.config.num_layers // 2 else skips.pop()
            if (
                (not self.enable_local_cross_attn)
                and len(self.global_attn_block_ids) > 0
                and (layer not in self.global_attn_block_ids)
            ):
                # If in non-global attention block and disable local cross attention, use negative encoder_hidden_states
                # Do not inject control signal into non-global attention block
                input_encoder_hidden_states = negative_encoder_hidden_states
            elif (
                (not self.enable_global_cross_attn)
                and len(self.global_attn_block_ids) > 0
                and (layer in self.global_attn_block_ids)
            ):
                # If in global attention block and disable global cross attention, use negative encoder_hidden_states
                # Do not inject control signal into global attention block
                input_encoder_hidden_states = negative_encoder_hidden_states
            else:
                input_encoder_hidden_states = encoder_hidden_states

            if len(self.global_attn_block_ids) > 0 and (layer in self.global_attn_block_ids):
                # Inject control signal into global attention block
                input_attention_kwargs = attention_kwargs
            else:
                input_attention_kwargs = None

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    input_encoder_hidden_states,
                    temb,
                    final_adaln_input,
                    image_rotary_emb,
                    skip,
                    input_attention_kwargs,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=input_encoder_hidden_states,
                    temb=temb,
                    part_condition = final_adaln_input,
                    image_rotary_emb=image_rotary_emb,
                    skip=skip,
                    attention_kwargs=input_attention_kwargs,
                )  # (N, T+1, D)

            if layer < self.config.num_layers // 2:
                skips.append(hidden_states)

        # final layer
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states[:, -T:]  # (N, T, D)
        hidden_states = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer1DModelOutput(sample=hidden_states,aux_loss=all_loss)

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(
        self, chunk_size: Optional[int] = None, dim: int = 0
    ) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(
            module: torch.nn.Module, chunk_size: int, dim: int
        ):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.disable_forward_chunking
    def disable_forward_chunking(self):
        def fn_recursive_feed_forward(
            module: torch.nn.Module, chunk_size: int, dim: int
        ):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, None, 0)
