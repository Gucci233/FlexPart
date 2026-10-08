# Copyright (c) 2025 Yuchen Lin

# This code is based on TripoSG (https://github.com/VAST-AI-Research/TripoSG). Below is the statement from the original repository:

# This code is based on Tencent HunyuanDiT (https://huggingface.co/Tencent-Hunyuan/HunyuanDiT),
# which is licensed under the Tencent Hunyuan Community License Agreement.
# Portions of this code are copied or adapted from HunyuanDiT.
# See the original license below:

# ---- Start of Tencent Hunyuan Community License Agreement ----

# TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT
# Tencent Hunyuan DiT Release Date: 14 May 2024
# THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION AND IS EXPRESSLY LIMITED TO THE TERRITORY, AS DEFINED BELOW.
# By clicking to agree or by using, reproducing, modifying, distributing, performing or displaying any portion or element of the Tencent Hunyuan Works, including via any Hosted Service, You will be deemed to have recognized and accepted the content of this Agreement, which is effective immediately.
# 1.	DEFINITIONS.
# a.	“Acceptable Use Policy” shall mean the policy made available by Tencent as set forth in the Exhibit A.
# b.	“Agreement” shall mean the terms and conditions for use, reproduction, distribution, modification, performance and displaying of Tencent Hunyuan Works or any portion or element thereof set forth herein.
# c.	“Documentation” shall mean the specifications, manuals and documentation for Tencent Hunyuan made publicly available by Tencent.
# d.	“Hosted Service” shall mean a hosted service offered via an application programming interface (API), web access, or any other electronic or remote means.
# e.	“Licensee,” “You” or “Your” shall mean a natural person or legal entity exercising the rights granted by this Agreement and/or using the Tencent Hunyuan Works for any purpose and in any field of use.
# f.	“Materials” shall mean, collectively, Tencent’s proprietary Tencent Hunyuan and Documentation (and any portion thereof) as made available by Tencent under this Agreement.
# g.	“Model Derivatives” shall mean all: (i) modifications to Tencent Hunyuan or any Model Derivative of Tencent Hunyuan; (ii) works based on Tencent Hunyuan or any Model Derivative of Tencent Hunyuan; or (iii) any other machine learning model which is created by transfer of patterns of the weights, parameters, operations, or Output of Tencent Hunyuan or any Model Derivative of Tencent Hunyuan, to that model in order to cause that model to perform similarly to Tencent Hunyuan or a Model Derivative of Tencent Hunyuan, including distillation methods, methods that use intermediate data representations, or methods based on the generation of synthetic data Outputs by Tencent Hunyuan or a Model Derivative of Tencent Hunyuan for training that model. For clarity, Outputs by themselves are not deemed Model Derivatives.
# h.	“Output” shall mean the information and/or content output of Tencent Hunyuan or a Model Derivative that results from operating or otherwise using Tencent Hunyuan or a Model Derivative, including via a Hosted Service.
# i.	“Tencent,” “We” or “Us” shall mean THL A29 Limited.
# j.	“Tencent Hunyuan” shall mean the large language models, text/image/video/audio/3D generation models, and multimodal large language models and their software and algorithms, including trained model weights, parameters (including optimizer states), machine-learning model code, inference-enabling code, training-enabling code, fine-tuning enabling code and other elements of the foregoing made publicly available by Us, including, without limitation to, Tencent Hunyuan DiT released at https://huggingface.co/Tencent-Hunyuan/HunyuanDiT.
# k.	“Tencent Hunyuan Works” shall mean: (i) the Materials; (ii) Model Derivatives; and (iii) all derivative works thereof.
# l.	“Territory” shall mean the worldwide territory, excluding the territory of the European Union.
# m.	“Third Party” or “Third Parties” shall mean individuals or legal entities that are not under common control with Us or You.
# n.	“including” shall mean including but not limited to.
# 2.	GRANT OF RIGHTS.
# We grant You, for the Territory only, a non-exclusive, non-transferable and royalty-free limited license under Tencent’s intellectual property or other rights owned by Us embodied in or utilized by the Materials to use, reproduce, distribute, create derivative works of (including Model Derivatives), and make modifications to the Materials, only in accordance with the terms of this Agreement and the Acceptable Use Policy, and You must not violate (or encourage or permit anyone else to violate) any term of this Agreement or the Acceptable Use Policy.
# 3.	DISTRIBUTION.
# You may, subject to Your compliance with this Agreement, distribute or make available to Third Parties the Tencent Hunyuan Works, exclusively in the Territory, provided that You meet all of the following conditions:
# a.	You must provide all such Third Party recipients of the Tencent Hunyuan Works or products or services using them a copy of this Agreement;
# b.	You must cause any modified files to carry prominent notices stating that You changed the files;
# c.	You are encouraged to: (i) publish at least one technology introduction blogpost or one public statement expressing Your experience of using the Tencent Hunyuan Works; and (ii) mark the products or services developed by using the Tencent Hunyuan Works to indicate that the product/service is “Powered by Tencent Hunyuan”; and
# d.	All distributions to Third Parties (other than through a Hosted Service) must be accompanied by a “Notice” text file that contains the following notice: “Tencent Hunyuan is licensed under the Tencent Hunyuan Community License Agreement, Copyright © 2024 Tencent. All Rights Reserved. The trademark rights of “Tencent Hunyuan” are owned by Tencent or its affiliate.”
# You may add Your own copyright statement to Your modifications and, except as set forth in this Section and in Section 5, may provide additional or different license terms and conditions for use, reproduction, or distribution of Your modifications, or for any such Model Derivatives as a whole, provided Your use, reproduction, modification, distribution, performance and display of the work otherwise complies with the terms and conditions of this Agreement (including as regards the Territory). If You receive Tencent Hunyuan Works from a Licensee as part of an integrated end user product, then this Section 3 of this Agreement will not apply to You.
# 4.	ADDITIONAL COMMERCIAL TERMS.
# If, on the Tencent Hunyuan version release date, the monthly active users of all products or services made available by or for Licensee is greater than 100 million monthly active users in the preceding calendar month, You must request a license from Tencent, which Tencent may grant to You in its sole discretion, and You are not authorized to exercise any of the rights under this Agreement unless or until Tencent otherwise expressly grants You such rights.
# 5.	RULES OF USE.
# a.	Your use of the Tencent Hunyuan Works must comply with applicable laws and regulations (including trade compliance laws and regulations) and adhere to the Acceptable Use Policy for the Tencent Hunyuan Works, which is hereby incorporated by reference into this Agreement. You must include the use restrictions referenced in these Sections 5(a) and 5(b) as an enforceable provision in any agreement (e.g., license agreement, terms of use, etc.) governing the use and/or distribution of Tencent Hunyuan Works and You must provide notice to subsequent users to whom You distribute that Tencent Hunyuan Works are subject to the use restrictions in these Sections 5(a) and 5(b).
# b.	You must not use the Tencent Hunyuan Works or any Output or results of the Tencent Hunyuan Works to improve any other large language model (other than Tencent Hunyuan or Model Derivatives thereof).
# c.	You must not use, reproduce, modify, distribute, or display the Tencent Hunyuan Works, Output or results of the Tencent Hunyuan Works outside the Territory. Any such use outside the Territory is unlicensed and unauthorized under this Agreement.
# 6.	INTELLECTUAL PROPERTY.
# a.	Subject to Tencent’s ownership of Tencent Hunyuan Works made by or for Tencent and intellectual property rights therein, conditioned upon Your compliance with the terms and conditions of this Agreement, as between You and Tencent, You will be the owner of any derivative works and modifications of the Materials and any Model Derivatives that are made by or for You.
# b.	No trademark licenses are granted under this Agreement, and in connection with the Tencent Hunyuan Works, Licensee may not use any name or mark owned by or associated with Tencent or any of its affiliates, except as required for reasonable and customary use in describing and distributing the Tencent Hunyuan Works. Tencent hereby grants You a license to use “Tencent Hunyuan” (the “Mark”) in the Territory solely as required to comply with the provisions of Section 3(c), provided that You comply with any applicable laws related to trademark protection. All goodwill arising out of Your use of the Mark will inure to the benefit of Tencent.
# c.	If You commence a lawsuit or other proceedings (including a cross-claim or counterclaim in a lawsuit) against Us or any person or entity alleging that the Materials or any Output, or any portion of any of the foregoing, infringe any intellectual property or other right owned or licensable by You, then all licenses granted to You under this Agreement shall terminate as of the date such lawsuit or other proceeding is filed. You will defend, indemnify and hold harmless Us from and against any claim by any Third Party arising out of or related to Your or the Third Party’s use or distribution of the Tencent Hunyuan Works.
# d.	Tencent claims no rights in Outputs You generate. You and Your users are solely responsible for Outputs and their subsequent uses.
# 7.	DISCLAIMERS OF WARRANTY AND LIMITATIONS OF LIABILITY.
# a.	We are not obligated to support, update, provide training for, or develop any further version of the Tencent Hunyuan Works or to grant any license thereto.
# b.	UNLESS AND ONLY TO THE EXTENT REQUIRED BY APPLICABLE LAW, THE TENCENT HUNYUAN WORKS AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED “AS IS” WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES OF ANY KIND INCLUDING ANY WARRANTIES OF TITLE, MERCHANTABILITY, NONINFRINGEMENT, COURSE OF DEALING, USAGE OF TRADE, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING, REPRODUCING, MODIFYING, PERFORMING, DISPLAYING OR DISTRIBUTING ANY OF THE TENCENT HUNYUAN WORKS OR OUTPUTS AND ASSUME ANY AND ALL RISKS ASSOCIATED WITH YOUR OR A THIRD PARTY’S USE OR DISTRIBUTION OF ANY OF THE TENCENT HUNYUAN WORKS OR OUTPUTS AND YOUR EXERCISE OF RIGHTS AND PERMISSIONS UNDER THIS AGREEMENT.
# c.	TO THE FULLEST EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL TENCENT OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, FOR ANY DAMAGES, INCLUDING ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL, EXEMPLARY, CONSEQUENTIAL OR PUNITIVE DAMAGES, OR LOST PROFITS OF ANY KIND ARISING FROM THIS AGREEMENT OR RELATED TO ANY OF THE TENCENT HUNYUAN WORKS OR OUTPUTS, EVEN IF TENCENT OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF ANY OF THE FOREGOING.
# 8.	SURVIVAL AND TERMINATION.
# a.	The term of this Agreement shall commence upon Your acceptance of this Agreement or access to the Materials and will continue in full force and effect until terminated in accordance with the terms and conditions herein.
# b.	We may terminate this Agreement if You breach any of the terms or conditions of this Agreement. Upon termination of this Agreement, You must promptly delete and cease use of the Tencent Hunyuan Works. Sections 6(a), 6(c), 7 and 9 shall survive the termination of this Agreement.
# 9.	GOVERNING LAW AND JURISDICTION.
# a.	This Agreement and any dispute arising out of or relating to it will be governed by the laws of the Hong Kong Special Administrative Region of the People’s Republic of China, without regard to conflict of law principles, and the UN Convention on Contracts for the International Sale of Goods does not apply to this Agreement.
# b.	Exclusive jurisdiction and venue for any dispute arising out of or relating to this Agreement will be a court of competent jurisdiction in the Hong Kong Special Administrative Region of the People’s Republic of China, and Tencent and Licensee consent to the exclusive jurisdiction of such court with respect to any such dispute.
#
# EXHIBIT A
# ACCEPTABLE USE POLICY

# Tencent reserves the right to update this Acceptable Use Policy from time to time.
# Last modified: [insert date]

# Tencent endeavors to promote safe and fair use of its tools and features, including Tencent Hunyuan. You agree not to use Tencent Hunyuan or Model Derivatives:
# 1.	Outside the Territory;
# 2.	In any way that violates any applicable national, federal, state, local, international or any other law or regulation;
# 3.	To harm Yourself or others;
# 4.	To repurpose or distribute output from Tencent Hunyuan or any Model Derivatives to harm Yourself or others;
# 5.	To override or circumvent the safety guardrails and safeguards We have put in place;
# 6.	For the purpose of exploiting, harming or attempting to exploit or harm minors in any way;
# 7.	To generate or disseminate verifiably false information and/or content with the purpose of harming others or influencing elections;
# 8.	To generate or facilitate false online engagement, including fake reviews and other means of fake online engagement;
# 9.	To intentionally defame, disparage or otherwise harass others;
# 10.	To generate and/or disseminate malware (including ransomware) or any other content to be used for the purpose of harming electronic systems;
# 11.	To generate or disseminate personal identifiable information with the purpose of harming others;
# 12.	To generate or disseminate information (including images, code, posts, articles), and place the information in any public context (including –through the use of bot generated tweets), without expressly and conspicuously identifying that the information and/or content is machine generated;
# 13.	To impersonate another individual without consent, authorization, or legal right;
# 14.	To make high-stakes automated decisions in domains that affect an individual’s safety, rights or wellbeing (e.g., law enforcement, migration, medicine/health, management of critical infrastructure, safety components of products, essential services, credit, employment, housing, education, social scoring, or insurance);
# 15.	In a manner that violates or disrespects the social ethics and moral standards of other countries or regions;
# 16.	To perform, facilitate, threaten, incite, plan, promote or encourage violent extremism or terrorism;
# 17.	For any use intended to discriminate against or harm individuals or groups based on protected characteristics or categories, online or offline social behavior or known or predicted personal or personality characteristics;
# 18.	To intentionally exploit any of the vulnerabilities of a specific group of persons based on their age, social, physical or mental characteristics, in order to materially distort the behavior of a person pertaining to that group in a manner that causes or is likely to cause that person or another person physical or psychological harm;
# 19.	For military purposes;
# 20.	To engage in the unauthorized or unlicensed practice of any profession including, but not limited to, financial, legal, medical/health, or other professional practices.

# ---- End of Tencent Hunyuan Community License Agreement ----

# Please note that the use of this code is subject to the terms and conditions
# of the Tencent Hunyuan Community License Agreement, including the Acceptable Use Policy.

# Extracted from PartCrafter for the unconditioned VAE blocks.
# Source: https://github.com/wgsxm/PartCrafter/blob/main/src/models/transformers/partcrafter_transformer.py

from typing import Any, Dict, Optional
import torch
from torch import nn
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils.torch_utils import maybe_allow_in_graph
from ..attention_processor import TripoSGAttnProcessor2_0

@maybe_allow_in_graph
class DiTBlock(nn.Module):
    r"""
    Transformer block used in Hunyuan-DiT model (https://github.com/Tencent/HunyuanDiT). Allow skip connection and
    QKNorm

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of headsto use for multi-head attention.
        cross_attention_dim (`int`,*optional*):
            The size of the encoder_hidden_states vector for cross attention.
        dropout(`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        activation_fn (`str`,*optional*, defaults to `"geglu"`):
            Activation function to be used in feed-forward. .
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            A small constant added to the denominator in normalization layers to prevent division by zero.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*):
            The size of the hidden layer in the feed-forward block. Defaults to `None`.
        ff_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the feed-forward block.
        skip (`bool`, *optional*, defaults to `False`):
            Whether to use skip connection. Defaults to `False` for down-blocks and mid-blocks.
        qk_norm (`bool`, *optional*, defaults to `True`):
            Whether to use normalization in QK calculation. Defaults to `True`.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        use_self_attention: bool = True,
        self_attention_norm_type: Optional[str] = None,
        use_cross_attention: bool = True, # ada layer norm
        cross_attention_dim: Optional[int] = None,
        cross_attention_norm_type: Optional[str] = "fp32_layer_norm",
        dropout=0.0,
        activation_fn: str = "gelu",
        norm_type: str = "fp32_layer_norm",  # TODO
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,  # int(dim * 4) if None
        ff_bias: bool = True,
        skip: bool = False,
        skip_concat_front: bool = False,  # [x, skip] or [skip, x]
        skip_norm_last: bool = False,  # this is an error
        qk_norm: bool = True,
        qkv_bias: bool = True,
    ):
        super().__init__()

        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention
        self.skip_concat_front = skip_concat_front
        self.skip_norm_last = skip_norm_last
        # Define 3 blocks. Each block has its own normalization layer.
        # NOTE: when new version comes, check norm2 and norm 3
        # 1. Self-Attn
        if use_self_attention:
            if (
                self_attention_norm_type == "fp32_layer_norm"
                or self_attention_norm_type is None
            ):
                self.norm1 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)
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

        # 2. Cross-Attn
        if use_cross_attention:
            assert cross_attention_dim is not None

            self.norm2 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

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

        # 3. Feed-forward
        self.norm3 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.ff = FeedForward(
            dim,
            dropout=dropout,  ### 0.0
            activation_fn=activation_fn,  ### approx GeLU
            final_dropout=final_dropout,  ### 0.0
            inner_dim=ff_inner_dim,  ### int(dim * mlp_ratio)
            bias=ff_bias,
        )

        # 4. Skip Connection
        if skip:
            self.skip_norm = FP32LayerNorm(dim, norm_eps, elementwise_affine=True)
            self.skip_linear = nn.Linear(2 * dim, dim)
        else:
            self.skip_linear = None

        # let chunk size default to None
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        skip: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        # Prepare attention kwargs
        attention_kwargs = attention_kwargs or {}

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Long Skip Connection
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
                # don't do this
                hidden_states = self.skip_linear(cat)
                hidden_states = self.skip_norm(hidden_states)
            else:
                cat = self.skip_norm(cat)
                hidden_states = self.skip_linear(cat)

        # 1. Self-Attention
        if self.use_self_attention:
            norm_hidden_states = self.norm1(hidden_states)
            attn_output = self.attn1(
                norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **attention_kwargs,
            )
            hidden_states = hidden_states + attn_output

        # 2. Cross-Attention
        if self.use_cross_attention:
            hidden_states = hidden_states + self.attn2(
                self.norm2(hidden_states),
                encoder_hidden_states=encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **attention_kwargs,
            )

        # FFN Layer ### TODO: switch norm2 and norm3 in the state dict
        mlp_inputs = self.norm3(hidden_states)
        hidden_states = hidden_states + self.ff(mlp_inputs)

        return hidden_states
