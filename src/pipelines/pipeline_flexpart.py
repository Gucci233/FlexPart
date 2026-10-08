import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import PIL.Image
import torch
import trimesh
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    BitImageProcessor,
    Dinov2Model,
)
from ..utils.inference_utils import hierarchical_extract_geometry

from ..models.autoencoders import TripoSGVAEModel
from ..models.transformers.flexpart_transformer import FlexPartDiTModel
from .pipeline_output import PipelineOutput
from .pipeline_utils import TransformerDiffusionMixin
from ..utils.postprocess import postprocess_mesh,filter_mesh
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class FlexPartPipeline(DiffusionPipeline, TransformerDiffusionMixin):
    """
    Pipeline for image to 3D part-level object generation.
    """

    def __init__(
        self,
        vae: TripoSGVAEModel,
        transformer: FlexPartDiTModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_encoder_dinov2: Dinov2Model,
        feature_extractor_dinov2: BitImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder_dinov2=image_encoder_dinov2,
            feature_extractor_dinov2=feature_extractor_dinov2,
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def decode_progressive(self):
        return self._decode_progressive

    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder_dinov2.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor_dinov2(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder_dinov2(image).last_hidden_state
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        uncond_image_embeds = torch.zeros_like(image_embeds)

        return image_embeds, uncond_image_embeds

    def prepare_latents(
        self,
        batch_size,
        num_tokens,
        num_channels_latents,
        dtype,
        device,
        generator,
        latents: Optional[torch.Tensor] = None,
    ):
        shape = (batch_size, num_tokens, num_channels_latents)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return noise

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        num_inference_steps: int = 50,
        num_tokens: int = 2048,
        timesteps: List[int] = None,
        guidance_scale: float = 0.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        bounds: Union[Tuple[float], List[float], float] = (-1.005, -1.005, -1.005, 1.005, 1.005, 1.005),
        dense_octree_depth: int = 8,
        hierarchical_octree_depth: int = 9,
        max_num_expanded_coords: int = 1e8,
        flash_octree_depth: int = 9,
        use_flash_decoder: bool = True,
        return_dict: bool = True,
        postprocess: bool = False,
        points_prompt : Optional[torch.Tensor] = None,
        boxes_prompt : Optional[torch.Tensor] = None,
        masks_prompt : Optional[torch.Tensor] = None,
        valid_boxes :Optional[torch.Tensor] = None,
        valid_points :Optional[torch.Tensor] = None,
        valid_masks :Optional[torch.Tensor] = None,

        box3d_prompt : Optional[torch.Tensor] = None,
    ):
        # 1. Define call parameters
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # self.do_classifier_free_guidance=False
        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        elif isinstance(image, torch.Tensor):
            batch_size = image.shape[0]
        else:
            raise ValueError("Invalid input type for image")

        device = self._execution_device
        dtype = self.image_encoder_dinov2.dtype

        # 3. Encode condition
        image_embeds, negative_image_embeds = self.encode_image(
            image, device, num_images_per_prompt
        )
        if points_prompt:
            points_prompt = [torch.FloatTensor(point).to(device=device,dtype=dtype) for point in points_prompt]
            points_prompt = torch.stack(points_prompt,dim=0)
        if boxes_prompt:
            boxes_prompt = [torch.FloatTensor(box).to(device=device,dtype=dtype) for box in boxes_prompt]
            boxes_prompt = torch.stack(boxes_prompt,dim=0)
        if masks_prompt:
            masks_prompt = [torch.FloatTensor(mask).to(device=device,dtype=dtype) for mask in masks_prompt]
            masks_prompt = torch.stack(masks_prompt,dim=0)
        # if box3d_prompt:
        #     box3d_prompt = [torch.FloatTensor(box3d).to(device=device,dtype=dtype) for box3d in box3d_prompt]
        #     box3d_prompt = torch.stack(box3d_prompt,dim=0)


        if valid_points:
            valid_points = [torch.tensor(valid).to(device=device,dtype=torch.bool) for valid in valid_points]
            valid_points = torch.stack(valid_points,dim=0)
        if valid_boxes:
            valid_boxes = [torch.tensor(valid).to(device=device,dtype=torch.bool) for valid in valid_boxes]
            valid_boxes = torch.stack(valid_boxes,dim=0)
        if valid_masks:
            valid_masks = [torch.tensor(valid).to(device=device,dtype=torch.bool) for valid in valid_masks]
            valid_masks = torch.stack(valid_masks,dim=0)

        if self.do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)
            points_prompt = torch.cat([points_prompt]*2, dim=0) if points_prompt is not None else None
            boxes_prompt = torch.cat([boxes_prompt]*2, dim=0) if boxes_prompt is not None else None
            masks_prompt = torch.cat([masks_prompt]*2, dim=0) if masks_prompt is not None else None
            valid_points = torch.cat([valid_points]*2, dim=0) if valid_points is not None else None
            valid_boxes = torch.cat([valid_boxes]*2, dim=0) if valid_boxes is not None else None
            valid_masks = torch.cat([valid_masks]*2, dim=0) if valid_masks is not None else None
            # box3d_prompt = torch.cat([box3d_prompt]*2, dim=0) if valid_masks is not None else None

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_tokens,
            num_channels_latents,
            image_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        self.set_progress_bar_config(
            desc="Denoising",
            ncols=125,
            disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
        )
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    point_prompt_xy = points_prompt,
                    box_prompt_xyxy = boxes_prompt,
                    mask_prompt = masks_prompt,

                   keep_box = valid_boxes,
                   keep_point = valid_points,
                   keep_mask = valid_masks,
                   box3d_prompt = box3d_prompt
                )[0].to(dtype)

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_image - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    image_embeds_1 = callback_outputs.pop(
                        "image_embeds_1", image_embeds_1
                    )
                    negative_image_embeds_1 = callback_outputs.pop(
                        "negative_image_embeds_1", negative_image_embeds_1
                    )
                    image_embeds_2 = callback_outputs.pop(
                        "image_embeds_2", image_embeds_2
                    )
                    negative_image_embeds_2 = callback_outputs.pop(
                        "negative_image_embeds_2", negative_image_embeds_2
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()


        # 7. decoder mesh
        self.vae.set_flash_decoder()
        output, meshes = [], []
        self.set_progress_bar_config(
            desc="Decoding",
            ncols=125,
            disable=self._progress_bar_config['disable'] if hasattr(self, '_progress_bar_config') else False,
        )
        with self.progress_bar(total=batch_size) as progress_bar:
            for i in range(batch_size):
                geometric_func = lambda x: self.vae.decode(latents[i].unsqueeze(0), sampled_points=x).sample
                try:
                    mesh_v_f = hierarchical_extract_geometry(
                        geometric_func,
                        device,
                        dtype=latents.dtype,
                        bounds=bounds,
                        dense_octree_depth=dense_octree_depth,
                        hierarchical_octree_depth=hierarchical_octree_depth,
                        max_num_expanded_coords=max_num_expanded_coords,
                        # verbose=True
                    )
                    mesh = trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1].astype(np.float32))
                    mesh = filter_mesh(mesh)
                    if postprocess:
                        mesh = postprocess_mesh(mesh)

                except:
                    mesh_v_f = None
                    mesh = None
                output.append(mesh_v_f)
                meshes.append(mesh)
                progress_bar.update()

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (output, meshes)

        return PipelineOutput(samples=output, meshes=meshes)

