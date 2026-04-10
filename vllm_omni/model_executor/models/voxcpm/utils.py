from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import tempfile
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm
from vllm.config import VllmConfig
from vllm.logger import init_logger

from .voxcpm_stage_wrappers import _DirectVoxCPMAudioVAE, _DirectVoxCPMLatentGenerator

logger = init_logger(__name__)


def _device_to_string(device: torch.device) -> str:
    if device.index is None:
        return device.type
    return f"{device.type}:{device.index}"


def _normalize_dtype_name(dtype: Any) -> str | None:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        mapping = {
            torch.bfloat16: "bfloat16",
            torch.float16: "float16",
            torch.float32: "float32",
        }
        return mapping.get(dtype, str(dtype).removeprefix("torch."))
    dtype_str = str(dtype)
    return dtype_str.removeprefix("torch.")


def _resolve_runtime_device(vllm_config: VllmConfig) -> torch.device:
    try:
        from vllm_omni.platforms import current_omni_platform

        return current_omni_platform.get_torch_device()
    except Exception:
        pass

    device = getattr(getattr(vllm_config, "device_config", None), "device", None)
    if isinstance(device, torch.device):
        return device
    if device:
        return torch.device(device)
    return torch.device("cpu")


def _prepare_runtime_model_dir(
    model_path: str | Path,
    *,
    target_device: torch.device,
    target_dtype: str | None,
) -> str:
    source_dir = Path(model_path)
    config_path = source_dir / "config.json"
    if not config_path.exists():
        return str(source_dir)

    config_text = config_path.read_text()
    config_dict = json.loads(config_text)
    desired_device = target_device.type
    desired_dtype = target_dtype or config_dict.get("dtype")

    if config_dict.get("device") == desired_device and config_dict.get("dtype") == desired_dtype:
        return str(source_dir)

    digest = sha256(f"{source_dir.resolve()}:{config_text}:{desired_device}:{desired_dtype}".encode()).hexdigest()[:16]
    runtime_dir = Path(tempfile.gettempdir()) / "vllm_omni_voxcpm_runtime" / digest
    runtime_dir.mkdir(parents=True, exist_ok=True)

    for entry in source_dir.iterdir():
        target = runtime_dir / entry.name
        if entry.name == "config.json" or target.exists():
            continue
        try:
            target.symlink_to(entry, target_is_directory=entry.is_dir())
        except OSError as exc:
            logger.warning(
                "Falling back to copying VoxCPM runtime artifact %s into %s because symlink creation failed: %s",
                entry,
                runtime_dir,
                exc,
            )
            if entry.is_dir():
                shutil.copytree(entry, target, dirs_exist_ok=True)
            else:
                shutil.copy2(entry, target)

    patched_config = dict(config_dict)
    patched_config["device"] = desired_device
    if desired_dtype is not None:
        patched_config["dtype"] = desired_dtype
    (runtime_dir / "config.json").write_text(json.dumps(patched_config, indent=2, sort_keys=True))
    return str(runtime_dir)


@contextmanager
def _force_cuda_available_for_npu(device: torch.device):
    if device.type != "npu":
        yield
        return

    with patch("torch.cuda.is_available", return_value=True):
        yield


def _is_torchcodec_load_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "torchcodec" in message or "load_with_torchcodec" in message


def _load_audio_with_soundfile(
    prompt_wav_path: str,
    *,
    sample_rate: int,
) -> torch.Tensor:
    try:
        import soundfile as sf
    except ImportError:
        raise

    audio_np, source_sr = sf.read(prompt_wav_path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(np.ascontiguousarray(audio_np.T))

    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)

    if int(source_sr) != int(sample_rate):
        try:
            import torchaudio
        except ImportError as exc:
            raise ImportError("torchaudio is required for resampling prompt audio.") from exc
        audio = torchaudio.functional.resample(audio, int(source_sr), int(sample_rate))

    return audio


def _build_prompt_cache_with_soundfile(model: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    if args:
        prompt_text = args[0]
        prompt_wav_path = args[1] if len(args) > 1 else kwargs.get("prompt_wav_path")
    else:
        prompt_text = kwargs.get("prompt_text")
        prompt_wav_path = kwargs.get("prompt_wav_path")

    if not prompt_text or not prompt_wav_path:
        raise ValueError("prompt_text and prompt_wav_path are required")

    audio = _load_audio_with_soundfile(prompt_wav_path, sample_rate=int(model.sample_rate))

    patch_len = model.patch_size * model.chunk_size
    if audio.size(1) % patch_len != 0:
        padding_size = patch_len - audio.size(1) % patch_len
        audio = torch.nn.functional.pad(audio, (padding_size, 0))

    audio_feat = model.audio_vae.encode(audio.to(model.device), model.sample_rate).cpu()
    audio_feat = audio_feat.view(
        model.audio_vae.latent_dim,
        -1,
        model.patch_size,
    ).permute(1, 2, 0)

    return {
        "prompt_text": prompt_text,
        "audio_feat": audio_feat,
    }


def _iter_voxcpm_src_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.environ.get("VLLM_OMNI_VOXCPM_CODE_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    repo_root = Path(__file__).resolve().parents[4]
    candidates.append(repo_root.parent / "VoxCPM" / "src")

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        unique_candidates.append(candidate)
    return unique_candidates


def _prepend_voxcpm_src(candidate: Path) -> None:
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


def _import_voxcpm_attrs(module_name: str, *attr_names: str) -> tuple[Any, ...]:
    last_exc: ImportError | None = None
    for candidate in _iter_voxcpm_src_candidates():
        if not candidate.exists():
            continue
        _prepend_voxcpm_src(candidate)
        try:
            module = importlib.import_module(module_name)
            return tuple(getattr(module, attr_name) for attr_name in attr_names)
        except ImportError as exc:
            last_exc = exc

    try:
        module = importlib.import_module(module_name)
        return tuple(getattr(module, attr_name) for attr_name in attr_names)
    except ImportError as exc:
        last_exc = exc

    raise ImportError(f"Failed to import {module_name}.") from last_exc


def _import_voxcpm_base_model_class():
    """Import upstream ``VoxCPMModel`` from ``VoxCPM/src/voxcpm`` (env, sibling tree, or pip)."""
    try:
        (VoxCPMModel,) = _import_voxcpm_attrs("voxcpm.model.voxcpm", "VoxCPMModel")
        return VoxCPMModel
    except ImportError as exc:
        raise ImportError(
            "Failed to import VoxCPMModel. Install the `voxcpm` package or set "
            "`VLLM_OMNI_VOXCPM_CODE_PATH` to the VoxCPM repository `src` directory "
            "(the parent of the `voxcpm` package that contains `model/` and `modules/`)."
        ) from exc


def _import_voxcpm_audio_vae_classes():
    try:
        return _import_voxcpm_attrs("voxcpm.modules.audiovae", "AudioVAE", "AudioVAEConfig")
    except ImportError as exc:
        raise ImportError(
            "Failed to import VoxCPM AudioVAE. Install the `voxcpm` package or set "
            "`VLLM_OMNI_VOXCPM_CODE_PATH` to the VoxCPM repository `src` directory."
        ) from exc


def _make_voxcpm_model_for_omni(base: type[Any]) -> type[Any]:
    """Subclass upstream VoxCPMModel: local ``_inference`` + ``latents_only`` prompt-cache generation."""

    from voxcpm.model.utils import get_dtype

    class VoxCPMModelForOmni(base):
        @torch.inference_mode()
        def build_prompt_cache(self, *args: Any, **kwargs: Any):
            try:
                return super().build_prompt_cache(*args, **kwargs)
            except (ImportError, ModuleNotFoundError) as exc:
                if not _is_torchcodec_load_error(exc):
                    raise
                return _build_prompt_cache_with_soundfile(self, *args, **kwargs)

        @torch.inference_mode()
        def _inference(
            self,
            text: torch.Tensor,
            text_mask: torch.Tensor,
            feat: torch.Tensor,
            feat_mask: torch.Tensor,
            min_len: int = 2,
            max_len: int = 2000,
            inference_timesteps: int = 10,
            cfg_value: float = 2.0,
            streaming: bool = False,
            streaming_prefix_len: int = 3,
        ) -> Generator[tuple[torch.Tensor, torch.Tensor | list[torch.Tensor]], None, None]:
            """Core inference loop aligned with upstream ``VoxCPMModel._inference``."""
            B, _, _, _ = feat.shape

            feat_embed = self.feat_encoder(feat)
            feat_embed = self.enc_to_lm_proj(feat_embed)

            if self.config.lm_config.use_mup:
                scale_emb = self.config.lm_config.scale_emb
            else:
                scale_emb = 1.0

            text_embed = self.base_lm.embed_tokens(text) * scale_emb
            combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed

            prefix_feat_cond = feat[:, -1, ...]
            pred_feat_seq: list[torch.Tensor] = []

            audio_patch_count = int(feat_mask.sum().item())
            if audio_patch_count > 0:
                context_len = min(streaming_prefix_len - 1, audio_patch_count)
                prompt_context_patches = list(feat[:, -context_len:, :, :].split(1, dim=1))
                pred_feat_seq = prompt_context_patches + pred_feat_seq

            enc_outputs, kv_cache_tuple = self.base_lm(
                inputs_embeds=combined_embed,
                is_causal=True,
            )
            self.base_lm.kv_cache.fill_caches(kv_cache_tuple)

            enc_outputs = self.fsq_layer(enc_outputs) * feat_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
            lm_hidden = enc_outputs[:, -1, :]

            residual_enc_outputs, residual_kv_cache_tuple = self.residual_lm(
                inputs_embeds=enc_outputs + feat_mask.unsqueeze(-1) * feat_embed,
                is_causal=True,
            )
            self.residual_lm.kv_cache.fill_caches(residual_kv_cache_tuple)
            residual_hidden = residual_enc_outputs[:, -1, :]

            for step_idx in tqdm(range(max_len)):
                dit_hidden_1 = self.lm_to_dit_proj(lm_hidden)
                dit_hidden_2 = self.res_to_dit_proj(residual_hidden)
                dit_hidden = dit_hidden_1 + dit_hidden_2

                pred_feat = self.feat_decoder(
                    mu=dit_hidden,
                    patch_size=self.patch_size,
                    cond=prefix_feat_cond.transpose(1, 2).contiguous(),
                    n_timesteps=inference_timesteps,
                    cfg_value=cfg_value,
                ).transpose(1, 2)

                curr_embed = self.feat_encoder(pred_feat.unsqueeze(1))
                curr_embed = self.enc_to_lm_proj(curr_embed)

                pred_feat_seq.append(pred_feat.unsqueeze(1))
                prefix_feat_cond = pred_feat

                if streaming:
                    pred_feat_chunk = torch.cat(pred_feat_seq[-streaming_prefix_len:], dim=1)
                    feat_pred = rearrange(pred_feat_chunk, "b t p d -> b d (t p)", b=B, p=self.patch_size)
                    yield feat_pred, pred_feat_seq

                stop_flag = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden))).argmax(dim=-1)[0].cpu().item()
                if step_idx > min_len and stop_flag == 1:
                    break

                lm_hidden = self.base_lm.forward_step(
                    curr_embed[:, 0, :],
                    torch.tensor([self.base_lm.kv_cache.step()], device=curr_embed.device),
                ).clone()

                lm_hidden = self.fsq_layer(lm_hidden)
                residual_hidden = self.residual_lm.forward_step(
                    lm_hidden + curr_embed[:, 0, :],
                    torch.tensor([self.residual_lm.kv_cache.step()], device=curr_embed.device),
                ).clone()

            if not streaming:
                pred_feat_seq_cat = torch.cat(pred_feat_seq, dim=1)
                feat_pred = rearrange(pred_feat_seq_cat, "b t p d -> b d (t p)", b=B, p=self.patch_size)
                yield feat_pred, pred_feat_seq_cat.squeeze(0).cpu()

        @torch.inference_mode()
        def generate_latents_with_prompt_cache(
            self,
            target_text: str,
            prompt_cache: dict,
            min_len: int = 2,
            max_len: int = 2000,
            inference_timesteps: int = 10,
            cfg_value: float = 2.0,
            retry_badcase: bool = False,
            retry_badcase_max_times: int = 3,
            retry_badcase_ratio_threshold: float = 6.0,
            streaming_prefix_len: int = 3,
        ) -> tuple[None, torch.Tensor, torch.Tensor]:
            return next(
                self._generate_with_prompt_cache(
                    target_text=target_text,
                    prompt_cache=prompt_cache,
                    min_len=min_len,
                    max_len=max_len,
                    inference_timesteps=inference_timesteps,
                    cfg_value=cfg_value,
                    retry_badcase=retry_badcase,
                    retry_badcase_max_times=retry_badcase_max_times,
                    retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                    streaming=False,
                    streaming_prefix_len=streaming_prefix_len,
                    latents_only=True,
                )
            )

        @torch.inference_mode()
        def generate_latents_with_prompt_cache_streaming(
            self,
            target_text: str,
            prompt_cache: dict,
            min_len: int = 2,
            max_len: int = 2000,
            inference_timesteps: int = 10,
            cfg_value: float = 2.0,
            retry_badcase: bool = False,
            retry_badcase_max_times: int = 3,
            retry_badcase_ratio_threshold: float = 6.0,
            streaming_prefix_len: int = 3,
        ) -> Generator[tuple[None, torch.Tensor, torch.Tensor], None, None]:
            return self._generate_with_prompt_cache(
                target_text=target_text,
                prompt_cache=prompt_cache,
                min_len=min_len,
                max_len=max_len,
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                retry_badcase=retry_badcase,
                retry_badcase_max_times=retry_badcase_max_times,
                retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                streaming=True,
                streaming_prefix_len=streaming_prefix_len,
                latents_only=True,
            )

        @torch.inference_mode()
        def _generate_with_prompt_cache(
            self,
            target_text: str,
            prompt_cache: dict,
            min_len: int = 2,
            max_len: int = 2000,
            inference_timesteps: int = 10,
            cfg_value: float = 2.0,
            retry_badcase: bool = False,
            retry_badcase_max_times: int = 3,
            retry_badcase_ratio_threshold: float = 6.0,
            streaming: bool = False,
            streaming_prefix_len: int = 3,
            latents_only: bool = False,
        ) -> Generator[
            tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | list[torch.Tensor]],
            None,
            None,
        ]:
            if retry_badcase and streaming:
                warnings.warn(
                    "Retry on bad cases is not supported in streaming mode, setting retry_badcase=False.",
                )
                retry_badcase = False
            if prompt_cache is None:
                prompt_audio_feat = torch.empty(
                    (0, self.patch_size, self.audio_vae.latent_dim),
                    dtype=torch.float32,
                )
                text = target_text
            else:
                prompt_audio_feat = prompt_cache["audio_feat"]
                prompt_text = prompt_cache["prompt_text"]
                text = prompt_text + target_text

            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat(
                [
                    text_token,
                    torch.tensor(
                        [self.audio_start_token],
                        dtype=torch.int32,
                        device=text_token.device,
                    ),
                ],
                dim=-1,
            )

            target_text_token = torch.LongTensor(self.text_tokenizer(target_text))

            audio_length = prompt_audio_feat.size(0)
            text_length = text_token.shape[0]
            text_pad_token = torch.zeros(audio_length, dtype=torch.int32, device=text_token.device)
            audio_pad_feat = torch.zeros(
                (text_token.shape[0], self.patch_size, self.audio_vae.latent_dim),
                dtype=torch.float32,
                device=text_token.device,
            )
            text_token = torch.cat([text_token, text_pad_token])
            audio_feat = torch.cat([audio_pad_feat, prompt_audio_feat], dim=0)
            text_mask = (
                torch.cat([torch.ones(text_length), torch.zeros(audio_length)]).type(torch.int32).to(text_token.device)
            )
            audio_mask = (
                torch.cat([torch.zeros(text_length), torch.ones(audio_length)]).type(torch.int32).to(text_token.device)
            )

            text_token = text_token.unsqueeze(0).to(self.device)
            text_mask = text_mask.unsqueeze(0).to(self.device)
            audio_feat = audio_feat.unsqueeze(0).to(self.device).to(get_dtype(self.config.dtype))
            audio_mask = audio_mask.unsqueeze(0).to(self.device)

            target_text_length = len(self.text_tokenizer(target_text))
            retry_badcase_times = 0
            while retry_badcase_times < retry_badcase_max_times:
                max_len_cap = min(int(target_text_length * retry_badcase_ratio_threshold + 10), max_len)
                inference_result = self._inference(
                    text_token,
                    text_mask,
                    audio_feat,
                    audio_mask,
                    min_len=min_len,
                    max_len=max_len if not retry_badcase else max_len_cap,
                    inference_timesteps=inference_timesteps,
                    cfg_value=cfg_value,
                    streaming=streaming,
                    streaming_prefix_len=streaming_prefix_len,
                )
                if streaming:
                    patch_len = self.patch_size * self.chunk_size
                    for latent_pred, pred_audio_feat in inference_result:
                        if latents_only:
                            decode_audio = None
                            yield (decode_audio, target_text_token, latent_pred)
                        else:
                            decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
                            decode_audio = decode_audio[..., -patch_len:].squeeze(1).cpu()
                            yield (decode_audio, target_text_token, pred_audio_feat)
                    break

                latent_pred, pred_audio_feat = next(inference_result)
                if retry_badcase:
                    if pred_audio_feat.shape[0] >= target_text_length * retry_badcase_ratio_threshold:
                        ratio = pred_audio_feat.shape[0] / target_text_length
                        print(
                            f"  Badcase detected, audio_text_ratio={ratio}, retrying...",
                            file=sys.stderr,
                        )
                        retry_badcase_times += 1
                        continue
                break

            if not streaming:
                if latents_only:
                    decode_audio = None
                else:
                    decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
                    patch_len = self.patch_size * self.chunk_size
                    if audio_mask.sum().item() > 0:
                        decode_audio = decode_audio[..., patch_len * (streaming_prefix_len - 1) :].squeeze(1).cpu()
                    else:
                        decode_audio = decode_audio[..., :].squeeze(1).cpu()
                yield (decode_audio, target_text_token, pred_audio_feat)

    VoxCPMModelForOmni.__name__ = "VoxCPMModelForOmni"
    VoxCPMModelForOmni.__qualname__ = "VoxCPMModelForOmni"
    return VoxCPMModelForOmni


def _import_voxcpm_model_class() -> type[Any]:
    base = _import_voxcpm_base_model_class()
    return _make_voxcpm_model_for_omni(base)


def _load_native_voxcpm_model(
    model_path: str,
    *,
    device: torch.device,
    dtype: str | None,
):
    VoxCPMModel = _import_voxcpm_model_class()
    runtime_model_path = _prepare_runtime_model_dir(model_path, target_device=device, target_dtype=dtype)

    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.set_device(device)

    with _force_cuda_available_for_npu(device):
        tts_model = VoxCPMModel.from_local(
            runtime_model_path,
            optimize=device.type == "cuda",
        )

    return tts_model


def _load_native_voxcpm_latent_generator(
    model_path: str,
    *,
    device: torch.device,
    dtype: str | None,
) -> _DirectVoxCPMLatentGenerator:
    return _DirectVoxCPMLatentGenerator(_load_native_voxcpm_model(model_path, device=device, dtype=dtype))


def _load_native_voxcpm_audio_vae(
    model_path: str,
    *,
    device: torch.device,
) -> _DirectVoxCPMAudioVAE:
    AudioVAE, AudioVAEConfig = _import_voxcpm_audio_vae_classes()
    runtime_model_path = _prepare_runtime_model_dir(model_path, target_device=device, target_dtype="float32")
    config_dict = json.loads((Path(runtime_model_path) / "config.json").read_text())
    audio_vae_config = config_dict.get("audio_vae_config")
    if audio_vae_config is not None:
        audio_vae = AudioVAE(config=AudioVAEConfig(**audio_vae_config))
    else:
        audio_vae = AudioVAE()

    state_dict = torch.load(
        Path(runtime_model_path) / "audiovae.pth",
        map_location="cpu",
        weights_only=True,
    )["state_dict"]
    audio_vae.load_state_dict(state_dict, strict=True)
    audio_vae = audio_vae.to(device=device, dtype=torch.float32).eval()
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.set_device(device)
    patch_size = int(config_dict.get("patch_size", 2))
    return _DirectVoxCPMAudioVAE(audio_vae, patch_size=patch_size)
