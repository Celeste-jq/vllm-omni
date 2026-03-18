from __future__ import annotations

from typing import Any

from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt


def latent2vae(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    del prompt, requires_multimodal_data

    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    source_outputs = stage_list[source_stage_id].engine_outputs
    if source_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    vae_inputs: list[OmniTokensPrompt] = []
    for source_output in source_outputs:
        output = source_output.outputs[0]
        multimodal_output = getattr(output, "multimodal_output", None)
        if not isinstance(multimodal_output, dict) or "latent_audio_feat" not in multimodal_output:
            raise ValueError(
                "VoxCPM latent stage output missing 'latent_audio_feat'. "
                f"request_id={getattr(source_output, 'request_id', None)}"
            )

        additional_information = {
            "latent_audio_feat": multimodal_output["latent_audio_feat"],
        }
        if "sr" in multimodal_output:
            additional_information["sample_rate"] = [int(multimodal_output["sr"])]

        vae_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                additional_information=additional_information,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return vae_inputs
