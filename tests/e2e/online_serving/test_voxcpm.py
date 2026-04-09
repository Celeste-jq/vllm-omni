# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E online tests for VoxCPM via /v1/audio/speech.

The online path supports:
- plain text-to-speech
- voice cloning with ref_audio + ref_text
- async-chunk streaming with PCM output
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

from pathlib import Path

import httpx
import pytest

from tests.conftest import OmniServerParams
from tests.utils import hardware_test

pytest.importorskip("voxcpm")

MODEL = "OpenBMB/VoxCPM1.5"
STAGE_CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "vllm_omni" / "model_executor" / "stage_configs"
ASYNC_STAGE_CONFIG = str(STAGE_CONFIG_DIR / "voxcpm.yaml")
SYNC_STAGE_CONFIG = str(STAGE_CONFIG_DIR / "voxcpm_no_async_chunk.yaml")
EXTRA_ARGS = ["--trust-remote-code", "--enforce-eager", "--disable-log-stats"]

REF_AUDIO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."

ASYNC_PARAMS = OmniServerParams(model=MODEL, stage_config_path=ASYNC_STAGE_CONFIG, server_args=EXTRA_ARGS)
SYNC_PARAMS = OmniServerParams(model=MODEL, stage_config_path=SYNC_STAGE_CONFIG, server_args=EXTRA_ARGS)
TEST_PARAMS = [ASYNC_PARAMS, SYNC_PARAMS]

MIN_AUDIO_BYTES = 10000
MIN_PCM_BYTES = 4096


def make_speech_request(
    host: str,
    port: int,
    *,
    text: str,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    response_format: str = "wav",
    timeout: float = 180.0,
) -> httpx.Response:
    url = f"http://{host}:{port}/v1/audio/speech"
    payload: dict[str, object] = {
        "model": MODEL,
        "input": text,
        "response_format": response_format,
    }
    if ref_audio is not None:
        payload["ref_audio"] = ref_audio
    if ref_text is not None:
        payload["ref_text"] = ref_text

    with httpx.Client(timeout=timeout) as client:
        return client.post(url, json=payload)


def verify_wav_audio(content: bytes) -> bool:
    if len(content) < 44:
        return False
    return content[:4] == b"RIFF" and content[8:12] == b"WAVE"


@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True, ids=["async_chunk", "no_async_chunk"])
class TestVoxCPMOnlineSpeech:
    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_text_speech_wav(self, omni_server) -> None:
        response = make_speech_request(
            host=omni_server.host,
            port=omni_server.port,
            text="This is a VoxCPM online text-to-speech smoke test.",
        )

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert response.headers.get("content-type") == "audio/wav"
        assert verify_wav_audio(response.content), "Response is not valid WAV audio"
        assert len(response.content) > MIN_AUDIO_BYTES, (
            f"Audio content too small ({len(response.content)} bytes), expected at least {MIN_AUDIO_BYTES} bytes"
        )

    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_voice_clone_wav(self, omni_server) -> None:
        response = make_speech_request(
            host=omni_server.host,
            port=omni_server.port,
            text="This sentence is synthesized online with a cloned voice.",
            ref_audio=REF_AUDIO_URL,
            ref_text=REF_TEXT,
        )

        assert response.status_code == 200, f"Request failed: {response.text}"
        assert response.headers.get("content-type") == "audio/wav"
        assert verify_wav_audio(response.content), "Response is not valid WAV audio"
        assert len(response.content) > MIN_AUDIO_BYTES, (
            f"Audio content too small ({len(response.content)} bytes), expected at least {MIN_AUDIO_BYTES} bytes"
        )


@pytest.mark.parametrize("omni_server", [ASYNC_PARAMS], indirect=True, ids=["async_chunk"])
class TestVoxCPMOnlineStreaming:
    @pytest.mark.core_model
    @pytest.mark.omni
    @hardware_test(res={"cuda": "H100"}, num_cards=1)
    def test_stream_pcm(self, omni_server) -> None:
        url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
        payload = {
            "model": MODEL,
            "input": "This is a VoxCPM online streaming speech smoke test.",
            "stream": True,
            "response_format": "pcm",
        }

        total_bytes = 0
        first_chunk = None
        with httpx.Client(timeout=180.0) as client:
            with client.stream("POST", url, json=payload) as response:
                assert response.status_code == 200, response.read().decode("utf-8", errors="ignore")
                assert response.headers.get("content-type") == "audio/pcm"
                for chunk in response.iter_bytes():
                    if not chunk:
                        continue
                    if first_chunk is None:
                        first_chunk = chunk
                    total_bytes += len(chunk)

        assert first_chunk is not None, "Did not receive any PCM chunk from streaming response"
        assert total_bytes > MIN_PCM_BYTES, (
            f"Streaming PCM too small ({total_bytes} bytes), expected at least {MIN_PCM_BYTES} bytes"
        )
