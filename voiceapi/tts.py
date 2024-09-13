from typing import *
import os
import time
import sherpa_onnx
import logging
import numpy as np
import asyncio
import time
import soundfile
from scipy.signal import resample
import io

_tts_engine = None
logger = logging.getLogger(__file__)


class TTSResult:
    def __init__(self, pcm_bytes: bytes, finished: bool):
        self.pcm_bytes = pcm_bytes
        self.finished = False
        self.progress: float = 0.0
        self.elapsed: float = 0.0
        self.audio_duration: float = 0.0
        self.audio_size: int = 0

    def to_dict(self):
        return {
            "progress": self.progress,
            "elapsed": f'{int(self.elapsed * 1000)}ms',
            "duration": f'{self.audio_duration:.2f}s',
            "size": self.audio_size
        }


class TTSStream:
    def __init__(self, sid: int, speed: float = 1.0, sample_rate: int = 16000):
        self.sid = sid
        self.speed = speed
        self.outbuf: asyncio.Queue[TTSResult | None] = asyncio.Queue()
        self.is_closed = False
        self.original_sample_rate = 44100
        self.target_sample_rate = sample_rate

    def on_process(self, chunk: np.ndarray, progress: float):
        if self.is_closed:
            return 0
        # resample to 16k
        num_samples = int(len(chunk) * self.target_sample_rate /
                          self.original_sample_rate)
        resampled_chunk = resample(chunk, num_samples)
        resampled_chunk = resampled_chunk.astype(np.float32)

        scaled_chunk = resampled_chunk * 32768.0
        clipped_chunk = np.clip(scaled_chunk, -32768, 32767)
        int16_chunk = clipped_chunk.astype(np.int16)
        samples = int16_chunk.tobytes()
        self.outbuf.put_nowait(TTSResult(samples, False))
        return self.is_closed and 0 or 1

    async def write(self, text: str, interrput: bool = False):
        start = time.time()
        audio = _tts_engine.generate(text,
                                     sid=self.sid,
                                     speed=self.speed,
                                     callback=self.on_process)
        elapsed_seconds = time.time() - start
        audio_duration = len(audio.samples) / audio.sample_rate

        logger.info(f"tts: generated audio in {elapsed_seconds:.2f}s, "
                    f"audio duration: {audio_duration:.2f}s, "
                    f"sample rate: {audio.sample_rate}")

        r = TTSResult(None, True)
        r.elapsed = elapsed_seconds
        r.audio_duration = audio_duration
        r.audio_size = len(audio.samples)
        r.progress = 1.0
        r.finished = True
        await self.outbuf.put(r)

    async def close(self):
        self.is_closed = True
        self.outbuf.put_nowait(None)

    async def read(self) -> TTSResult:
        return await self.outbuf.get()

    async def generate(self, sid: int, text: str, samplerate: int, speed: float = 1.0) -> io.BytesIO:
        start = time.time()
        audio = _tts_engine.generate(text,
                                     sid=sid,
                                     speed=speed)
        elapsed_seconds = time.time() - start
        audio_duration = len(audio.samples) / audio.sample_rate

        logger.info(f"tts: generated audio in {elapsed_seconds:.2f}s, "
                    f"audio duration: {audio_duration:.2f}s, "
                    f"sample rate: {audio.sample_rate}")

        if samplerate != audio.sample_rate:
            audio.samples = resample(audio.samples,
                                     int(len(audio.samples) * samplerate / audio.sample_rate))
            audio.sample_rate = samplerate

        output = io.BytesIO()
        soundfile.write(output,
                        audio.samples,
                        samplerate=audio.sample_rate,
                        subtype="PCM_16",
                        format="WAV")
        output.seek(0)
        return output


def get_tts_config(args):
    model = os.path.join(args.model_root, 'vits-melo-tts-zh_en', 'model.onnx')
    lexicon = os.path.join(
        args.model_root, 'vits-melo-tts-zh_en', 'lexicon.txt')
    dict_dir = os.path.join(args.model_root, 'vits-melo-tts-zh_en', 'dict')
    tokens = os.path.join(args.model_root, 'vits-melo-tts-zh_en', 'tokens.txt')
    for f in [model, lexicon, dict_dir, tokens]:
        if not os.path.exists(f):
            raise FileNotFoundError(f)

    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=model,
                lexicon=lexicon,
                dict_dir=dict_dir,
                tokens=tokens,
            ),
            provider=args.provider,
            debug=0,
            num_threads=args.threads,
        ),
        max_num_sentences=20,
    )
    if not tts_config.validate():
        raise ValueError("tts: invalid config")
    return tts_config


def get_tts_engine(args):
    global _tts_engine
    if not _tts_engine:
        st = time.time()
        _tts_engine = sherpa_onnx.OfflineTts(get_tts_config(args))
        logger.info(f"tts: engine loaded in {time.time() - st:.2f}s")
    return _tts_engine


async def start_tts_stream(sid: int, samplerate: int, args) -> TTSStream:
    get_tts_engine(args)
    return TTSStream(sid)
