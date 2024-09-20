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
import re

_tts_engine = None
original_sample_rate = 44100
logger = logging.getLogger(__file__)

splitter = re.compile(r'[,，。.!?！？;；、\n]')


class TTSResult:
    def __init__(self, pcm_bytes: bytes, finished: bool):
        self.pcm_bytes = pcm_bytes
        self.finished = finished
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
        self.target_sample_rate = sample_rate

    def on_process(self, chunk: np.ndarray, progress: float):
        if self.is_closed:
            return 0

        # resample to target sample rate
        if self.target_sample_rate != original_sample_rate:
            num_samples = int(
                len(chunk) * self.target_sample_rate / original_sample_rate)
            resampled_chunk = resample(chunk, num_samples)
            chunk = resampled_chunk.astype(np.float32)

        scaled_chunk = chunk * 32768.0
        clipped_chunk = np.clip(scaled_chunk, -32768, 32767)
        int16_chunk = clipped_chunk.astype(np.int16)
        samples = int16_chunk.tobytes()
        self.outbuf.put_nowait(TTSResult(samples, False))
        return self.is_closed and 0 or 1

    async def write(self, text: str, split: bool):
        start = time.time()
        if split:
            texts = re.split(splitter, text)
        else:
            texts = [text]

        audio_duration = 0.0
        audio_size = 0

        for text in texts:
            text = text.strip()
            if not text:
                continue
            sub_start = time.time()
            audio = _tts_engine.generate(text,
                                         sid=self.sid,
                                         speed=self.speed,
                                         callback=self.on_process)
            
            if not audio or not audio.sample_rate or not audio.samples:
                logger.error(f"tts: failed to generate audio for '{
                             text}' (audio={audio})")
                continue
            
            audio_duration += len(audio.samples) / audio.sample_rate
            audio_size += len(audio.samples)
            elapsed_seconds = time.time() - sub_start
            logger.info(f"tts: generated audio for '{text}', "
                        f"audio duration: {audio_duration:.2f}s, "
                        f"elapsed: {elapsed_seconds:.2f}s")

        elapsed_seconds = time.time() - start
        logger.info(f"tts: generated audio in {elapsed_seconds:.2f}s, "
                    f"audio duration: {audio_duration:.2f}s")

        r = TTSResult(None, True)
        r.elapsed = elapsed_seconds
        r.audio_duration = audio_duration
        r.progress = 1.0
        r.finished = True
        await self.outbuf.put(r)

    async def close(self):
        self.is_closed = True
        self.outbuf.put_nowait(None)
        logger.info("tts: stream closed")

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

    global original_sample_rate
    original_sample_rate = 44100

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
    global _tts_engine, original_sample_rate
    if not _tts_engine:
        st = time.time()
        _tts_engine = sherpa_onnx.OfflineTts(get_tts_config(args))
        print(_tts_engine)
        logger.info(f"tts: engine loaded in {time.time() - st:.2f}s")
    return _tts_engine


async def start_tts_stream(sid: int, sample_rate: int, speed: float, args) -> TTSStream:
    get_tts_engine(args)
    return TTSStream(sid, speed, sample_rate)
