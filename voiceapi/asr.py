from typing import *
import logging
import time
import logging
import sherpa_onnx
import os
import asyncio
import numpy as np

logger = logging.getLogger(__file__)
_asr_engine = None


class ASRResult:
    def __init__(self, text: str, finished: bool, idx: int):
        self.text = text
        self.finished = finished
        self.idx = idx

    def to_dict(self):
        return {"text": self.text, "finished": self.finished, "idx": self.idx}


class ASRStream:
    def __init__(self, sample_rate: int) -> None:
        self.inbuf = asyncio.Queue()
        self.outbuf = asyncio.Queue()
        self.sample_rate = sample_rate
        self.is_closed = False

    async def start(self):
        asyncio.create_task(self.run_online())

    async def run_online(self):
        stream = _asr_engine.create_stream()
        last_result = ""
        segment_id = 0
        logger.info('asr: start real-time recognizer')
        while not self.is_closed:
            samples = await self.inbuf.get()
            stream.accept_waveform(self.sample_rate, samples)
            while _asr_engine.is_ready(stream):
                _asr_engine.decode_stream(stream)

            is_endpoint = _asr_engine.is_endpoint(stream)
            result = _asr_engine.get_result(stream)

            if result and (last_result != result):
                last_result = result
                logger.info(f' > {segment_id}:{result}')
                self.outbuf.put_nowait(
                    ASRResult(result, False, segment_id))

            if is_endpoint:
                if result:
                    logger.info(f'{segment_id}: {result}')
                    self.outbuf.put_nowait(
                        ASRResult(result, True, segment_id))
                    segment_id += 1
                _asr_engine.reset(stream)

    async def close(self):
        self.is_closed = True
        self.outbuf.put_nowait(None)

    async def write(self, pcm_bytes: bytes):
        pcm_data = np.frombuffer(pcm_bytes, dtype=np.int16)
        samples = pcm_data.astype(np.float32) / 32768.0
        self.inbuf.put_nowait(samples)

    async def read(self) -> ASRResult:
        return await self.outbuf.get()


def create_zipformer(samplerate: int, args) -> sherpa_onnx.OnlineRecognizer:
    d = os.path.join(
        args.model_root, 'sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20')
    encoder = os.path.join(d, "encoder-epoch-99-avg-1.onnx")
    decoder = os.path.join(d, "decoder-epoch-99-avg-1.onnx")
    joiner = os.path.join(d, "joiner-epoch-99-avg-1.onnx")
    tokens = os.path.join(d, "tokens.txt")

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        provider=args.provider,
        num_threads=args.threads,
        sample_rate=samplerate,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=20,  # it essentially disables this rule
    )
    return recognizer


def get_asr_engine(samplerate: int, args) -> sherpa_onnx.OnlineRecognizer:
    global _asr_engine
    if not _asr_engine:
        st = time.time()
        _asr_engine = create_zipformer(samplerate, args)
        logger.info(f"asr: engine loaded in {time.time() - st:.2f}s")
    return _asr_engine


async def start_asr_stream(samplerate: int, args) -> ASRStream:
    """
    Start a ASR stream
    """
    get_asr_engine(samplerate, args)
    stream = ASRStream(samplerate)
    await stream.start()
    return stream
