from typing import *
import logging
import time
import logging
import sherpa_onnx
import os
import asyncio
import numpy as np

logger = logging.getLogger(__file__)
_asr_engines = {}


class ASRResult:
    def __init__(self, text: str, finished: bool, idx: int, start: float = 0.0, end: float = 0.0, channel: Optional[int] = None):
        self.text = text
        self.finished = finished
        self.idx = idx
        self.start = start
        self.end = end
        self.channel = channel

    def to_dict(self):
        return {
            "text": self.text,
            "finished": self.finished,
            "idx": self.idx,
            "start": self.start,
            "end": self.end,
            "channel": self.channel,
        }


class ASRStream:
    def __init__(self, recognizer: Union[sherpa_onnx.OnlineRecognizer | sherpa_onnx.OfflineRecognizer], sample_rate: int) -> None:
        self.recognizer = recognizer
        self.inbuf = asyncio.Queue()
        self.outbuf = asyncio.Queue()
        self.sample_rate = sample_rate
        self.is_closed = False
        self.online = isinstance(recognizer, sherpa_onnx.OnlineRecognizer)

    async def start(self):
        if self.online:
            asyncio.create_task(self.run_online())
        else:
            asyncio.create_task(self.run_offline())

    async def run_online(self):
        stream = self.recognizer.create_stream()
        last_result = ""
        segment_id = 0
        logger.info('asr: start real-time recognizer')
        while not self.is_closed:
            samples = await self.inbuf.get()
            stream.accept_waveform(self.sample_rate, samples)
            while self.recognizer.is_ready(stream):
                self.recognizer.decode_stream(stream)

            is_endpoint = self.recognizer.is_endpoint(stream)
            result = self.recognizer.get_result(stream)

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
                self.recognizer.reset(stream)

    async def run_offline(self):
        vad = _asr_engines['vad']
        vad.reset()
        segment_id = 0
        processed_samples = 0
        while not self.is_closed:
            samples = await self.inbuf.get()
            vad.accept_waveform(samples)
            while not vad.empty():
                segment = vad.front
                segment_samples = np.asarray(segment.samples, dtype=np.float32)
                start_time, end_time, processed_samples = _resolve_segment_times(
                    segment,
                    len(segment_samples),
                    self.sample_rate,
                    processed_samples,
                )

                stream = self.recognizer.create_stream()
                stream.accept_waveform(self.sample_rate, segment_samples)
                stream.input_finished()

                vad.pop()
                self.recognizer.decode_stream(stream)

                result = stream.result.text.strip()
                if result:
                    logger.info(f'{segment_id}:{result} ({start_time:.2f}s - {end_time:.2f}s)')
                    self.outbuf.put_nowait(ASRResult(result, True, segment_id, start_time, end_time))
                    segment_id += 1

    async def close(self):
        self.is_closed = True
        self.outbuf.put_nowait(None)

    async def write(self, pcm_bytes: bytes):
        pcm_data = np.frombuffer(pcm_bytes, dtype=np.int16)
        samples = pcm_data.astype(np.float32) / 32768.0
        self.inbuf.put_nowait(samples)

    async def read(self) -> ASRResult:
        return await self.outbuf.get()


def _resolve_segment_times(segment, segment_len: int, sample_rate: int, fallback_start_samples: int) -> tuple[float, float, int]:
    """Return (start_time, end_time, end_samples) for a VAD segment."""
    start_time = getattr(segment, "start_time", None)
    if start_time is not None:
        start_time = float(start_time)
        start_samples = int(round(start_time * sample_rate))
    else:
        start_attr = getattr(segment, "start", fallback_start_samples)
        if isinstance(start_attr, (int, np.integer)):
            start_samples = int(start_attr)
        else:
            start_samples = int(round(float(start_attr) * sample_rate))
        start_time = start_samples / sample_rate

    end_time = getattr(segment, "end_time", None)
    if end_time is not None:
        end_time = float(end_time)
        end_samples = int(round(end_time * sample_rate))
    else:
        end_attr = getattr(segment, "end", start_samples + segment_len)
        if isinstance(end_attr, (int, np.integer)):
            end_samples = int(end_attr)
        else:
            end_samples = int(round(float(end_attr) * sample_rate))
        end_time = end_samples / sample_rate

    if end_samples < start_samples + segment_len:
        end_samples = start_samples + segment_len
        end_time = end_samples / sample_rate

    return start_time, end_time, end_samples


def create_zipformer(samplerate: int, args) -> sherpa_onnx.OnlineRecognizer:
    d = os.path.join(
        args.models_root, 'sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20')
    if not os.path.exists(d):
        raise ValueError(f"asr: model not found {d}")

    encoder = os.path.join(d, "encoder-epoch-99-avg-1.onnx")
    decoder = os.path.join(d, "decoder-epoch-99-avg-1.onnx")
    joiner = os.path.join(d, "joiner-epoch-99-avg-1.onnx")
    tokens = os.path.join(d, "tokens.txt")

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tokens,
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        provider=args.asr_provider,
        num_threads=args.threads,
        sample_rate=samplerate,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=20,  # it essentially disables this rule
    )
    return recognizer


def create_sensevoice(samplerate: int, use_int8:bool, args) -> sherpa_onnx.OfflineRecognizer:
    d = os.path.join(args.models_root,
                     'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17')

    if not os.path.exists(d):
        raise ValueError(f"asr: model not found {d}")

    # prefer explicit int8 model file if present (backwards compatible)
    model_path = os.path.join(d, 'model.onnx')
    if use_int8 and os.path.exists(os.path.join(d, 'model.int8.onnx')):
        model_path = os.path.join(d, 'model.int8.onnx')

    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model_path,
        tokens=os.path.join(d, 'tokens.txt'),
        num_threads=args.threads,
        sample_rate=samplerate,
        use_itn=True,
        debug=0,
        language=args.asr_lang,
    )
    return recognizer

def create_paraformer_trilingual(samplerate: int, args) -> sherpa_onnx.OnlineRecognizer:
    d = os.path.join(
        args.models_root, 'sherpa-onnx-paraformer-trilingual-zh-cantonese-en')
    if not os.path.exists(d):
        raise ValueError(f"asr: model not found {d}")

    recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=os.path.join(d, 'model.onnx'),
        tokens=os.path.join(d, 'tokens.txt'),
        num_threads=args.threads,
        sample_rate=samplerate,
        debug=0,
        provider=args.asr_provider,
    )
    return recognizer


def create_paraformer_en(samplerate: int, args) -> sherpa_onnx.OnlineRecognizer:
    d = os.path.join(
        args.models_root, 'sherpa-onnx-paraformer-en')
    if not os.path.exists(d):
        raise ValueError(f"asr: model not found {d}")

    recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=os.path.join(d, 'model.onnx'),
        tokens=os.path.join(d, 'tokens.txt'),
        num_threads=args.threads,
        sample_rate=samplerate,
        use_itn=True,
        debug=0,
        provider=args.asr_provider,
    )
    return recognizer

def create_fireredasr(samplerate: int, args) -> sherpa_onnx.OnlineRecognizer:
    d = os.path.join(
        args.models_root, 'sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16')
    if not os.path.exists(d):
        raise ValueError(f"asr: model not found {d}")

    encoder = os.path.join(d, "encoder.int8.onnx")
    decoder = os.path.join(d, "decoder.int8.onnx")
    tokens = os.path.join(d, "tokens.txt")

    recognizer = sherpa_onnx.OfflineRecognizer.from_fire_red_asr(
        encoder=encoder,
        decoder=decoder,
        tokens=tokens,
        debug=0,
        provider=args.asr_provider,
    )
    return recognizer



def load_asr_engine(samplerate: int, args) -> sherpa_onnx.OnlineRecognizer:
    cache_engine = _asr_engines.get(args.asr_model)
    if cache_engine:
        return cache_engine
    st = time.time()
    if args.asr_model == 'zipformer-bilingual':
        cache_engine = create_zipformer(samplerate, args)
    elif args.asr_model == 'sensevoice':
        cache_engine = create_sensevoice(samplerate, False, args)
        _asr_engines['vad'] = load_vad_engine(samplerate, args)
    elif args.asr_model == 'sensevoice-int8':
        cache_engine = create_sensevoice(samplerate, True, args)
        _asr_engines['vad'] = load_vad_engine(samplerate, args)
    elif args.asr_model == 'paraformer-trilingual':
        cache_engine = create_paraformer_trilingual(samplerate, args)
        _asr_engines['vad'] = load_vad_engine(samplerate, args)
    elif args.asr_model == 'paraformer-en':
        cache_engine = create_paraformer_en(samplerate, args)
        _asr_engines['vad'] = load_vad_engine(samplerate, args)
    elif args.asr_model == 'fireredasr':
        cache_engine = create_fireredasr(samplerate, args)
        _asr_engines['vad'] = load_vad_engine(samplerate, args)
    else:
        raise ValueError(f"asr: unknown model {args.asr_model}")
    _asr_engines[args.asr_model] = cache_engine
    logger.info(f"asr: engine loaded in {time.time() - st:.2f}s")
    return cache_engine


def load_vad_engine(samplerate: int, args, min_silence_duration: float = 0.25, buffer_size_in_seconds: int = 100) -> sherpa_onnx.VoiceActivityDetector:
    config = sherpa_onnx.VadModelConfig()
    d = os.path.join(args.models_root, 'silero_vad')
    if not os.path.exists(d):
        raise ValueError(f"vad: model not found {d}")

    config.silero_vad.model = os.path.join(d, 'silero_vad.onnx')
    config.silero_vad.min_silence_duration = min_silence_duration
    config.sample_rate = samplerate
    config.provider = args.asr_provider
    config.num_threads = args.threads

    vad = sherpa_onnx.VoiceActivityDetector(
        config,
        buffer_size_in_seconds=buffer_size_in_seconds)
    return vad


async def start_asr_stream(samplerate: int, args) -> ASRStream:
    """
    Start a ASR stream
    """
    stream = ASRStream(load_asr_engine(samplerate, args), samplerate)
    await stream.start()
    return stream

async def process_asr_file(samples: np.ndarray, samplerate: int, args, channel: Optional[int] = None) -> List[ASRResult]:
    recognizer = load_asr_engine(samplerate, args)
    vad = load_vad_engine(samplerate, args)
    vad.reset()
    samples = np.asarray(samples, dtype=np.float32)

    results: List[ASRResult] = []
    segment_id = 0
    processed_samples = 0
    chunk_size = max(1, int(samplerate * 0.02))  # feed 20 ms per VAD step

    def consume_vad_segments():
        nonlocal processed_samples, segment_id
        while not vad.empty():
            segment = vad.front
            segment_samples = np.asarray(segment.samples, dtype=np.float32)
            start_time, end_time, next_processed_samples = _resolve_segment_times(
                segment,
                len(segment_samples),
                samplerate,
                processed_samples,
            )
            vad.pop()

            processed_samples = next_processed_samples
            if segment_samples.size == 0:
                continue

            stream = recognizer.create_stream()
            stream.accept_waveform(samplerate, segment_samples)
            recognizer.decode_stream(stream)

            result_text = stream.result.text.strip()
            if result_text:
                results.append(ASRResult(result_text, True, segment_id, start_time, end_time, channel))
                segment_id += 1

    for offset in range(0, len(samples), chunk_size):
        chunk = samples[offset:offset + chunk_size]
        vad.accept_waveform(chunk)
        consume_vad_segments()

    vad.flush()
    consume_vad_segments()

    if not results and samples.size:
        stream = recognizer.create_stream()
        stream.accept_waveform(samplerate, samples)
        recognizer.decode_stream(stream)
        result_text = stream.result.text.strip()
        if result_text:
            duration = len(samples) / samplerate
            results.append(ASRResult(result_text, True, 0, 0.0, duration, channel))
    return results