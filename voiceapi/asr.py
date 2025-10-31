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
    def __init__(self, text: str, finished: bool, idx: int, start: float = 0.0, end: float = 0.0):
        self.text = text
        self.finished = finished
        self.idx = idx
        self.start = start
        self.end = end

    def to_dict(self):
        return {"text": self.text, "finished": self.finished, "idx": self.idx, "start": self.start, "end": self.end}


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
        segment_id = 0
        total_samples = 0
        while not self.is_closed:
            samples = await self.inbuf.get()
            vad.accept_waveform(samples)
            while not vad.empty():
                segment_samples = vad.front.samples
                start_time = total_samples / self.sample_rate
                end_time = (total_samples + len(segment_samples)) / self.sample_rate
                total_samples += len(segment_samples)
                
                stream = self.recognizer.create_stream()
                stream.accept_waveform(self.sample_rate, segment_samples)

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


def create_sensevoice(samplerate: int, args) -> sherpa_onnx.OfflineRecognizer:
    d = os.path.join(args.models_root,
                     'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17')

    if not os.path.exists(d):
        raise ValueError(f"asr: model not found {d}")

    # prefer explicit int8 model file if present (backwards compatible)
    model_path = os.path.join(d, 'model.onnx')
    if os.path.exists(os.path.join(d, 'model.int8.onnx')):
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


def create_sensevoice_int8(samplerate: int, args) -> sherpa_onnx.OfflineRecognizer:
    """
    Load the int8 variant of SenseVoice if available. This function will try
    to load `model.int8.onnx` first and fall back to `model.onnx` if needed.
    """
    d = os.path.join(args.models_root,
                     'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09')

    if not os.path.exists(d):
        raise ValueError(f"asr: model not found {d}")

    # prefer explicit int8 model file but accept model.onnx if that's what user has
    model_path = os.path.join(d, 'model.int8.onnx')
    if not os.path.exists(model_path):
        model_path = os.path.join(d, 'model.onnx')

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
        cache_engine = create_sensevoice(samplerate, args)
        _asr_engines['vad'] = load_vad_engine(samplerate, args)
    elif args.asr_model == 'sensevoice-int8':
        cache_engine = create_sensevoice_int8(samplerate, args)
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

async def process_asr_file(samples: np.ndarray, samplerate: int, args) -> List[ASRResult]:
    recognizer = load_asr_engine(samplerate, args)
    vad = load_vad_engine(samplerate, args)
    results = []
    segment_id = 0
    total_samples = 0
    vad.accept_waveform(samples)
    while not vad.empty():
        segment_samples = vad.front.samples
        start_time = total_samples / samplerate
        end_time = (total_samples + len(segment_samples)) / samplerate
        total_samples += len(segment_samples)
        
        stream = recognizer.create_stream()
        stream.accept_waveform(samplerate, segment_samples)
        recognizer.decode_stream(stream)
        
        result_text = stream.result.text.strip()
        if result_text:
            results.append(ASRResult(result_text, True, segment_id, start_time, end_time))
            segment_id += 1
        vad.pop()
    return results
