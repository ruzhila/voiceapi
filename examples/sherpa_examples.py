#!/bin/env python3
"""
    Real-time ASR using microphone
"""

import argparse
import logging
import sherpa_onnx
import os
import time
import struct
import asyncio
import soundfile

try:
    import pyaudio
except ImportError:
    raise ImportError('Please install pyaudio with `pip install pyaudio`')

logger = logging.getLogger(__name__)
SAMPLE_RATE = 16000

pactx = pyaudio.PyAudio()
models_root: str = None
num_threads: int = 1


def create_zipformer(args) -> sherpa_onnx.OnlineRecognizer:
    d = os.path.join(
        models_root, 'sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20')
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
        num_threads=num_threads,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=20,  # it essentially disables this rule
    )
    return recognizer


def create_sensevoice(args) -> sherpa_onnx.OfflineRecognizer:
    model = os.path.join(
        models_root, 'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17', 'model.onnx')
    tokens = os.path.join(
        models_root, 'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17', 'tokens.txt')
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model,
        tokens=tokens,
        num_threads=num_threads,
        use_itn=True,
        debug=0,
        language=args.lang,
    )
    return recognizer


async def run_online(buf, recognizer):
    stream = recognizer.create_stream()
    last_result = ""
    segment_id = 0
    logger.info('Start real-time recognizer')
    while True:
        samples = await buf.get()
        stream.accept_waveform(SAMPLE_RATE, samples)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        is_endpoint = recognizer.is_endpoint(stream)
        result = recognizer.get_result(stream)

        if result and (last_result != result):
            last_result = result
            logger.info(f' > {segment_id}:{result}')

        if is_endpoint:
            if result:
                logger.info(f'{segment_id}: {result}')
                segment_id += 1
            recognizer.reset(stream)


async def run_offline(buf, recognizer):
    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = os.path.join(
        models_root, 'silero_vad', 'silero_vad.onnx')
    config.silero_vad.min_silence_duration = 0.25
    config.sample_rate = SAMPLE_RATE
    vad = sherpa_onnx.VoiceActivityDetector(
        config, buffer_size_in_seconds=100)

    logger.info('Start offline recognizer with VAD')
    texts = []
    while True:
        samples = await buf.get()
        vad.accept_waveform(samples)
        while not vad.empty():
            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, vad.front.samples)

            vad.pop()
            recognizer.decode_stream(stream)

            text = stream.result.text.strip().lower()
            if len(text):
                idx = len(texts)
                texts.append(text)
                logger.info(f"{idx}: {text}")


async def handle_asr(args):
    action_func = None
    if args.model == 'zipformer':
        recognizer = create_zipformer(args)
        action_func = run_online
    elif args.model == 'sensevoice':
        recognizer = create_sensevoice(args)
        action_func = run_offline
    else:
        raise ValueError(f'Unknown model: {args.model}')
    buf = asyncio.Queue()
    recorder_task = asyncio.create_task(run_record(buf))
    asr_task = asyncio.create_task(action_func(buf, recognizer))
    await asyncio.gather(asr_task, recorder_task)


async def handle_tts(args):
    model = os.path.join(
        models_root, 'vits-melo-tts-zh_en', 'model.onnx')
    lexicon = os.path.join(
        models_root, 'vits-melo-tts-zh_en', 'lexicon.txt')
    dict_dir = os.path.join(
        models_root, 'vits-melo-tts-zh_en', 'dict')
    tokens = os.path.join(
        models_root, 'vits-melo-tts-zh_en', 'tokens.txt')
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
            num_threads=num_threads,
        ),
        max_num_sentences=args.max_num_sentences,
    )
    if not tts_config.validate():
        raise ValueError("Please check your config")

    tts = sherpa_onnx.OfflineTts(tts_config)

    start = time.time()
    audio = tts.generate(args.text, sid=args.sid,
                         speed=args.speed)
    elapsed_seconds = time.time() - start
    audio_duration = len(audio.samples) / audio.sample_rate
    real_time_factor = elapsed_seconds / audio_duration

    if args.output:
        logger.info(f"Saved to {args.output}")
        soundfile.write(
            args.output,
            audio.samples,
            samplerate=audio.sample_rate,
            subtype="PCM_16",
        )

    logger.info(f"The text is '{args.text}'")
    logger.info(f"Elapsed seconds: {elapsed_seconds:.3f}")
    logger.info(f"Audio duration in seconds: {audio_duration:.3f}")
    logger.info(
        f"RTF: {elapsed_seconds:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")


async def run_record(buf: asyncio.Queue[list[float]]):
    loop = asyncio.get_event_loop()

    def on_input(in_data, frame_count, time_info, status):
        samples = [
            v/32768.0 for v in list(struct.unpack('<' + 'h' * frame_count, in_data))]
        loop.create_task(buf.put(samples))
        return (None, pyaudio.paContinue)

    frame_size = 320
    recorder = pactx.open(format=pyaudio.paInt16, channels=1,
                          rate=SAMPLE_RATE, input=True,
                          frames_per_buffer=frame_size,
                          stream_callback=on_input)
    recorder.start_stream()
    logger.info('Start recording')

    while recorder.is_active():
        await asyncio.sleep(0.1)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', default='cpu',
                        help='onnxruntime provider, default is cpu, use cuda for GPU')

    subparsers = parser.add_subparsers(help='commands help')

    asr_parser = subparsers.add_parser('asr', help='run asr mode')
    asr_parser.add_argument('--model', default='zipformer',
                            help='model name, default is zipformer')
    asr_parser.add_argument('--lang',  default='zh',
                            help='language, default is zh')
    asr_parser.set_defaults(func=handle_asr)

    tts_parser = subparsers.add_parser('tts', help='run tts mode')
    tts_parser.add_argument('--sid', type=int, default=0, help="""Speaker ID. Used only for multi-speaker models, e.g.
        models trained using the VCTK dataset. Not used for single-speaker
        models, e.g., models trained using the LJ speech dataset.
        """)
    tts_parser.add_argument('--output', type=str, default='output.wav',
                            help='output file name, default is output.wav')
    tts_parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed. Larger->faster; smaller->slower",
    )
    tts_parser.add_argument(
        "--max-num-sentences",
        type=int,
        default=2,
        help="""Max number of sentences in a batch to avoid OOM if the input
        text is very long. Set it to -1 to process all the sentences in a
        single batch. A smaller value does not mean it is slower compared
        to a larger one on CPU.
        """,
    )
    tts_parser.add_argument(
        "text",
        type=str,
        help="The input text to generate audio for",
    )
    tts_parser.set_defaults(func=handle_tts)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        await args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s: %(asctime)s %(name)s:%(lineno)s %(message)s')
    logging.getLogger().setLevel(logging.INFO)
    painfo = pactx.get_default_input_device_info()
    assert painfo['maxInputChannels'] >= 1, 'No input device'
    logger.info('Default input device: %s', painfo['name'])

    for d in ['.', '..', '../..']:
        if os.path.isdir(f'{d}/models'):
            models_root = f'{d}/models'
            break
    assert models_root, 'Could not find models directory'
    asyncio.run(main())
