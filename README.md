# voiceapi - a streaming voice transcription/synthesis API with sherpa-onnx

## How to use
Thanks to [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), we can easily build a voice API with Python.

## Run the app (only tested on Linux/MacOS with CPU)
![screenshot](./screenshot.jpg)

```shell
python3 -m venv venv
. venv/bin/activate

pip install -r requirements.txt
python app.py
```

## Download models
All models are stored in the `models` directory
Only download the models you need. default models are:
- asr models: `sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`(Bilingual, Chinese + English)
- tts models: `vits-melo-tts-zh_en` (Chinese + English)

### vits-melo-tts-zh_en
```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2
```
### sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
```bash 
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
```

### silero_vad.onnx
```bash
curl -SL -O https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
```
### sherpa-onnx-paraformer-trilingual-zh-cantonese-en
```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-trilingual-zh-cantonese-en.tar.bz2
```
### whisper
```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
```
### sensevoice
```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
```

### sherpa-onnx-streaming-paraformer-bilingual-zh-en
```bash
curl -SL -O  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
```
