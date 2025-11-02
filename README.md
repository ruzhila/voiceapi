# voiceapi - A simple and clean voice transcription/synthesis API with sherpa-onnx

Thanks to [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), we can easily build a voice API with Python.
<img src="./screenshot.png" width="60%">

## Supported models
| Model                                       | Language                      | Type        | Description                         |
| ------------------------------------------- | ----------------------------- | ----------- | ----------------------------------- |
| zipformer-bilingual-zh-en-2023-02-20        | Chinese + English             | Online ASR  | Streaming Zipformer, Bilingual      |
| sense-voice-zh-en-ja-ko-yue-2024-07-17      | Chinese + English             | Offline ASR | SenseVoice, Bilingual               |
| paraformer-trilingual-zh-cantonese-en       | Chinese + Cantonese + English | Offline ASR | Paraformer, Trilingual              |
| paraformer-en-2024-03-09                    | English                       | Offline ASR | Paraformer, English                 |
| vits-zh-hf-theresa                          | Chinese                       | TTS         | VITS, Chinese, 804 speakers         |
| melo-tts-zh_en                              | Chinese + English             | TTS         | Melo, Chinese + English, 1 speakers |
| kokoro-multi-lang-v1_0                      | Chinese + English             | TTS         | Chinese + English, 53 speakers      |

## Run the app locally
Python 3.10+ is required

```shell
python3 -m venv venv
. venv/bin/activate

pip install -r requirements.txt
python app.py
```

Visit `http://localhost:8000/` to see the demo page

## Build cuda image (for Chinese users)
```shell
docker build -t voiceapi:cuda_dev -f Dockerfile.cuda.cn .
```

## Streaming API (via WebSocket)
### /asr 
Send PCM 16bit audio data to the server, and the server will return the transcription result.
- `samplerate` can be set in the query string, default is 16000. 

The server will return the transcription result in JSON format, with the following fields:
- `text`: the transcription result
- `finished`: whether the segment is finished
- `idx`: the index of the segment

```javascript
    const ws = new WebSocket('ws://localhost:8000/asr?samplerate=16000');
    ws.onopen = () => {
        console.log('connected');
        ws.send('{"sid": 0}');
    };
    ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        const { text, finished, idx } = data;
        // do something with text
        // finished is true when the segment is finished
    };
    // send audio data
    // PCM 16bit, with samplerate
    ws.send(int16Array.buffer);
```
### /tts
Send text to the server, and the server will return the synthesized audio data.
- `samplerate` can be set in the query string, default is 16000. 
- `sid` is the Speaker ID, default is 0.
- `speed` is the speed of the synthesized audio, default is 1.0.
- `chunk_size` is the size of the audio chunk, default is 1024. 

The server will return the synthesized audio data in binary format.
- The audio data is in PCM 16bit format, with the binary data in the response body.
- The server will return the synthesized result with json format, with the following fields:
    - `elapsed`: the elapsed time
    - `progress`: the progress of the synthesis
    - `duration`: the duration of the synthesis
    - `size`: the size of the synthesized audio data

```javascript
    const ws = new WebSocket('ws://localhost:8000/tts?samplerate=16000');
    ws.onopen = () => {
        console.log('connected');
        ws.send('Your text here');
    };
    ws.onmessage = (e) => {
        if (e.data instanceof Blob) {
            // Chunked audio data
            e.data.arrayBuffer().then((arrayBuffer) => {
                const int16Array = new Int16Array(arrayBuffer);
                let float32Array = new Float32Array(int16Array.length);
                for (let i = 0; i < int16Array.length; i++) {
                    float32Array[i] = int16Array[i] / 32768.;
                }
                playNode.port.postMessage({ message: 'audioData', audioData: float32Array });
            });
        } else {
            // The server will return the synthesized result
            const {elapsed, progress, duration, size } = JSON.parse(e.data);
            this.elapsedTime = elapsed;
        }
    };
```

### No Streaming API
#### /tts 
Send text to the server, and the server will return the synthesized audio data.

- `text` is the text to be synthesized.
- `samplerate` can be set in the query string, default is 16000.
- `sid` is the Speaker ID, default is 0.
- `speed` is the speed of the synthesized audio, default is 1.0.
- 
```shell
curl -X POST "http://localhost:8000/tts" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "Hello, world!",
           "sid": 0,
           "samplerate": 16000
         }' -o helloworkd.wav
```

### File Upload API
#### /asr_file
Send an audio file (wav, mp3 or ogg) to the server, and the server will return the transcription with timestamps for each segment.

- `file`: The audio file to transcribe (wav,mp3 or ogg).
- `samplerate`: Target sample rate for processing, default is 16000.

The server will return the transcription results in JSON format, with the following fields:
- `segments`: An array of transcription segments, each containing:
  - `text`: The transcribed text for the segment.
  - `finished`: Always true for file processing.
  - `idx`: The index of the segment.
  - `start`: The start time of the segment in seconds.
  - `end`: The end time of the segment in seconds.
  - `channel`: The index of the channel.

```shell
curl -X POST "http://localhost:8000/asr_file" \
     -F "file=@audio.wav" \
     -o result.json
```

## Download models
All models are stored in the `models` directory
Only download the models you need. default models are:
- asr models: `sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`(Bilingual, Chinese + English). Streaming
- tts models: `vits-zh-hf-theresa` (Chinese + English)

### silero_vad.onnx
> silero is required for ASR
```bash
mkdir -p silero_vad
cd silero_vad
curl -SL -o silero_vad/silero_vad.onnx https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
```

###  FireRedASR-AED-L
```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
```
### kokoro-multi-lang-v1_0
```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2
```

### vits-zh-hf-theresa
```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-zh-hf-theresa.tar.bz2
```

### vits-melo-tts-zh_en
```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2
```
### sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
```bash 
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
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

### sherpa-onnx-paraformer-trilingual-zh-cantonese-en
```bash
curl -SL -O  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-trilingual-zh-cantonese-en.tar.bz2
```
### sherpa-onnx-paraformer-en
```bash
curl -SL -O  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-en-2024-03-09.tar.bz2
```
