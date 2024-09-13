const demoapp = {
    text: '您好，有什么我可以帮助您的吗？',
    recording: false,
    asrWS: null,
    async init() {
    },
    async dotts() {
        let audioContext = new AudioContext({ sampleRate: 16000 })
        await audioContext.audioWorklet.addModule('./audio_process.js')

        const ws = new WebSocket('/tts');
        ws.onopen = () => {
            ws.send(this.text);
        };
        const playNode = new AudioWorkletNode(audioContext, 'play-audio-processor');
        playNode.connect(audioContext.destination);

        ws.onmessage = async (e) => {
            if (e.data instanceof Blob) {
                e.data.arrayBuffer().then((arrayBuffer) => {
                    const int16Array = new Int16Array(arrayBuffer);
                    let float32Array = new Float32Array(int16Array.length);
                    for (let i = 0; i < int16Array.length; i++) {
                        float32Array[i] = int16Array[i] / 32768.;
                    }
                    playNode.port.postMessage({ message: 'audioData', audioData: float32Array });
                });
            } else {
                console.log(e.data)
            }
        }
    },

    async stopasr() {
        if (!this.asrWS) {
            return;
        }
        this.asrWS.close();
        this.asrWS = null;
        this.recording = false;
    },

    async doasr() {
        const audioConstraints = {
            video: false,
            audio: true,
        };

        const mediaStream = await navigator.mediaDevices.getUserMedia(audioConstraints);

        const ws = new WebSocket('/asr');
        ws.onopen = () => {
        };

        ws.onmessage = (e) => {
            console.log(e.data);
        }

        let audioContext = new AudioContext({ sampleRate: 16000 })
        await audioContext.audioWorklet.addModule('./audio_process.js')

        const recordNode = new AudioWorkletNode(audioContext, 'record-audio-processor');
        recordNode.connect(audioContext.destination);
        recordNode.port.onmessage = (event) => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                const int16Array = event.data.data;
                ws.send(int16Array.buffer);
            }
        }
        const source = audioContext.createMediaStreamSource(mediaStream);
        source.connect(recordNode);
        this.asrWS = ws;
        this.recording = true;
    }
}
