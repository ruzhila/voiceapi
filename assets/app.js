const demoapp = {
    text: '讲个冷笑话吧，要很好笑的那种。',
    recording: false,
    asrWS: null,
    currentText: null,
    disabled: false,
    elapsedTime: null,
    logs: [{ idx: 0, text: 'Happily here at ruzhila.cn.' }],
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

        this.disabled = true;
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
                this.elapsedTime = JSON.parse(e.data)?.elapsed;
                this.disabled = false;
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
        if (this.currentText) {
            this.logs.push({ idx: this.logs.length + 1, text: this.currentText });
        }
        this.currentText = null;

    },

    async doasr() {
        const audioConstraints = {
            video: false,
            audio: true,
        };

        const mediaStream = await navigator.mediaDevices.getUserMedia(audioConstraints);

        const ws = new WebSocket('/asr');
        let currentMessage = '';

        ws.onopen = () => {
            this.logs = [];
        };

        ws.onmessage = (e) => {
            const data = JSON.parse(e.data);
            const { text, finished, idx } = data;

            currentMessage = text;
            this.currentText = text

            if (finished) {
                this.logs.push({ text: currentMessage, idx: idx });
                currentMessage = '';
                this.currentText = null
            }
        };

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
