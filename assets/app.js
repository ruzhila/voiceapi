document.addEventListener('alpine:init', () => {
    Alpine.data('demoapp', () => ({
        text: '讲个冷笑话吧，要很好笑的那种。',
        recording: false,
        asrWS: null,
        currentText: null,
        disabled: false,
        elapsedTime: null,
        logs: [{ idx: 0, text: 'Happily here at ruzhila.cn.' }],
        file: null,
        fileResults: [],
        fileElapsed: null,
        fileSize: null,
        fileAudioDuration: null,
        fileRtf: null,
        fileModel: null,
        useInt8: false,
        formatBytes(bytes) {
            if (typeof bytes !== 'number' || !isFinite(bytes)) {
                return '-';
            }
            if (bytes === 0) {
                return '0 B';
            }
            const units = ['B', 'KB', 'MB', 'GB', 'TB'];
            const exponent = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
            const value = bytes / Math.pow(1024, exponent);
            const digits = value >= 10 ? 0 : 1;
            return `${value.toFixed(digits)} ${units[exponent]}`;
        },
        formatSeconds(value) {
            if (typeof value !== 'number' || !isFinite(value)) {
                return '-';
            }
            if (value === 0) {
                return '0s';
            }
            const digits = value >= 10 ? 1 : 2;
            return `${value.toFixed(digits)}s`;
        },
        async init() {
        },
        async dotts() {
            const audioContext = new AudioContext({ sampleRate: 16000 });
            await audioContext.audioWorklet.addModule('./audio_process.js');

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
                        const float32Array = new Float32Array(int16Array.length);
                        for (let i = 0; i < int16Array.length; i++) {
                            float32Array[i] = int16Array[i] / 32768.;
                        }
                        playNode.port.postMessage({ message: 'audioData', audioData: float32Array });
                    });
                } else {
                    this.elapsedTime = JSON.parse(e.data)?.elapsed;
                    this.disabled = false;
                }
            };
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
                this.currentText = text;

                if (finished) {
                    this.logs.push({ text: currentMessage, idx: idx });
                    currentMessage = '';
                    this.currentText = null;
                }
            };

            const audioContext = new AudioContext({ sampleRate: 16000 });
            await audioContext.audioWorklet.addModule('./audio_process.js');

            const recordNode = new AudioWorkletNode(audioContext, 'record-audio-processor');
            recordNode.connect(audioContext.destination);
            recordNode.port.onmessage = (event) => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    const int16Array = event.data.data;
                    ws.send(int16Array.buffer);
                }
            };
            const source = audioContext.createMediaStreamSource(mediaStream);
            source.connect(recordNode);
            this.asrWS = ws;
            this.recording = true;
        },

        async uploadFile() {
            if (!this.file) {
                alert('Please select a file first.');
                return;
            }

            const formData = new FormData();
            formData.append('file', this.file);

            this.fileElapsed = null;
            this.fileSize = null;
            this.fileResults = [];
            this.fileAudioDuration = null;
            this.fileRtf = null;
            this.fileModel = null;

            try {
                const endpoint = this.useInt8 ? '/asr_file?use_int8=true' : '/asr_file';
                const response = await fetch(endpoint, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const result = await response.json();
                this.fileResults = (result.segments || []).slice().sort((a, b) => {
                    if (a.start === b.start) {
                        return (a.channel ?? 0) - (b.channel ?? 0);
                    }
                    return a.start - b.start;
                });
                this.fileElapsed = typeof result.elapsed === 'number' ? result.elapsed : null;
                this.fileSize = typeof result.data_length === 'number' ? result.data_length : null;
                this.fileAudioDuration = typeof result.audio_duration === 'number' ? result.audio_duration : null;
                this.fileRtf = typeof result.rtf === 'number' ? result.rtf : null;
                this.fileModel = typeof result.asr_model === 'string' ? result.asr_model : null;
            } catch (error) {
                console.error('Error uploading file:', error);
                alert('Error uploading file: ' + error.message);
            }
        }
    }));
});
