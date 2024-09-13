class PlayerAudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = new Float32Array();
        this.port.onmessage = (event) => {
            let newFetchedData = new Float32Array(this.buffer.length + event.data.audioData.length);
            newFetchedData.set(this.buffer, 0);
            newFetchedData.set(event.data.audioData, this.buffer.length);
            this.buffer = newFetchedData;
        };
    }

    process(inputs, outputs, parameters) {
        const output = outputs[0];
        const channel = output[0];
        const bufferLength = this.buffer.length;
        for (let i = 0; i < channel.length; i++) {
            channel[i] = (i < bufferLength) ? this.buffer[i] : 0;
        }
        this.buffer = this.buffer.slice(channel.length);
        return true;
    }
}

class RecordAudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
    }

    process(inputs, outputs, parameters) {
        const channel = inputs[0][0];
        if (!channel || channel.length === 0) {
            return true;
        }
        const int16Array = new Int16Array(channel.length);
        for (let i = 0; i < channel.length; i++) {
            int16Array[i] = channel[i] * 32767;
        }
        this.port.postMessage({ data: int16Array });
        return true
    }
}

registerProcessor('play-audio-processor', PlayerAudioProcessor);
registerProcessor('record-audio-processor', RecordAudioProcessor);