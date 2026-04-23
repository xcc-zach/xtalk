// VAD Audio Worklet Processor with Resampling
class VADProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        const opts = options.processorOptions || {};
        this.targetSampleRate = opts.targetSampleRate || 16000;
        this.targetFrameSize = opts.targetFrameSize || 512;
        this.nativeSampleRate = sampleRate; // AudioWorklet's global sampleRate
        
        this.inputBuffer = [];
        this.outputBuffer = [];
    }

    // Simple linear interpolation resampler
    resample(inputData) {
        if (this.nativeSampleRate === this.targetSampleRate) {
            return inputData;
        }

        const ratio = this.nativeSampleRate / this.targetSampleRate;
        const outputLength = Math.floor(inputData.length / ratio);
        const output = new Float32Array(outputLength);

        for (let i = 0; i < outputLength; i++) {
            const pos = i * ratio;
            const index = Math.floor(pos);
            const frac = pos - index;
            
            const sample1 = inputData[index] || 0;
            const sample2 = inputData[Math.min(index + 1, inputData.length - 1)] || sample1;
            
            output[i] = sample1 + (sample2 - sample1) * frac;
        }

        return output;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        
        if (input && input.length > 0) {
            const inputChannel = input[0];
            
            for (let i = 0; i < inputChannel.length; i++) {
                this.inputBuffer.push(inputChannel[i]);
            }
            
            const minInputSamples = Math.ceil(this.targetFrameSize * this.nativeSampleRate / this.targetSampleRate);
            
            while (this.inputBuffer.length >= minInputSamples) {
                const chunk = this.inputBuffer.splice(0, minInputSamples);
                const resampled = this.resample(chunk);
                
                for (let i = 0; i < resampled.length; i++) {
                    this.outputBuffer.push(resampled[i]);
                }
                
                while (this.outputBuffer.length >= this.targetFrameSize) {
                    const frameData = this.outputBuffer.splice(0, this.targetFrameSize);
                    const frame = new Float32Array(frameData);
                    
                    this.port.postMessage({
                        type: 'audioFrame',
                        frame: frame
                    }, [frame.buffer]);
                }
            }
        }
        
        return true;
    }
}

registerProcessor('vad-processor', VADProcessor);
