// Browser resampling is performed by AudioContext at 16 kHz. Send transport
// packets of 2048 mono PCM16 samples; keep the final short packet on flush.
class PCMRecorder extends AudioWorkletProcessor {
  constructor() {
    super(); this.samples = new Int16Array(2048); this.used = 0; this.finished = false;
    this.port.onmessage = (event) => { if (event.data === 'flush') { this.finished = true; this.flush(); this.port.postMessage({ flushed: true }); } };
  }
  flush() {
    if (this.used) {
      // DataView makes the wire format explicitly little endian.
      const buffer = new ArrayBuffer(this.used * 2); const view = new DataView(buffer);
      for (let i = 0; i < this.used; i++) view.setInt16(i * 2, this.samples[i], true);
      this.port.postMessage(buffer, [buffer]); this.used = 0;
    }
  }
  process(inputs) {
    if (this.finished) return true;
    const channel = inputs[0]?.[0];
    if (channel) for (const sample of channel) {
      this.samples[this.used++] = Math.round(Math.max(-1, Math.min(1, sample)) * 32767);
      if (this.used === this.samples.length) this.flush();
    }
    return true;
  }
}
registerProcessor('pcm-recorder', PCMRecorder);
