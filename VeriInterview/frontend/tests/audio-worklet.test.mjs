import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

test('worklet preserves the final short packet and sends explicit PCM16 little endian', () => {
  let Recorder;
  const sent = [];
  class Processor { constructor() { this.port = { postMessage: message => sent.push(message) }; } }
  vm.runInNewContext(readFileSync(new URL('../public/pcm-worklet.js', import.meta.url), 'utf8'), {
    AudioWorkletProcessor: Processor, registerProcessor: (name, constructor) => { Recorder = constructor; },
  });
  const recorder = new Recorder();
  const samples = new Float32Array(2148).fill(.5);
  recorder.process([[samples]]);
  assert.equal(sent.length, 1);
  recorder.port.onmessage({ data: 'flush' });
  assert.equal(sent[0].byteLength, 4096);
  assert.equal(sent[1].byteLength, 200);
  assert.equal(new DataView(sent[0]).getInt16(0, true), 16384);
  assert.equal(sent[2].flushed, true);
  recorder.process([[samples]]);
  assert.equal(sent.length, 3, 'no audio may follow finalization');
});

test('worklet clips out-of-range samples without integer wraparound', () => {
  let Recorder; const sent = [];
  class Processor { constructor() { this.port = { postMessage: message => sent.push(message) }; } }
  vm.runInNewContext(readFileSync(new URL('../public/pcm-worklet.js', import.meta.url), 'utf8'), {
    AudioWorkletProcessor: Processor, registerProcessor: (_, constructor) => { Recorder = constructor; },
  });
  const recorder = new Recorder();
  recorder.process([[new Float32Array([2, -2])]]);
  recorder.port.onmessage({ data: 'flush' });
  const view = new DataView(sent[0]);
  assert.equal(view.getInt16(0, true), 32767);
  assert.equal(view.getInt16(2, true), -32767);
});
