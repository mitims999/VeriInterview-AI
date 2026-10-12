import { useEffect, useRef, useState } from 'react';
import { analyzeAudio } from '../audioApi';
import type { AnswerWindow, AudioAnalysis, AudioAnswer, LiveMessage, SpeechReport, Transcript } from '../audioApi';
import './AudioWorkspace.css';

type Phase = 'idle' | 'connecting' | 'recording' | 'finishing' | 'between' | 'saving';
type Capture = { stream: MediaStream; context: AudioContext; node: AudioWorkletNode };
const format = (value: number | null, suffix = '') => value == null ? 'Unavailable' : `${value.toFixed(1)}${suffix}`;

export default function AudioWorkspace() {
  const [tab, setTab] = useState<'file' | 'live'>('file');
  const [file, setFile] = useState<File | null>(null);
  const [windows, setWindows] = useState<AnswerWindow[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<AudioAnalysis | null>(null);
  const [phase, setPhase] = useState<Phase>('idle');
  const [liveText, setLiveText] = useState<Record<string, { text: string; final: boolean }>>({});
  const [question, setQuestion] = useState('Q1');
  const [liveAnswers, setLiveAnswers] = useState<AudioAnswer[]>([]);
  const [liveReport, setLiveReport] = useState<SpeechReport | null>(null);
  const socket = useRef<WebSocket | null>(null);
  const capture = useRef<Capture | null>(null);
  const intentionalClose = useRef(false);
  const flushDone = useRef<(() => void) | null>(null);
  const mounted = useRef(true);
  const elapsedTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  async function stopCapture(flush = true) {
    if (elapsedTimer.current) clearTimeout(elapsedTimer.current);
    const current = capture.current; capture.current = null;
    if (!current) return;
    if (flush) await new Promise<void>(resolve => {
      const timeout = setTimeout(() => { flushDone.current = null; resolve(); }, 500);
      flushDone.current = () => { clearTimeout(timeout); flushDone.current = null; resolve(); };
      current.node.port.postMessage('flush');
    });
    current.node.disconnect(); current.stream.getTracks().forEach(track => track.stop());
    await current.context.close();
  }

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false; intentionalClose.current = true;
      if (elapsedTimer.current) clearTimeout(elapsedTimer.current);
      const current = capture.current;
      if (current) { current.node.disconnect(); current.stream.getTracks().forEach(t => t.stop()); void current.context.close(); }
      socket.current?.close();
    };
  }, []);

  async function beginCapture() {
    let stream: MediaStream | null = null;
    let context: AudioContext | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: {
        channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false,
      } });
      if (!mounted.current || socket.current?.readyState !== WebSocket.OPEN) throw new Error('Audio session closed.');
      context = new AudioContext({ sampleRate: 16000 });
      if (context.sampleRate !== 16000) throw new Error('This browser cannot record at 16 kHz. Try a current Chrome browser.');
      await context.audioWorklet.addModule('/pcm-worklet.js');
      await context.resume();
      if (!mounted.current || socket.current?.readyState !== WebSocket.OPEN) throw new Error('Audio session closed.');
      const node = new AudioWorkletNode(context, 'pcm-recorder');
      const source = context.createMediaStreamSource(stream);
      const mute = context.createGain(); mute.gain.value = 0;
      source.connect(node); node.connect(mute); mute.connect(context.destination);
      node.port.onmessage = event => {
        if (event.data?.flushed) { flushDone.current?.(); return; }
        if (socket.current?.readyState === WebSocket.OPEN) {
          if (socket.current.bufferedAmount > 1024 * 1024) {
            setError('Connection is too slow for live audio. Start a new session.');
            intentionalClose.current = true; void stopCapture(false); socket.current.close(); setPhase('idle');
          } else socket.current.send(event.data);
        }
      };
      capture.current = { stream, context, node };
      setPhase('recording');
      // Leave headroom below the server's hard 120-second limit.
      elapsedTimer.current = setTimeout(() => { void finishAnswer(); }, 115000);
    } catch (e) {
      stream?.getTracks().forEach(track => track.stop());
      if (context && context.state !== 'closed') await context.close();
      setError(e instanceof Error ? e.message : 'Microphone access failed.');
      intentionalClose.current = true; socket.current?.close(); socket.current = null; setPhase('idle');
    }
  }

  function startAnswer() {
    if (!question.trim()) { setError('Enter a question ID.'); return; }
    if (liveAnswers.some(a => a.question_id === question.trim()) && socket.current) {
      setError('Use a new question ID for each answer.'); return;
    }
    setError(''); setPhase('connecting'); setResult(null);
    const control = { action: socket.current ? 'start_answer' : 'start', question_id: question.trim(), sample_rate: 16000, format: 'pcm_s16le' };
    if (socket.current?.readyState === WebSocket.OPEN) { socket.current.send(JSON.stringify(control)); return; }
    setLiveAnswers([]); setLiveReport(null); setLiveText({}); intentionalClose.current = false;
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/audio/live`);
    socket.current = ws;
    ws.onopen = () => ws.send(JSON.stringify(control));
    ws.onmessage = event => {
      if (socket.current !== ws) return;
      const message = JSON.parse(event.data) as LiveMessage;
      if (message.type === 'ready') { void beginCapture(); }
      if (message.type === 'transcript' && message.transcription && message.question_id) {
        const text = message.transcription.text || 'No speech detected.';
        setLiveText(previous => ({ ...previous, [message.question_id!]: { text, final: !!message.final } }));
        if (message.final && message.answer && message.speech_behavior) {
          setLiveAnswers(previous => [...previous, message.answer!]); setLiveReport(message.speech_behavior);
          setPhase('between'); setQuestion(`Q${message.speech_behavior.answers.length + 1}`);
        }
      }
      if (message.type === 'complete' && message.result) {
        intentionalClose.current = true; setResult(message.result); setPhase('idle'); setQuestion('Q1'); socket.current = null;
      }
      if (message.type === 'error') {
        setError(message.message || 'Live session failed.'); intentionalClose.current = true;
        void stopCapture(false); ws.close(); socket.current = null; setPhase('idle');
      }
    };
    ws.onerror = () => setError('Live connection failed. Check that the backend is running.');
    ws.onclose = () => {
      if (socket.current !== ws) return;
      void stopCapture(false);
      if (!intentionalClose.current && mounted.current) { setError('Live session disconnected. Start a new session to continue.'); setPhase('idle'); }
      if (socket.current === ws) socket.current = null;
    };
  }

  async function finishAnswer() {
    setPhase('finishing'); await stopCapture();
    socket.current?.send(JSON.stringify({ action: 'end_answer' }));
  }

  function saveSession() { setPhase('saving'); socket.current?.send(JSON.stringify({ action: 'stop' })); }

  async function submitFile() {
    if (!file) return;
    setError(''); setBusy(true); setResult(null);
    try { setResult(await analyzeAudio(file, windows)); }
    catch (e) { setError(e instanceof Error ? e.message : 'Audio analysis failed.'); }
    finally { setBusy(false); }
  }

  function download() {
    if (!result) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = `audio-analysis-${result.analysis_id || 'session'}.json`; link.click(); URL.revokeObjectURL(url);
  }

  const displayedAnswers = result?.answers || liveAnswers;
  const report = result?.speech_behavior || liveReport;
  const locked = phase !== 'idle';
  const transcript: Transcript | undefined = result?.transcription;

  return <div className="audio-workspace">
    <div className="audio-tabs" role="tablist" aria-label="Audio input">
      <button role="tab" aria-selected={tab === 'file'} disabled={locked} onClick={() => { if (tab !== 'file') { setResult(null); setLiveText({}); setLiveAnswers([]); setLiveReport(null); setError(''); } setTab('file'); }}>Uploaded recording</button>
      <button role="tab" aria-selected={tab === 'live'} disabled={busy} onClick={() => { if (tab !== 'live') { setResult(null); setLiveText({}); setLiveAnswers([]); setLiveReport(null); setError(''); } setTab('live'); }}>Live microphone</button>
    </div>
    <section className="audio-panel">
      <h2>{tab === 'file' ? 'Transcribe and analyze a recording' : 'Record one answer at a time'}</h2>
      <p>Pitch, loudness, pace, and pauses are compared with the speaker’s own earlier answers. Two usable baseline answers and a later answer are needed for an anomaly score.</p>
      {tab === 'file' ? <>
        <label className="audio-file-label">Interview video or audio
          <input type="file" accept="audio/*,video/*" disabled={busy} onChange={e => setFile(e.target.files?.[0] || null)} />
        </label>
        <p className="audio-hint">Add answer boundaries to compare questions. Without boundaries, the whole recording is treated as one answer and only its measurements are available.</p>
        <div className="audio-windows">
          {windows.map((window, i) => <div className="audio-window" key={i}>
            {(['question_id', 'start_sec', 'end_sec'] as const).map(key => <label key={key}>{key === 'question_id' ? 'Question ID' : key === 'start_sec' ? 'Start (seconds)' : 'End (seconds)'}
              <input type={key === 'question_id' ? 'text' : 'number'} min={key === 'question_id' ? undefined : 0} step="0.1" value={window[key]} disabled={busy}
                onChange={e => setWindows(previous => previous.map((w, n) => n === i ? { ...w, [key]: key === 'question_id' ? e.target.value : Number(e.target.value) } : w))} />
            </label>)}
            <button className="button secondary" disabled={busy} onClick={() => setWindows(previous => previous.filter((_, n) => n !== i))}>Remove</button>
          </div>)}
        </div>
        <div className="audio-actions"><button className="button secondary" disabled={busy} onClick={() => setWindows(previous => [...previous, { question_id: `Q${previous.length + 1}`, start_sec: previous.at(-1)?.end_sec || 0, end_sec: (previous.at(-1)?.end_sec || 0) + 10 }])}>Add answer boundary</button>
          <button className="button primary" disabled={!file || busy} onClick={() => void submitFile()}>{busy ? 'Analyzing recording…' : 'Analyze audio'}</button></div>
      </> : <>
        <label className="audio-question">Question ID<input value={question} maxLength={100} disabled={phase !== 'idle' && phase !== 'between'} onChange={e => setQuestion(e.target.value)} /></label>
        <p className="audio-hint">Allow microphone access when prompted. Each answer can be up to two minutes. Provisional text may change until you finish the answer. Keep microphone position and volume consistent.</p>
        <div className="audio-actions">
          {(phase === 'idle' || phase === 'between') && <button className="button primary" onClick={startAnswer}>{phase === 'between' ? 'Record next answer' : 'Start recording'}</button>}
          {phase === 'recording' && <button className="button primary" onClick={() => void finishAnswer()}>Finish answer</button>}
          {phase === 'between' && <button className="button secondary" onClick={saveSession}>Save session</button>}
          {phase !== 'idle' && <span role="status" className="audio-phase">{phase === 'recording' ? '● Microphone recording' : phase === 'between' ? 'Answer finalized' : phase === 'connecting' ? 'Connecting…' : phase === 'saving' ? 'Saving…' : 'Finalizing answer…'}</span>}
        </div>
      </>}
      {error && <p className="audio-error" role="alert">{error}</p>}
    </section>
    {(transcript || Object.keys(liveText).length > 0) && <section className="audio-panel" aria-live="polite">
      <div className="audio-section-heading"><h2>Transcript</h2>{result && <button className="button secondary" onClick={download}>Download results</button>}</div>
      {transcript ? <>{transcript.status === 'NO_SPEECH' ? <p>No speech detected.</p> : transcript.segments.map(segment => <p className="audio-transcript" key={segment.segment_id}><time>{segment.start_sec.toFixed(1)}–{segment.end_sec.toFixed(1)}s</time>{segment.text}</p>)}</> : Object.entries(liveText).map(([id, entry]) => <p className="audio-transcript" key={id}><b>{id} · {entry.final ? 'Final' : 'Provisional'}</b>{entry.text}</p>)}
      {result?.analysis_id && <p className="audio-hint">Results saved locally. Analysis ID: {result.analysis_id}</p>}
    </section>}
    {report && <section className="audio-panel"><h2>Speech behavior</h2>
      <p className="audio-score">{report.result.status === 'OK' ? `${report.result.score.toFixed(1)} / 100 · ${report.result.level}` : 'Building baseline / insufficient usable speech'}</p>
      {report.result.status === 'OK' && <p>Measurement reliability: {Math.round(report.result.confidence * 100)}% · prototype heuristic</p>}
      <p className="audio-hint">Speech changes can reflect thinking, recording conditions, or natural variation. This score does not determine cheating.</p>
      {displayedAnswers.map(answer => { const score = report.answers.find(a => a.question_id === answer.question_id); return <details className="audio-answer" key={answer.question_id}>
        <summary>{answer.question_id}<span>{report.baseline_question_ids.includes(answer.question_id) ? 'Baseline answer' : score?.score == null ? 'Insufficient evidence' : `${score.score.toFixed(1)} / 100`}</span></summary>
        <dl className="audio-metrics"><div><dt>Mean pitch</dt><dd>{format(answer.metrics.pitch_mean_hz, ' Hz')}</dd></div><div><dt>Recording level</dt><dd>{format(answer.metrics.rms_dbfs, ' dBFS')}</dd></div><div><dt>Speech rate</dt><dd>{format(answer.metrics.speech_rate_wpm, ' WPM')}</dd></div><div><dt>Internal pauses</dt><dd>{answer.metrics.pause_count}</dd></div><div><dt>Speech ratio</dt><dd>{Math.round(answer.metrics.speech_ratio * 100)}%</dd></div><div><dt>Audio quality</dt><dd>{Math.round(answer.quality.score * 100)}%</dd></div></dl>
        <ul>{score?.evidence.map((text, i) => <li key={i}>{text}</li>)}</ul>
        <p className="audio-hint">{answer.quality.warnings.join(' ')}</p>
      </details>; })}
    </section>}
  </div>;
}
