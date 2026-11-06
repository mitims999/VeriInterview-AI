import type { ModuleResult } from './api';

export interface AnswerWindow { question_id: string; start_sec: number; end_sec: number }
export interface Transcript {
  status: 'COMPLETE' | 'NO_SPEECH'; text: string; duration_sec: number;
  segments: { segment_id: string; start_sec: number; end_sec: number; text: string;
    words: { word: string; start_sec: number; end_sec: number; probability: number }[] }[];
}
export interface AudioAnswer {
  question_id: string; start_sec: number; end_sec: number;
  metrics: { pitch_mean_hz: number | null; rms_dbfs: number; speech_rate_wpm: number | null;
    pause_count: number; pause_longest_sec: number; speech_ratio: number };
  quality: { score: number; warnings: string[] };
}
export interface SpeechReport {
  result: ModuleResult; baseline_question_ids: string[];
  answers: { question_id: string; status: string; score: number | null; confidence: number; evidence: string[] }[];
}
export interface AudioAnalysis {
  analysis_id?: string; transcription: Transcript; answers: AudioAnswer[]; speech_behavior: SpeechReport;
}
export interface LiveMessage {
  type: 'ready' | 'transcript' | 'complete' | 'error'; question_id?: string;
  final?: boolean; transcription?: Transcript; answer?: AudioAnswer;
  speech_behavior?: SpeechReport; result?: AudioAnalysis; message?: string;
}

export async function analyzeAudio(file: File, windows: AnswerWindow[]): Promise<AudioAnalysis> {
  const body = new FormData(); body.append('file', file);
  body.append('answers', JSON.stringify(windows));
  const response = await fetch('/api/speech-behavior/analyze', { method: 'POST', body });
  if (!response.ok) {
    let detail = 'Audio analysis failed. Check that the backend is running.';
    try { const result = await response.json(); if (typeof result.detail === 'string') detail = result.detail; } catch { /* proxy error */ }
    throw new Error(detail);
  }
  return response.json();
}
