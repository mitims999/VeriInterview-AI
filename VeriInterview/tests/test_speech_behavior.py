from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend.audio_schemas import AnswerAudioFeatures, AcousticMetrics, AudioQuality, TimeRange
from backend.adapters import opensmile_adapter
from backend.modules.speech_behavior.quality import assess_quality
from backend.modules.speech_behavior.metrics import timing_metrics


def answer(q, *, pitch=150, rate=140, level=-20, quality=.9, start=0):
    return AnswerAudioFeatures(question_id=q, start_sec=start, end_sec=start+10,
        metrics=AcousticMetrics(pitch_mean_hz=pitch, pitch_std_hz=15, rms_dbfs=level,
            loudness_std=.1, voiced_frame_ratio=.7, speech_duration_sec=8,
            silence_duration_sec=2, speech_ratio=.8, pause_ratio=.1, pause_count=2,
            pause_mean_sec=.5, pause_median_sec=.5, pause_longest_sec=.6, speech_rate_wpm=rate),
        quality=AudioQuality(score=quality, duration_sec=10, speech_duration_sec=8,
                             clipping_ratio=0, rms_dbfs=level))


def test_baseline_is_not_scored_against_itself():
    result = opensmile_adapter.analyze_answers([answer('Q1'), answer('Q2', start=10)])
    assert result.result.status == 'INSUFFICIENT_EVIDENCE'
    assert result.result.confidence == 0
    assert all(a.score is None for a in result.answers)


def test_normal_speech_and_zero_variance_baseline():
    result = opensmile_adapter.analyze_answers([
        answer('Q1'), answer('Q2', start=10), answer('Q3', start=20, pitch=151, rate=142)])
    assert result.result.level == 'LOW'
    assert result.baseline_question_ids == ['Q1','Q2']
    assert result.answers[2].score < 10
    assert opensmile_adapter.compute([a.model_dump() for a in
        [answer('Q1'), answer('Q2', start=10), answer('Q3', start=20)]]).level == 'LOW'


def test_deviation_is_explained_and_has_timestamp():
    result = opensmile_adapter.analyze_answers([
        answer('Q1'), answer('Q2', start=10), answer('Q3', start=20, pitch=230, rate=230, level=-8)])
    assert result.answers[-1].score > 40
    assert result.result.evidence[0].question_id == 'Q3'
    assert result.result.evidence[0].timestamp == 20
    assert all('cheat' not in e.message.lower() for e in result.result.evidence)


def test_low_quality_answer_is_unavailable_not_high_risk():
    report = opensmile_adapter.analyze_answers([
        answer('Q1'), answer('Q2', start=10), answer('Q3', start=20, pitch=300, quality=.1)])
    assert report.result.status == 'INSUFFICIENT_EVIDENCE'
    assert report.answers[-1].score is None


def test_lower_quality_reduces_confidence():
    baseline = [answer('Q1'), answer('Q2', start=10)]
    clean = opensmile_adapter.analyze_answers(baseline+[answer('Q3', start=20)])
    noisy = opensmile_adapter.analyze_answers(baseline+[answer('Q3', start=20, quality=.4)])
    assert noisy.result.confidence < clean.result.confidence


def test_absolute_pitch_does_not_determine_anomaly():
    low = opensmile_adapter.analyze_answers([answer(f'Q{i}',start=i*10,pitch=100) for i in range(3)])
    high = opensmile_adapter.analyze_answers([answer(f'Q{i}',start=i*10,pitch=280) for i in range(3)])
    assert low.result.score == high.result.score == 0


def test_timing_uses_internal_pauses_and_two_rate_definitions():
    result = timing_metrics(10, [TimeRange(start_sec=1,end_sec=3), TimeRange(start_sec=5,end_sec=9)], 20)
    assert result['speech_rate_wpm'] == 120
    assert result['articulation_rate_wpm'] == 200
    assert result['pause_count'] == 1 and result['pause_longest_sec'] == 2
    assert result['speech_ratio'] == .6
    assert result['pause_ratio'] == .25


def test_quality_penalizes_noise_clipping_and_short_audio():
    rng = np.random.default_rng(17)
    signal = np.zeros(160000,dtype=np.float32)
    signal[16000:144000] = .2*np.sin(2*np.pi*150*np.arange(128000)/16000)
    speech = [TimeRange(start_sec=1,end_sec=9)]
    clean = assess_quality(signal,speech)
    noisy = assess_quality(signal+rng.normal(0,.15,len(signal)),speech)
    clipped = assess_quality(signal*20,speech)
    short = assess_quality(signal[16000:20000],[TimeRange(start_sec=0,end_sec=.25)])
    assert noisy.score < clean.score
    assert clipped.score < clean.score
    assert short.score < clean.score


def test_real_silent_extraction_has_missing_pitch(tmp_path):
    path=tmp_path/'silence.wav'
    sf.write(path,np.zeros(32000),16000,subtype='FLOAT')
    result=opensmile_adapter.extract_features(path,'Q1')
    assert result.metrics.pitch_mean_hz is None
    assert result.quality.score == 0
    assert result.metrics.speech_rate_wpm is None


def test_duplicate_or_overlapping_answers_rejected():
    with pytest.raises(ValueError,match='unique'):
        opensmile_adapter.analyze_answers([answer('Q1'),answer('Q1',start=10)])
    with pytest.raises(ValueError,match='overlap'):
        opensmile_adapter.analyze_answers([answer('Q1'),answer('Q2',start=5)])


def test_real_opensmile_pitch_conversion_and_gain_change(tmp_path, monkeypatch):
    from backend.modules.speech_behavior import extractor
    monkeypatch.setattr(extractor, 'speech_intervals', lambda _: [TimeRange(start_sec=0,end_sec=4)])
    t=np.arange(64000)/16000
    # Harmonic carrier has a known F0; VAD is bypassed only for this unit check.
    audio=.15*np.sin(2*np.pi*150*t)+.05*np.sin(2*np.pi*300*t)
    sf.write(tmp_path/'low.wav',audio,16000,subtype='FLOAT')
    sf.write(tmp_path/'high.wav',audio*2,16000,subtype='FLOAT')
    low=extractor.extract_features(tmp_path/'low.wav','Q1',word_count=10)
    high=extractor.extract_features(tmp_path/'high.wav','Q2',word_count=10)
    assert low.metrics.pitch_median_hz == pytest.approx(150,abs=8)
    assert high.metrics.rms_dbfs-low.metrics.rms_dbfs == pytest.approx(6.0206,abs=.01)
    assert high.metrics.loudness_mean > low.metrics.loudness_mean


def test_high_new_audio_score_alone_does_not_create_high_overall_risk():
    import json
    from backend.modules.pipeline import run_full_analysis
    record=json.loads((Path(__file__).resolve().parents[1]/'data/synthetic/genuine.json').read_text())
    changed=answer('Q3',start=20,pitch=300,rate=280,level=-5)
    changed.metrics.pitch_std_hz=90
    changed.metrics.loudness_std=1
    changed.metrics.pause_ratio=.6
    changed.metrics.speech_ratio=.3
    record['speech_features']=[a.model_dump() for a in [answer('Q1'),answer('Q2',start=10),changed]]
    report=run_full_analysis(record)
    assert report['assessment']['components']['speech_behavior_risk']['level']=='HIGH'
    assert report['assessment']['risk_level']!='HIGH'
