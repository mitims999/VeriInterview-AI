"""Opt-in checks of cached Whisper + real openSMILE. No mocks or downloads.

AUDIO_TEST_WAV=/path/to/synthetic-speech.wav python -m pytest tests/test_audio_real.py -q
"""
import io
import json
import os

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from backend.main import app

pytestmark=pytest.mark.skipif(not os.environ.get('AUDIO_TEST_WAV'),reason='Set AUDIO_TEST_WAV to a synthetic speech fixture')


def speech():
    audio,sr=sf.read(os.environ['AUDIO_TEST_WAV'],dtype='float32')
    assert sr==16000 and audio.ndim==1
    return audio


def test_real_upload_transcription_prosody_and_storage():
    audio=speech(); duration=len(audio)/16000
    combined=np.concatenate([audio*.4,audio*.4,audio])
    body=io.BytesIO(); sf.write(body,combined,16000,format='WAV',subtype='FLOAT')
    windows=[dict(question_id=f'Q{i+1}',start_sec=i*duration,end_sec=(i+1)*duration) for i in range(3)]
    with TestClient(app) as client:
        response=client.post('/api/speech-behavior/analyze',data={'answers':json.dumps(windows),'language':'en'},
                             files={'file':('synthetic.wav',body.getvalue(),'audio/wav')})
        assert response.status_code==200,response.text
        result=response.json()
        assert result['transcription']['status']=='COMPLETE'
        assert result['transcription']['segments'][0]['words']
        assert all(a['metrics']['pitch_mean_hz'] is not None for a in result['answers'])
        assert result['speech_behavior']['result']['status']=='OK'
        assert result['speech_behavior']['answers'][-1]['score']>0
        assert any(s['feature']=='rms_dbfs' and s['signed_change']>7
                   for s in result['speech_behavior']['answers'][-1]['signals'])
        assert client.get('/api/audio/analyses/'+result['analysis_id']).json()==result


def test_real_websocket_audio_and_final_word_timestamps():
    audio=speech(); pcm=(np.clip(audio,-1,1)*32767).astype('<i2').tobytes()
    with TestClient(app) as client, client.websocket_connect('/api/audio/live') as ws:
        ws.send_json(dict(action='start',question_id='Q1',sample_rate=16000,format='pcm_s16le',language='en'))
        assert ws.receive_json()['type']=='ready'
        since_update=0; updates=[]
        for index in range(0,len(pcm),16000):
            chunk=pcm[index:index+16000]; ws.send_bytes(chunk); since_update+=len(chunk)//2
            if since_update>=48000:
                updates.append(ws.receive_json()); since_update=0
        ws.send_json({'action':'end_answer'})
        final=ws.receive_json()
        assert updates and all(not u['final'] for u in updates)
        assert final['final'] and final['transcription']['status']=='COMPLETE'
        assert final['transcription']['segments'][0]['words']
        assert final['transcription']['duration_sec']==pytest.approx(len(audio)/16000)
        ws.send_json({'action':'stop'})
        done=ws.receive_json()
        assert done['type']=='complete' and done['result']['analysis_id']
        assert done['result']['transcription']['text']==final['transcription']['text']
