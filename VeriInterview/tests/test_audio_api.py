import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app
from backend import audio_routes
from backend.database.models import Base, Candidate, Interview, Question
from backend.database.session import get_db


@pytest.fixture
def client(monkeypatch):
    engine = create_engine('sqlite://', connect_args={'check_same_thread':False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    def dependency():
        with factory() as db:
            yield db
    app.dependency_overrides[get_db] = dependency
    monkeypatch.setattr(audio_routes, 'SessionLocal', factory)
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()
    engine.dispose()


def silence(seconds=4):
    data = io.BytesIO()
    sf.write(data,np.zeros(seconds*16000),16000,format='WAV',subtype='FLOAT')
    return data.getvalue()


def test_upload_transcription_silence_and_invalid_media(client):
    r=client.post('/api/audio/transcribe',files={'file':('silence.wav',silence(),'audio/wav')})
    assert r.status_code==200 and r.json()['status']=='NO_SPEECH'
    r=client.post('/api/audio/transcribe',files={'file':('bad.mp4',b'not a video')})
    assert r.status_code==422


def test_analysis_persists_and_returns_unavailable_for_silence(client):
    r=client.post('/api/speech-behavior/analyze',files={'file':('silence.wav',silence())})
    assert r.status_code==200, r.text
    result=r.json()
    assert result['speech_behavior']['result']['level']=='UNAVAILABLE'
    assert result['answers'][0]['metrics']['pitch_mean_hz'] is None
    saved=client.get('/api/audio/analyses/'+result['analysis_id'])
    assert saved.status_code==200 and saved.json()==result


def test_answer_windows_must_be_valid(client):
    for windows in ['not json',json.dumps([{'question_id':'Q1','start_sec':0,'end_sec':8}]),
                    json.dumps([{'question_id':'Q1','start_sec':0,'end_sec':3},
                                {'question_id':'Q2','start_sec':2,'end_sec':4}])]:
        r=client.post('/api/speech-behavior/analyze',data={'answers':windows},
                      files={'file':('silence.wav',silence())})
        assert r.status_code==422


def test_live_stream_revisions_finalize_flush_and_persist(client):
    with client.websocket_connect('/api/audio/live') as ws:
        ws.send_json({'action':'start','question_id':'Q1','format':'pcm_s16le','sample_rate':16000})
        assert ws.receive_json()['type']=='ready'
        ws.send_bytes(b'\0\0'*32000)
        ws.send_bytes(b'\0\0'*16000)
        update=ws.receive_json()
        assert update['final'] is False and update['transcription']['status']=='NO_SPEECH'
        ws.send_bytes(b'\0\0'*8000) # shorter final chunk must not be lost
        ws.send_json({'action':'end_answer'})
        final=ws.receive_json()
        assert final['final'] is True
        assert final['transcription']['duration_sec']==3.5
        ws.send_json({'action':'start_answer','question_id':'Q2','format':'pcm_s16le','sample_rate':16000})
        assert ws.receive_json()['type']=='ready'
        ws.send_bytes(b'\0\0'*8000)
        ws.send_json({'action':'stop'})
        assert ws.receive_json()['final'] is True
        done=ws.receive_json()
        assert done['type']=='complete'
        assert len(done['result']['answers'])==2
        assert done['result']['answers'][1]['start_sec']==3.5
        assert client.get('/api/audio/analyses/'+done['result']['analysis_id']).status_code==200


def test_live_rejects_wrong_audio_format_and_untrusted_origin(client):
    with client.websocket_connect('/api/audio/live') as ws:
        ws.send_json({'action':'start','question_id':'Q1','format':'webm','sample_rate':48000})
        assert ws.receive_json()['type']=='error'
    with pytest.raises(Exception):
        with client.websocket_connect('/api/audio/live',headers={'origin':'https://example.org'}):
            pass


def test_existing_interview_report_consumes_attached_audio(client):
    # Verify the connection to the existing pipeline without changing its owner.
    with audio_routes.SessionLocal() as db:
        c=Candidate(name='Audio Test'); db.add(c); db.flush()
        i=Interview(candidate_id=c.id); db.add(i); db.flush()
        questions=[]
        for n in range(3):
            q=Question(interview_id=i.id,question_text='Explain Python',order_index=n)
            db.add(q); db.flush(); questions.append(q.id)
        db.commit(); interview_id=i.id
    windows=[{'question_id':str(q),'start_sec':n*4,'end_sec':(n+1)*4} for n,q in enumerate(questions)]
    r=client.post('/api/speech-behavior/analyze',data={'answers':json.dumps(windows),'interview_id':interview_id},
                  files={'file':('silence.wav',silence(12))})
    assert r.status_code==200, r.text
    report=client.get(f'/api/interviews/{interview_id}/report')
    assert report.status_code==200
    assert report.json()['assessment']['components']['speech_behavior_risk']['status']=='INSUFFICIENT_EVIDENCE'
