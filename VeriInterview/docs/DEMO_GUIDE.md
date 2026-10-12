# Demo Guide

## 1. Start the backend

```bash
cd VeriInterview
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

## 2. Start the frontend

```bash
cd VeriInterview/frontend
npm install
npm run dev
```

Open http://localhost:5173.

## 3. Upload & Analyze mode (default tab)

The app opens on **Upload & Analyze**: a form to submit a resume, an ideal-answer/reference
document, an interview transcript, and an optional interview video (for the lip-sync /
face-identity check). Synthetic sample files (no video) ship in
`data/samples/` — pick any one resume/transcript pair against the shared `ideal_answers.txt`:

- `resume_genuine_strong.txt` + `transcript_genuine_strong.txt` → LOW risk, strong capability
- `resume_genuine_weak.txt` + `transcript_genuine_weak.txt` → LOW risk, low capability
- `resume_ai_assisted.txt` + `transcript_ai_assisted.txt` → MEDIUM/HIGH risk (Q3 latency spike)

Leave the video field empty to see lip-sync/face-identity marked UNAVAILABLE (confidence
drops accordingly instead of guessing); attach any short video file to see those two
checks run (deterministically derived from the file's bytes — see
`backend/adapters/video_intake.py` and `docs/LIMITATIONS.md` for why real SyncNet/
InsightFace decoding isn't wired into this environment).

## 4. Demo Candidates mode (dropdown, top right)

| Candidate | Scenario | What to point out |
|---|---|---|
| Ananya Rao | Genuine Strong | Risk LOW (~10), all 6 components LOW, skills Strongly/Verified, capability ~53 |
| Rohit Mehta | Genuine Weak | Risk still LOW (~11) despite low capability (~17) — **the system does not confuse "bad candidate" with "suspicious candidate"** |
| Karan Verma | AI-assisted Simulation | Risk MEDIUM (~51): Voice Modulation HIGH (sudden pace/pause change on Q3), Knowledge Ownership HIGH (fails the counterfactual follow-up despite a polished original answer), Gaze MEDIUM — three independent signals corroborating each other |
| Deep Fake Test Subject | Deepfake / Identity Simulation | Face Identity HIGH (embedding drift) + A/V Lip-Sync HIGH (9-frame offset, low SyncNet confidence) while gaze/speech/timing stay clean — isolates the authenticity question from the behavior question |
| Priya Nair | Nervous Genuine Candidate | Voice Modulation HIGH and Gaze/Behavior MEDIUM (pauses, pitch, off-screen glances) but **overall risk stays LOW** — demonstrates the evidence-fusion rule that one anomalous behavioral channel never dominates the score |

## 5. Key talking points

- **Two separate numbers, never merged:** Integrity Risk (0-100) vs Capability Score — a
  weak-but-honest candidate (Rohit) and a strong genuine candidate (Ananya) both score LOW
  risk, but very different capability.
- **Confidence is separate from risk.** Point at the deepfake candidate: SyncNet's own
  confidence is low (0.35) because the offset is large and unstable — the system surfaces
  that as reduced confidence rather than hiding it.
- **Corroboration, not single-signal verdicts.** Open `backend/modules/evidence_fusion.py`
  and show the weighted formula + the cap that stops one noisy module from dominating +
  the small corroboration boost that only fires when 2+ independent modules land HIGH.
- **Every evidence line is traceable** to a specific module, question, and measurement —
  click through the Evidence Timeline section and read a few messages aloud.
- **Recommended action, never a verdict** — bottom of every report is a human-review
  recommendation string, never "reject" or "cheater."

## 6. If asked to prove it isn't just a static demo

Edit `data/synthetic/genuine.json`'s `Q3` candidate_answer to something empty, refresh the
report (`GET /api/demo/candidates/genuine/report`), and show the Machine Learning skill
verdict and capability score change live — the scoring is computed, not hard-coded.
