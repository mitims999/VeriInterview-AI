Sample intake files for the "Upload & Analyze" screen.

Mix and match any resume_*.txt with any transcript_*.txt against the shared
ideal_answers.txt reference document to see different risk outcomes. No video
files are provided — the video upload field is optional; leave it empty to see
lip-sync/identity marked UNAVAILABLE, or upload any short video file yourself
to see the (deterministic placeholder) lip-sync pipeline run end-to-end.

Files:
  ideal_answers.txt            - upload as "Ideal Answers" for any pairing below
  resume_genuine_strong.txt    + transcript_genuine_strong.txt   -> expect LOW risk, strong capability
  resume_genuine_weak.txt      + transcript_genuine_weak.txt     -> expect LOW risk, low capability
  resume_ai_assisted.txt       + transcript_ai_assisted.txt      -> expect MEDIUM/HIGH risk (Q3 latency spike + polished answer)

Format of ideal_answers.txt (one [Qn] block per question):
  [Q1]
  skill: <skill name>
  difficulty: Basic|Intermediate|Advanced
  expected_proficiency: <text>
  question: <question text>
  reference_answer: <model answer>
  concepts: <comma-separated required concepts>

Format of transcript_*.txt (one [Qn] block per question, matching question ids):
  [Q1]
  answer: <candidate's spoken answer, transcribed>
  latency: <seconds from question end to first speech>
  duration: <answer duration in seconds>
