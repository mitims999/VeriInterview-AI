"""
Parses the three plain-text intake documents an interviewer uploads through the
"New Analysis" screen: the reference/ideal-answer document, the candidate
transcript, and (loosely) resume claims. Kept as simple `[Qn]` block files so a
non-technical interviewer can prepare them in any text editor — matching
sample files ship in data/samples/.
"""
from __future__ import annotations
import re

_BLOCK_RE = re.compile(r"\[(Q\d+)\]\s*\n(.*?)(?=\n\[Q\d+\]|\Z)", re.DOTALL)


def _parse_blocks(text: str) -> dict[str, dict[str, str]]:
    blocks: dict[str, dict[str, str]] = {}
    for qid, body in _BLOCK_RE.findall(text.replace("\r\n", "\n") + "\n"):
        fields: dict[str, str] = {}
        for line in body.strip().splitlines():
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            fields[key.strip().lower()] = val.strip()
        blocks[qid] = fields
    return blocks


def parse_ideal_answers(text: str) -> dict[str, dict]:
    """[Qn] blocks with: skill, difficulty, expected_proficiency, question, reference_answer, concepts"""
    out = {}
    for qid, fields in _parse_blocks(text).items():
        out[qid] = {
            "skill": fields.get("skill", "General"),
            "difficulty": fields.get("difficulty", "Basic"),
            "expected_proficiency": fields.get("expected_proficiency", "Intermediate"),
            "question_text": fields.get("question", ""),
            "reference_answer": fields.get("reference_answer", ""),
            "required_concepts": [c.strip() for c in fields.get("concepts", "").split(",") if c.strip()],
        }
    return out


def parse_transcript(text: str) -> dict[str, dict]:
    """[Qn] blocks with: answer, latency, duration"""
    out = {}
    for qid, fields in _parse_blocks(text).items():
        out[qid] = {
            "candidate_answer": fields.get("answer", ""),
            "response_latency_sec": float(fields.get("latency", 2.0) or 2.0),
            "answer_duration_sec": float(fields.get("duration", 30.0) or 30.0),
        }
    return out


def build_qa_list(ideal: dict[str, dict], transcript: dict[str, dict]) -> list[dict]:
    qa_list = []
    for qid in sorted(ideal.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else x):
        ans = transcript.get(qid, {"candidate_answer": "", "response_latency_sec": 0.0, "answer_duration_sec": 0.0})
        qa_list.append({"question_id": qid, **ideal[qid], **ans})
    return qa_list


_STRONG_VERBS = ("designed", "built", "led", "architected", "deployed", "optimized", "shipped", "implemented")
_MEDIUM_VERBS = ("experience with", "worked with", "used", "developed", "contributed to")
_WEAK_VERBS = ("familiar with", "exposure to", "basic knowledge of", "learning")


def infer_resume_claims(resume_text: str, known_skills: list[str]) -> list[dict]:
    """
    Very small heuristic: split the resume into sentences, and for each sentence
    that mentions a skill already used in the question bank, classify claim
    strength from its verb. A real deployment would use an LLM extractor here
    (Master Spec Section 31 — LLM only ever generates/extracts text, never scores).
    """
    claims = []
    sentences = re.split(r"(?<=[.!?])\s+", resume_text)
    for sentence in sentences:
        sl = sentence.lower()
        for skill in known_skills:
            if skill.lower() in sl:
                if any(v in sl for v in _STRONG_VERBS):
                    strength = "strong"
                elif any(v in sl for v in _MEDIUM_VERBS):
                    strength = "medium"
                elif any(v in sl for v in _WEAK_VERBS):
                    strength = "weak"
                else:
                    strength = "medium"
                claims.append({"skill": skill, "claim_text": sentence.strip(), "claim_strength": strength})
                break
    return claims
