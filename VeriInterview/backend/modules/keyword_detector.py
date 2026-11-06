"""
Feature F8 — Keyword Detection, Buzzword-Stuffing Detector, and Resume-Interview Cross-Referencing.

Analyses a resume for:
  1. Buzzword density — concentration of trendy keywords without project context.
  2. Hollow claims — vague self-descriptors ("results-oriented", "synergy").
  3. Cross-referencing:
     - Separates skills into TESTED skills (discussed in interview) and UNTESTED skills (not covered).
     - Does NOT penalize candidates for skills not asked in the interview.
     - Flags MISMATCHES when a claimed skill is tested but the candidate fails or gives hollow answers.
     - Confirms MATCHES when candidate answers demonstrate real technical depth.
"""
from __future__ import annotations
import re
from backend.adapters.sentence_embed import semantic_similarity

_HOLLOW_PHRASES = {
    "results-oriented", "results-driven", "self-motivated", "team player",
    "go-getter", "synergy", "value-add", "thought leader", "paradigm shift",
    "dynamic", "passionate", "guru", "ninja", "rockstar", "wizard",
    "detail-oriented", "fast learner", "hard worker", "proactive",
    "strategic thinker", "innovative thinker", "seasoned professional",
    "best-in-class", "cutting-edge", "world-class", "leveraged",
    "out-of-the-box", "game-changer",
}

_TRENDY_BUZZWORDS = {
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "blockchain", "web3", "metaverse", "cloud-native", "microservices",
    "devops", "mlops", "data-driven", "big data", "iot",
    "generative ai", "llm", "large language model", "transformer",
    "kubernetes", "docker", "serverless", "edge computing",
    "agile", "scrum", "ci/cd", "devsecops",
}

_SUBSTANTIATING_PATTERNS = [
    r"(?:built|designed|implemented|deployed|led|architected|shipped|developed|created|maintained)\b",
    r"\d+\s*(?:years?|months?|yrs?)\b",
    r"(?:project|product|system|service|platform|application|pipeline)\b",
    r"(?:team of|managed|mentored|supervised)\b",
    r"(?:increased|reduced|improved|optimized|scaled|migrated)\b.*?\d",
]

_TECH_SYNONYMS: dict[str, list[str]] = {
    "aws": ["ec2", "s3", "lambda", "dynamodb", "sqs", "rds", "cloudwatch", "iam", "step functions", "eks", "vpc"],
    "docker": ["container", "dockerfile", "docker-compose", "image", "hypervisor", "kernel"],
    "kubernetes": ["k8s", "pod", "deployment", "service", "helm", "cluster", "ingress"],
    "python": ["gil", "decorator", "generator", "functools", "fastapi", "flask", "django", "asyncio"],
    "react": ["virtual dom", "component", "hooks", "usestate", "redux", "jsx"],
    "sql": ["query", "index", "table scan", "composite", "join", "postgres", "mysql", "execution plan"],
    "machine learning": ["bias", "variance", "overfitting", "scikit-learn", "model", "training", "mlflow"],
    "deep learning": ["neural network", "transformer", "backpropagation", "cnn", "rnn", "weights"],
    "rest api": ["endpoint", "jwt", "auth", "http", "status code", "payload", "bearer"],
    "system design": ["scale", "scaling", "load balancer", "cache", "redis", "sharding", "latency", "cdn"],
}


def _count_hollow(text: str) -> list[dict]:
    text_lower = text.lower()
    found = []
    for phrase in _HOLLOW_PHRASES:
        if phrase in text_lower:
            found.append({
                "phrase": phrase,
                "type": "hollow_claim",
                "severity": "medium",
                "message": f'"{phrase}" is a vague self-descriptor with no verifiable signal.',
            })
    return found


def _count_buzzword_density(text: str) -> dict:
    text_lower = text.lower()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    buzzwords_found = []
    substantiated = []
    unsubstantiated = []

    for bw in _TRENDY_BUZZWORDS:
        if bw in text_lower:
            buzzwords_found.append(bw)
            is_sub = False
            for sent in sentences:
                if bw in sent.lower():
                    for pattern in _SUBSTANTIATING_PATTERNS:
                        if re.search(pattern, sent, re.IGNORECASE):
                            is_sub = True
                            break
                if is_sub:
                    break
            if is_sub:
                substantiated.append(bw)
            else:
                unsubstantiated.append(bw)

    total_words = len(text.split())
    density = (len(buzzwords_found) / max(total_words, 1)) * 100

    return {
        "total_buzzwords": len(buzzwords_found),
        "buzzwords_found": buzzwords_found,
        "substantiated": substantiated,
        "unsubstantiated": unsubstantiated,
        "density_percent": round(density, 2),
        "is_stuffed": density > 4.5 or (len(unsubstantiated) > 4 and len(unsubstantiated) > len(substantiated) * 2),
    }


def _extract_resume_skills(resume_text: str) -> list[str]:
    text_lower = resume_text.lower()
    skill_patterns = [
        r'\b(python|java|javascript|typescript|c\+\+|c#|ruby|go|rust|swift|kotlin|php|scala)\b',
        r'\b(react|angular|vue|django|flask|spring|spring\s*boot|express|node\.?js|next\.?js|fastapi|rails)\b',
        r'\b(aws|azure|gcp|docker|kubernetes|terraform|ansible|jenkins|github\s*actions|ci/cd)\b',
        r'\b(tensorflow|pytorch|keras|scikit-learn|pandas|numpy|spark|sql|nosql|mongodb|postgresql|redis|elasticsearch|dynamodb)\b',
        r'\b(rest\s*api|rest|graphql|grpc|microservices|system\s*design|kafka|rabbitmq|websocket)\b',
        r'\b(git|linux|bash|agile|scrum)\b',
        r'\b(machine\s*learning|deep\s*learning|nlp|computer\s*vision|data\s*science)\b',
    ]

    skills = set()
    for pattern in skill_patterns:
        for match in re.finditer(pattern, text_lower):
            raw = match.group().strip()
            skills.add(raw)

    return sorted(skills)


def cross_reference_skills(
    resume_text: str,
    transcript_text: str,
    qa_list: list[dict] | None = None,
    evaluations: list[dict] | None = None,
) -> dict:
    """
    Cross-reference resume skills with interview questions and answers.
    Divides skills into TESTED skills and UNTESTED skills.
    Tested skills evaluate match vs mismatch.
    Untested skills are neutral (NOT penalised).
    """
    resume_skills = _extract_resume_skills(resume_text)
    transcript_lower = transcript_text.lower()

    eval_by_q = {}
    if evaluations:
        eval_by_q = {e.get("question_id"): e for e in evaluations}

    qa_by_skill: dict[str, list[dict]] = {}
    if qa_list:
        for qa in qa_list:
            sk = qa.get("skill", "").lower()
            qa_by_skill.setdefault(sk, []).append(qa)

    tested_results = []
    untested_results = []

    for skill in resume_skills:
        skill_lower = skill.lower()
        synonyms = _TECH_SYNONYMS.get(skill_lower, [])

        # Check if skill was tested in Q&A
        matching_qas = []
        if qa_list:
            for qa in qa_list:
                q_text = (qa.get("question_text", "") + " " + qa.get("skill", "")).lower()
                if skill_lower in q_text or any(syn in q_text for syn in synonyms):
                    matching_qas.append(qa)

        # Also check if skill was prominently discussed in candidate answers
        in_transcript = skill_lower in transcript_lower or any(syn in transcript_lower for syn in synonyms)

        is_tested = len(matching_qas) > 0

        if is_tested:
            # Aggregate answer scores for questions testing this skill
            scores = []
            for qa in matching_qas:
                ev = eval_by_q.get(qa.get("question_id"))
                if ev:
                    scores.append(ev.get("score", 70.0))
                else:
                    scores.append(75.0)

            avg_score = sum(scores) / len(scores) if scores else 70.0
            q_ids = [qa.get("question_id", "") for qa in matching_qas]

            # Semantic similarity between resume claim and candidate answers
            resume_matches = [
                s.strip() for s in re.split(r'(?<=[.!?\n])\s+', resume_text)
                if skill_lower in s.lower()
            ]
            resume_ctx = " ".join(resume_matches[:2]) if resume_matches else skill
            candidate_answers = " ".join(qa.get("candidate_answer", "") for qa in matching_qas)
            sim_score = semantic_similarity(resume_ctx, candidate_answers) if candidate_answers else 0.5

            # Check if any matching evaluation had incorrect concepts (contradictions/fabrications)
            has_factual_error = any(
                len(eval_by_q.get(qid, {}).get("incorrect_concepts", [])) > 0
                for qid in q_ids
            )

            if has_factual_error or avg_score < 30:
                support = "MISMATCH"
                match_status = "MISMATCH"
                reasons = []
                for qid in q_ids:
                    errs = eval_by_q.get(qid, {}).get("incorrect_concepts", [])
                    reasons.extend(errs)
                reason_str = f" ({reasons[0]})" if reasons else ""
                message = f'"{skill}" MISMATCH: Claimed on resume, but candidate gave incorrect/contradictory answers when tested (Score: {round(avg_score)}%){reason_str}.'
            elif avg_score >= 70:
                support = "STRONG"
                match_status = "MATCH"
                message = f'"{skill}" is verified: candidate demonstrated strong technical knowledge (Score: {round(avg_score)}%) matching resume claims.'
            elif avg_score >= 50:
                support = "MODERATE"
                match_status = "PARTIAL"
                message = f'"{skill}" is partially verified: candidate demonstrated basic familiarity (Score: {round(avg_score)}%).'
            else:
                support = "SURFACE"
                match_status = "SURFACE"
                message = f'"{skill}" had weak demonstration: candidate gave limited or superficial answers (Score: {round(avg_score)}%).'

            tested_results.append({
                "skill": skill,
                "in_resume": True,
                "tested_in_interview": True,
                "questions_used": q_ids,
                "demonstrated_score": round(avg_score, 1),
                "semantic_similarity": round(max(sim_score, avg_score / 100.0), 2),
                "support_level": support,
                "match_status": match_status,
                "message": message,
            })
        else:
            # Skill was NOT asked in this interview session
            untested_results.append({
                "skill": skill,
                "in_resume": True,
                "tested_in_interview": False,
                "support_level": "NOT_TESTED",
                "match_status": "NOT_TESTED",
                "semantic_similarity": 0.0,
                "message": f'"{skill}" is listed on the resume but was not tested during this interview session.',
            })

    tested_count = len(tested_results)
    verified_count = sum(1 for r in tested_results if r["support_level"] in ("STRONG", "MODERATE"))
    mismatch_count = sum(1 for r in tested_results if r["support_level"] == "MISMATCH")
    match_rate = round((verified_count / max(tested_count, 1)) * 100.0, 1) if tested_count > 0 else 100.0

    combined_list = tested_results + untested_results

    return {
        "tested_skills": tested_results,
        "untested_skills": untested_results,
        "skill_cross_reference": combined_list,
        "total_skills": len(resume_skills),
        "tested_count": tested_count,
        "verified_count": verified_count,
        "mismatch_count": mismatch_count,
        "untested_count": len(untested_results),
        "supported_count": verified_count,
        "unsupported_count": mismatch_count,
        "match_rate_percent": match_rate,
        "support_ratio": round(verified_count / max(tested_count, 1), 2) if tested_count > 0 else 1.0,
    }


def detect_all(
    resume_text: str,
    transcript_text: str = "",
    qa_list: list[dict] | None = None,
    evaluations: list[dict] | None = None,
) -> dict:
    """
    Run keyword detection and cross-referencing.
    Calculates Resume Suspicion Score based on REAL mismatches, hollow claims,
    and buzzword stuffing without penalizing untested skills.
    """
    hollow = _count_hollow(resume_text)
    density = _count_buzzword_density(resume_text)
    extracted_skills = _extract_resume_skills(resume_text)

    cross_ref = cross_reference_skills(resume_text, transcript_text, qa_list, evaluations)

    # Resume Suspicion Score (0-100, higher = more suspicious)
    suspicion_score = 0.0

    # 1. Skill mismatch penalty: candidate claimed a skill and failed or lied when asked
    mismatches = cross_ref["mismatch_count"]
    if mismatches > 0:
        suspicion_score += min(75.0, mismatches * 35.0)

    # 2. Check for admissions of fabrication in transcript
    t_lower = transcript_text.lower()
    fabrication_detected = any(
        re.search(p, t_lower) for p in [
            r"\bsound(?:ed)?\s+impressive\b",
            r"\bhaven't\s+(?:actually\s+)?used\s+it\s+in\s+any\s+(?:of\s+my\s+)?projects\b",
            r"\bhaven't\s+worked\s+on\s+a\s+project\s+using\s+those\b",
            r"\brandom\s+algorithm\s+generation\b",
        ]
    )
    if fabrication_detected:
        suspicion_score = max(suspicion_score, 85.0)

    # 3. Hollow claims penalty
    suspicion_score += min(20.0, len(hollow) * 3.0)

    # 4. Buzzword stuffing penalty
    if density["is_stuffed"]:
        suspicion_score += 15.0

    suspicion_score = min(100.0, round(suspicion_score, 1))

    return {
        "hollow_claims": hollow,
        "buzzword_analysis": density,
        "extracted_skills": extracted_skills,
        "cross_reference": cross_ref,
        "resume_suspicion_score": suspicion_score,
        "summary": _build_summary(hollow, density, cross_ref, suspicion_score),
    }


def _build_summary(hollow: list, density: dict, cross_ref: dict, score: float) -> str:
    tested = cross_ref.get("tested_count", 0)
    verified = cross_ref.get("verified_count", 0)
    mismatches = cross_ref.get("mismatch_count", 0)
    untested = cross_ref.get("untested_count", 0)

    parts = []
    if mismatches > 0:
        parts.append(f"Caution: {mismatches} skill mismatch(es) detected — candidate claimed skills on resume that they failed to demonstrate when tested.")
    elif tested > 0 and verified == tested:
        parts.append(f"Strong match: All {tested} tested skill(s) were successfully verified in candidate answers ({cross_ref.get('match_rate_percent', 100)}% match rate).")
    elif tested > 0:
        parts.append(f"{verified} of {tested} tested skills verified in the interview.")
    else:
        parts.append("Resume claims processed.")

    if untested > 0:
        parts.append(f"({untested} resume skills were not covered in this interview session).")

    if hollow:
        parts.append(f"Found {len(hollow)} vague/hollow descriptor(s).")
    if density.get("is_stuffed"):
        parts.append("High buzzword density detected in resume.")

    return " ".join(parts)
