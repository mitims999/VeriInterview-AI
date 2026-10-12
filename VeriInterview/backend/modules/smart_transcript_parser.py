"""
Smart Transcript Parser — extracts Q&A pairs from free-form interview transcripts.

Handles:
  - Multi-line questions (questions wrapping across line breaks)
  - Multi-paragraph candidate answers
  - Speaker labels with em-dashes / en-dashes / names (e.g. "Interviewee — Muskan:")
  - Standalone speaker labels followed by blank lines
  - Q:/A: style, Interviewer:/Candidate: style, Named speakers (e.g. "John: ... / Muskan: ...")
  - Hyphenated technical terms like "K-Means" are never mistaken for speaker labels
  - Auto-detects tested technical skills and generates reference contexts
"""
from __future__ import annotations
import re
from backend.adapters.sentence_embed import semantic_similarity, concept_coverage

# Regex identifying speaker headers.
# Must be at the start of a line (optional timestamp), have a speaker designation, and end with a COLON ':'.
# Requires colon so technical terms with hyphens (e.g. "K-Means", "T-Distributed") are never matched as speakers!
_SPEAKER_HEADER_RE = re.compile(
    r'^(?:\[[\d:\s\.]+\]\s*)?'  # optional timestamp e.g. [00:01:23]
    r'('
    r'(?:interviewer|interviewee|candidate|applicant|recruiter|panel(?:ist)?|hiring\s*manager)'
    r'(?:\s*[\—\-\–]\s*[A-Za-z0-9_\s]+)?'  # optional name e.g. " — Muskan"
    r'|q\s*\d*'
    r'|a\s*\d*'
    r'|question\s*\d*'
    r'|answer\s*\d*'
    r'|[A-Z][a-zA-Z0-9_\s]{1,25}'  # Named speaker e.g. "Muskan"
    r')\s*:\s*(.*)$',
    re.IGNORECASE
)

# Comprehensive technical taxonomy
_SKILL_KEYWORDS: dict[str, list[str]] = {
    "Python": ["python", "decorator", "generator", "list comprehension", "gil", "functools", "lru_cache", "pip", "virtualenv", "cpython", "dunder", "asyncio", "pydantic"],
    "FastAPI": ["fastapi", "uvicorn", "pydantic", "starlette", "dependency injection", "openapi", "swagger"],
    "Django": ["django", "orm", "queryset", "django rest framework", "drf", "middleware"],
    "Flask": ["flask", "jinja", "werkzeug", "blueprint"],
    "Java": ["java", "jvm", "spring", "spring boot", "maven", "gradle", "hibernate", "servlet", "garbage collector"],
    "JavaScript": ["javascript", "js", "es6", "node", "npm", "async", "promise", "callback", "event loop", "closure"],
    "TypeScript": ["typescript", "ts", "interface", "type alias", "generic", "tsc"],
    "React": ["react", "jsx", "virtual dom", "hooks", "usestate", "useeffect", "usememo", "redux", "state management", "component"],
    "SQL / Databases": ["sql", "query", "join", "select", "index", "composite index", "table scan", "normalization", "transaction", "acid", "execution plan", "stored procedure"],
    "PostgreSQL": ["postgresql", "postgres", "psql", "explain analyze", "vacuum", "wal"],
    "MongoDB / NoSQL": ["mongodb", "nosql", "document", "aggregation", "mongoose", "dynamodb", "cassandra"],
    "Redis / Caching": ["redis", "cache", "caching", "in-memory", "ttl", "memcached", "eviction"],
    "AWS": ["aws", "ec2", "s3", "lambda", "iam", "cloudfront", "sqs", "sns", "dynamodb", "rds", "vpc", "step functions", "cloudwatch", "serverless", "eks"],
    "Docker": ["docker", "container", "dockerfile", "docker-compose", "container image", "docker image", "daemon", "isolation", "cgroup", "namespace"],
    "Kubernetes": ["kubernetes", "k8s", "pod", "deployment", "service", "ingress", "helm", "cluster", "node", "kubelet"],
    "RAG / Generative AI": ["rag", "retrieval augmented", "retrieval-augmented", "vector database", "chunking", "reranking", "llm", "groq", "gemini"],
    "Machine Learning": ["machine learning", "ml", "k-means", "kmeans", "dbscan", "cluster", "clusters", "clustering", "unsupervised", "crescent", "centroid", "silhouette", "model", "training", "feature", "classification", "regression", "bias-variance", "scikit-learn", "mlflow"],
    "Deep Learning": ["deep learning", "neural network", "mobilenet", "mobilenetv2", "transfer learning", "cnn", "ann", "computer vision", "image classifier", "rnn", "lstm", "transformer", "attention", "backpropagation", "gradient", "pytorch", "tensorflow", "weights"],
    "Data Science": ["data science", "pandas", "numpy", "matplotlib", "seaborn", "statistics", "hypothesis", "eda", "data cleaning"],
    "System Design": ["system design", "scalability", "high availability", "load balancer", "horizontal scaling", "sharding", "replication", "cdn", "microservices", "latency", "throughput", "cap theorem", "rate limit"],
    "Microservices": ["microservices", "monolith", "service mesh", "circuit breaker", "api gateway", "grpc", "event-driven", "distributed system"],
    "REST API": ["rest", "api", "restful", "endpoint", "http", "get", "post", "put", "delete", "status code", "payload", "json"],
    "Authentication & Security": ["jwt", "oauth", "authentication", "authorization", "token", "rbac", "bearer", "session", "ssl", "tls", "encryption", "hash", "salt"],
    "DevOps & CI/CD": ["devops", "ci/cd", "continuous integration", "jenkins", "github actions", "gitlab ci", "pipeline", "automation", "terraform", "ansible"],
    "Git": ["git", "version control", "branch", "merge", "commit", "pull request", "rebase", "conflict"],
    "Data Structures & Algorithms": ["data structure", "algorithm", "binary search", "hash map", "linked list", "tree", "graph", "dynamic programming", "time complexity", "space complexity", "big o"],
    "Networking": ["networking", "tcp", "udp", "ip", "dns", "http", "https", "socket", "handshake", "packet", "subnet", "firewall"],
    "Linux": ["linux", "bash", "shell", "cron", "process", "kernel", "permissions", "chmod", "systemd"],
    "Testing": ["testing", "unit test", "integration test", "pytest", "jest", "mock", "tdd", "assertion", "test coverage"],
}


def _detect_skill(question_text: str, answer_text: str = "") -> str:
    """Auto-detect which skill is being tested using question + answer text."""
    combined = (question_text + " " + answer_text).lower()
    best_skill = "General"
    best_score = 0

    for skill, keywords in _SKILL_KEYWORDS.items():
        q_hits = sum(1 for kw in keywords if kw in question_text.lower())
        a_hits = sum(1 for kw in keywords if kw in answer_text.lower())
        score = q_hits * 4 + a_hits
        if score > best_score:
            best_score = score
            best_skill = skill

    if best_score == 0:
        best_sim = 0.0
        for skill in _SKILL_KEYWORDS:
            sim = semantic_similarity(question_text, skill)
            if sim > best_sim:
                best_sim = sim
                best_skill = skill
        if best_sim < 0.25:
            best_skill = "Software Engineering"

    return best_skill


def _estimate_difficulty(question_text: str) -> str:
    q_lower = question_text.lower()
    hard_signals = [
        "design", "architect", "trade-off", "tradeoff", "at scale", "optimize",
        "internals", "distributed", "concurrency", "bottleneck", "failure mode",
        "how would you handle", "deep dive"
    ]
    basic_signals = ["what is", "define", "name", "list", "difference between", "what are"]

    if any(s in q_lower for s in hard_signals):
        return "Advanced"
    if any(s in q_lower for s in basic_signals):
        return "Basic"
    return "Intermediate"


def _generate_reference_context(skill: str, question: str, resume_text: str) -> tuple[str, list[str]]:
    keywords = _SKILL_KEYWORDS.get(skill, [])
    concepts = [kw for kw in keywords if kw in question.lower()]
    for kw in keywords[:6]:
        if kw not in concepts:
            concepts.append(kw)

    resume_sentences = [
        s.strip() for s in re.split(r'(?<=[.!?\n])\s+', resume_text)
        if skill.lower() in s.lower()
    ]
    resume_context = " ".join(resume_sentences[:2]) if resume_sentences else ""

    ref_parts = [
        f"Demonstrate practical competence in {skill}.",
        f"Specifically regarding: {question}.",
    ]
    if resume_context:
        ref_parts.append(f"Resume claim context: {resume_context}")
    if keywords:
        ref_parts.append(f"Key concepts: {', '.join(keywords[:5])}")

    return " ".join(ref_parts), concepts[:8]


def _extract_turns(transcript_text: str) -> list[tuple[str, str, str]]:
    """
    Parse transcript into discrete speaker turns: (role, speaker_name, text_body).
    Handles multi-line questions and multi-paragraph answers cleanly.
    """
    lines = transcript_text.splitlines()
    turns: list[tuple[str, str, str]] = []
    current_role: str | None = None
    current_speaker: str | None = None
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        m = _SPEAKER_HEADER_RE.match(stripped)
        if m:
            # Flush previous speaker turn
            if current_role and current_lines:
                turns.append((current_role, current_speaker or "", " ".join(current_lines).strip()))

            raw_speaker = m.group(1).strip()
            content = m.group(2).strip()
            sl = raw_speaker.lower()

            if any(w in sl for w in ["interviewer", "recruiter", "question", "q"]):
                current_role = "interviewer"
            elif any(w in sl for w in ["interviewee", "candidate", "applicant", "answer", "a"]):
                current_role = "candidate"
            else:
                current_role = "speaker"

            current_speaker = raw_speaker
            current_lines = [content] if content else []
        else:
            # Continuation line of current turn
            if current_role is not None:
                current_lines.append(stripped)

    if current_role and current_lines:
        turns.append((current_role, current_speaker or "", " ".join(current_lines).strip()))

    return turns


def parse_free_transcript(
    transcript_text: str,
    resume_text: str = "",
) -> list[dict]:
    """
    Parse a free-form transcript into structured Q&A pairs.
    Handles multi-line questions, multi-line answers, and natural conversational transcripts.
    """
    turns = _extract_turns(transcript_text)

    # If no speaker turns were detected by header regex, try unlabelled paragraph fallback
    if not turns:
        chunks = [c.strip() for c in re.split(r'\n{2,}', transcript_text.strip()) if c.strip()]
        qa_pairs: list[tuple[str, str]] = []
        for i, chunk in enumerate(chunks):
            if '?' in chunk:
                parts = chunk.split('?', 1)
                qa_pairs.append((parts[0].strip() + '?', parts[1].strip() if len(parts) > 1 else ""))
            elif i % 2 == 0 and i + 1 < len(chunks):
                qa_pairs.append((chunk, chunks[i + 1]))
    else:
        # Resolve generic "speaker" roles if names were used
        # Check which speaker asks questions (contains '?')
        speaker_question_counts: dict[str, int] = {}
        for role, speaker, body in turns:
            if role == "speaker":
                if '?' in body:
                    speaker_question_counts[speaker] = speaker_question_counts.get(speaker, 0) + 1
                else:
                    speaker_question_counts.setdefault(speaker, 0)

        interviewer_speaker = None
        if speaker_question_counts:
            interviewer_speaker = max(speaker_question_counts, key=speaker_question_counts.get)

        # Normalize turns into interviewer and candidate
        normalized_turns: list[tuple[str, str]] = []
        for role, speaker, body in turns:
            if role == "interviewer" or (interviewer_speaker and speaker == interviewer_speaker):
                normalized_turns.append(("interviewer", body))
            elif role == "candidate" or (interviewer_speaker and speaker != interviewer_speaker):
                normalized_turns.append(("candidate", body))
            elif '?' in body:
                normalized_turns.append(("interviewer", body))
            else:
                normalized_turns.append(("candidate", body))

        # Pair questions with candidate answers
        qa_pairs = []
        current_q: str | None = None
        current_a_parts: list[str] = []

        for role, body in normalized_turns:
            if role == "interviewer":
                if current_q is not None and current_a_parts:
                    qa_pairs.append((current_q, " ".join(current_a_parts).strip()))
                    current_q = body
                    current_a_parts = []
                elif current_q is not None and not current_a_parts:
                    # Consecutive interviewer statements without candidate answer -> combine into one question!
                    current_q = f"{current_q} {body}".strip()
                else:
                    current_q = body
                    current_a_parts = []
            elif role == "candidate":
                if body:
                    current_a_parts.append(body)

        if current_q is not None:
            qa_pairs.append((current_q, " ".join(current_a_parts).strip()))

    # Filter out empty questions or social closing pleasantries
    _CLOSING_PHRASES = [
        "any questions from your side", "any questions for me", "any questions for us",
        "that concludes the interview", "that's all for today", "thats all for today",
        "wrap up here", "thank you for your time",
    ]
    valid_pairs = []
    for q, a in qa_pairs:
        q_clean = q.lower()
        if len(q.split()) < 3 and not a.strip():
            continue
        # Skip purely social closing questions
        if any(cp in q_clean for cp in _CLOSING_PHRASES) and len(q.split()) < 15:
            continue
        valid_pairs.append((q, a))

    qa_list = []
    prev_skill = None
    for i, (question, answer) in enumerate(valid_pairs, 1):
        skill = _detect_skill(question, answer)
        # Follow-up skill inheritance
        if prev_skill and any(p in question.lower() for p in [
            "those technologies", "that project", "those tools", "where you used them",
            "why did you choose it", "why did you pick it", "what would you do"
        ]):
            if skill in ("General", "Software Engineering"):
                skill = prev_skill
        prev_skill = skill

        difficulty = _estimate_difficulty(question)
        ref_answer, concepts = _generate_reference_context(skill, question, resume_text)

        word_count = len(answer.split())
        estimated_duration = max(5.0, word_count / 2.5)

        qa_list.append({
            "question_id": f"Q{i}",
            "question_text": question,
            "skill": skill,
            "difficulty": difficulty,
            "expected_proficiency": "Intermediate",
            "reference_answer": ref_answer,
            "required_concepts": concepts,
            "candidate_answer": answer,
            "response_latency_sec": 2.0,
            "answer_duration_sec": round(estimated_duration, 1),
        })

    return qa_list


def extract_skills_from_transcript(transcript_text: str) -> list[str]:
    """Extract all technical skills mentioned or discussed in transcript."""
    t_lower = transcript_text.lower()
    found = []
    for skill, keywords in _SKILL_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in t_lower)
        if hits >= 1:
            found.append(skill)
    return sorted(found)
