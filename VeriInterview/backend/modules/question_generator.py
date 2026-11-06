"""
Feature 5 (Member 5) — AI Question Generation (Master Spec Section 36).

Pipeline: resume claims -> skill + claim strength -> question generation ->
question/skill mapping -> reference answer -> key concepts -> difficulty ->
interviewer approval. The interviewer must always be able to edit or discard
AI-generated content before the interview starts (Master Spec Section 4/35) —
this module only ever returns `ai_generated=True` drafts; `PUT /api/questions/{id}`
(existing endpoint) is how the interviewer edits them.

Model-orchestration rule (Master Spec Section 31): the LLM, when configured,
only ever generates TEXT (question wording, reference answer prose). It never
produces a score or a risk weight, and this module runs fully offline with a
deterministic template bank when LLM_PROVIDER == "none" (the default) — no
network access and no LLM_API_KEY are required to demo this feature.
"""
from __future__ import annotations
from backend.config import LLM_PROVIDER
from backend.schemas import ResumeClaim

# Small curated question bank, keyed by normalized skill name. Each entry maps
# a claim-strength tier to (question_text, reference_answer, required_concepts).
# This is the deterministic fallback used whenever LLM_PROVIDER == "none"
# (default) or the configured provider call fails for any reason.
_QUESTION_BANK: dict[str, dict[str, dict]] = {
    "aws": {
        "weak": dict(
            question_text="What AWS services are you familiar with, and what does each one do at a high level?",
            reference_answer="A candidate at this level should name a few core services (e.g. EC2 for compute, "
                              "S3 for storage) and describe their basic purpose, even without hands-on depth.",
            required_concepts=["EC2", "S3", "compute", "storage"],
            difficulty="Basic",
        ),
        "medium": dict(
            question_text="Walk through how you have used AWS in a project. What services did you combine and why?",
            reference_answer="A solid answer names specific services used together (e.g. EC2 + S3 + IAM) and "
                              "explains the reasoning behind the combination, not just a service list.",
            required_concepts=["EC2", "S3", "IAM", "integration"],
            difficulty="Intermediate",
        ),
        "strong": dict(
            question_text="Describe how you designed and deployed an AWS architecture for production. "
                           "What services did you use and what reliability/scaling decisions did you make?",
            reference_answer="A strong answer covers compute (EC2/Lambda), storage (S3), access control (IAM), "
                              "and reliability mechanisms such as auto-scaling and load balancing, with concrete "
                              "reasoning for each choice.",
            required_concepts=["EC2", "S3", "IAM", "auto-scaling", "load balancer"],
            difficulty="Advanced",
        ),
    },
    "machine learning": {
        "weak": dict(
            question_text="What is overfitting in machine learning, and why does it matter?",
            reference_answer="Overfitting occurs when a model learns the training data too closely, including "
                              "noise, and therefore generalizes poorly to unseen data.",
            required_concepts=["training data", "generalization", "unseen data"],
            difficulty="Basic",
        ),
        "medium": dict(
            question_text="How would you evaluate a classification model, and what would make you distrust accuracy alone?",
            reference_answer="A solid answer mentions precision/recall/F1 and explains that accuracy is misleading "
                              "under class imbalance.",
            required_concepts=["precision", "recall", "F1", "class imbalance"],
            difficulty="Intermediate",
        ),
        "strong": dict(
            question_text="How would you handle a severely imbalanced dataset when training a classifier?",
            reference_answer="Techniques include resampling (oversampling minority/undersampling majority), class "
                              "weighting, and using metrics like F1 or PR-AUC instead of accuracy.",
            required_concepts=["resampling", "class weight", "F1", "precision", "recall"],
            difficulty="Advanced",
        ),
    },
    "docker": {
        "weak": dict(
            question_text="What is a Docker container, and how is it different from a virtual machine?",
            reference_answer="A container packages an application with its dependencies and shares the host OS "
                              "kernel, unlike a VM which virtualizes the whole OS — making containers lighter weight.",
            required_concepts=["container", "kernel", "virtual machine"],
            difficulty="Basic",
        ),
        "medium": dict(
            question_text="Walk through how you containerized an application. What went into the Dockerfile?",
            reference_answer="A solid answer describes base image choice, dependency installation, and exposing "
                              "the right port/entrypoint for the application.",
            required_concepts=["Dockerfile", "base image", "entrypoint"],
            difficulty="Intermediate",
        ),
        "strong": dict(
            question_text="How did you design a multi-container deployment, and how did the containers communicate?",
            reference_answer="A strong answer covers networking between containers (e.g. docker-compose or a "
                              "orchestrator), volume/data persistence, and health-check/restart strategy.",
            required_concepts=["networking", "volumes", "health check", "orchestration"],
            difficulty="Advanced",
        ),
    },
    "kubernetes": {
        "weak": dict(
            question_text="What problem does Kubernetes solve that plain Docker doesn't?",
            reference_answer="Kubernetes orchestrates many containers across machines: scheduling, scaling, "
                              "restarting failed containers, and service discovery — Docker alone runs single containers.",
            required_concepts=["orchestration", "scheduling", "scaling"],
            difficulty="Basic",
        ),
        "medium": dict(
            question_text="Describe a Kubernetes deployment you've worked with. What did the pods and services look like?",
            reference_answer="A solid answer names pods, deployments, and services, and explains how traffic reached the pods.",
            required_concepts=["pod", "deployment", "service"],
            difficulty="Intermediate",
        ),
        "strong": dict(
            question_text="How did you lead a migration to Kubernetes? What rollout strategy and failure-handling did you use?",
            reference_answer="A strong answer covers rollout strategy (e.g. rolling update/canary), resource limits, "
                              "and how failures/restarts were handled automatically.",
            required_concepts=["rolling update", "resource limits", "self-healing"],
            difficulty="Advanced",
        ),
    },
}

_GENERIC_TEMPLATE = dict(
    weak=dict(
        question_text="Can you describe what you know about {skill} and where you've encountered it?",
        reference_answer="A candidate at this level should be able to describe the basic purpose of {skill} "
                          "even without deep hands-on project experience.",
        required_concepts=[],
        difficulty="Basic",
    ),
    medium=dict(
        question_text="Describe a time you used {skill} in a project. What was your specific contribution?",
        reference_answer="A solid answer explains a concrete use of {skill}, not just that it appeared on a team project.",
        required_concepts=[],
        difficulty="Intermediate",
    ),
    strong=dict(
        question_text="You mentioned advanced experience with {skill}. Walk through the most complex problem you solved with it.",
        reference_answer="A strong answer demonstrates depth: specific design decisions, trade-offs considered, "
                          "and the reasoning behind them — not a surface-level summary.",
        required_concepts=[],
        difficulty="Advanced",
    ),
)

# Counterfactual perturbation templates for the knowledge-ownership / CCST module
# (Master Spec Section 13). Kept skill-agnostic so any generated question can
# get a follow-up stress test.
_COUNTERFACTUAL_TEMPLATE = dict(
    perturbed_assumption="one key constraint in your original answer is reversed (e.g. the scale, the data "
                          "distribution, or the resource budget changes significantly)",
    followup_question_template="You said {summary}. What would change in your approach if that assumption flipped?",
)


def _normalize_skill(skill: str) -> str:
    return skill.strip().lower()


def _bank_entry(skill: str, strength: str) -> dict:
    bank = _QUESTION_BANK.get(_normalize_skill(skill))
    if bank and strength in bank:
        return bank[strength]
    template = _GENERIC_TEMPLATE[strength]
    return {
        **template,
        "question_text": template["question_text"].format(skill=skill),
        "reference_answer": template["reference_answer"].format(skill=skill),
    }


def _llm_generate_question(skill: str, strength: str, job_description: str) -> dict | None:
    """
    Integration point for a real LLM provider (Master Spec Section 31: the LLM
    only ever generates question/answer TEXT, never a score). Returns None
    when LLM_PROVIDER == "none" or the call fails, so callers always fall back
    to the deterministic template bank above.
    """
    if LLM_PROVIDER == "none":
        return None
    try:
        # Placeholder for a real provider call — intentionally not implemented
        # here, since no LLM_API_KEY is configured in this environment. A real
        # integration would call the provider with (skill, strength, job_description)
        # and parse its response into the same dict shape as _bank_entry().
        return None
    except Exception:
        return None


def generate_questions(resume_claims: list[ResumeClaim] | list[dict], job_description: str = "",
                        max_questions: int = 6) -> list[dict]:
    """
    Generates draft questions from resume claims, one per claimed skill
    (deduplicated), ordered strongest-claim-first so the interviewer sees the
    highest-stakes verification questions first. Every result has
    `ai_generated=True` and must be reviewed/edited before the interview starts.
    """
    claims = [c if isinstance(c, ResumeClaim) else ResumeClaim(**c) for c in resume_claims]
    strength_rank = {"strong": 0, "medium": 1, "weak": 2}
    seen_skills: set[str] = set()
    questions: list[dict] = []

    for claim in sorted(claims, key=lambda c: strength_rank.get(c.claim_strength, 1)):
        skill_key = _normalize_skill(claim.skill)
        if skill_key in seen_skills:
            continue
        seen_skills.add(skill_key)

        draft = _llm_generate_question(claim.skill, claim.claim_strength, job_description)
        if draft is None:
            draft = _bank_entry(claim.skill, claim.claim_strength)

        questions.append({
            "question_text": draft["question_text"],
            "skill": claim.skill,
            "difficulty": draft["difficulty"],
            "expected_proficiency": {"weak": "Basic", "medium": "Intermediate", "strong": "Advanced"}[claim.claim_strength],
            "reference_answer": draft["reference_answer"],
            "required_concepts": draft["required_concepts"],
            "ai_generated": True,
            "source_claim": claim.claim_text,
        })
        if len(questions) >= max_questions:
            break

    return questions


def generate_counterfactual(question: dict) -> dict:
    """
    Generates a CCST follow-up probe (Master Spec Section 13) for one
    question. The perturbation is deliberately generic/template-based here —
    a real deployment would use the LLM to tailor the perturbed assumption to
    the specific claim, but the scoring in knowledge_ownership.py never
    depends on how the question text was produced.
    """
    required = question.get("required_concepts") or []
    half = max(1, len(required) // 2)
    return {
        "skill": question["skill"],
        "followup_question_id": f"{question.get('question_id', 'Q')}F",
        "perturbed_assumption": _COUNTERFACTUAL_TEMPLATE["perturbed_assumption"],
        "followup_question_text": _COUNTERFACTUAL_TEMPLATE["followup_question_template"].format(
            summary=question["question_text"][:80].rstrip(".?!") + "..."
        ),
        "expected_new_concepts": required[half:] or required,
        "expected_retained_concepts": required[:half] or required,
    }


def export_to_ideal_answers_format(questions: list[dict]) -> str:
    """
    Renders generated questions in the `[Qn]` block format consumed by
    `backend/modules/intake_parser.py::parse_ideal_answers`, so AI-generated
    questions can feed directly into the Upload & Analyze flow once the
    interviewer has approved/edited them.
    """
    blocks = []
    for i, q in enumerate(questions, start=1):
        concepts = ", ".join(q.get("required_concepts", []))
        blocks.append(
            f"[Q{i}]\n"
            f"skill: {q['skill']}\n"
            f"difficulty: {q['difficulty']}\n"
            f"expected_proficiency: {q['expected_proficiency']}\n"
            f"question: {q['question_text']}\n"
            f"reference_answer: {q['reference_answer']}\n"
            f"concepts: {concepts}\n"
        )
    return "\n".join(blocks)
