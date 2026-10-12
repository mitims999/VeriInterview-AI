"""
Generates the 5 synthetic demo candidates required by Master Spec Section 25 /
Build Spec Section 5.2. Run: python scripts/generate_fixtures.py
"""
import json
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "synthetic")
os.makedirs(OUT_DIR, exist_ok=True)

JD = (
    "We are hiring a Backend Engineer with strong Python skills and hands-on AWS experience. "
    "The ideal candidate has designed and deployed production systems on EC2, S3, and Lambda, "
    "understands IAM and VPC networking, has used auto-scaling and load balancer configurations, "
    "and is comfortable with machine learning concepts such as overfitting, class imbalance, and "
    "model evaluation using XGBoost or similar techniques. Kubernetes experience is a plus."
)

BASE_QUESTIONS = [
    dict(question_id="Q1", question_text="What is overfitting in machine learning and how do you detect it?",
         skill="Machine Learning", difficulty="Basic", expected_proficiency="Intermediate",
         reference_answer="Overfitting occurs when a model learns the training data too closely, including noise, "
                           "and therefore generalizes poorly to unseen data. It can be detected via a gap between "
                           "training and validation/test performance.",
         required_concepts=["training data", "noise", "generalization", "unseen data", "validation"]),
    dict(question_id="Q2", question_text="Describe how you deployed a system on AWS. What services did you use?",
         skill="AWS", difficulty="Intermediate", expected_proficiency="Advanced",
         reference_answer="A strong answer describes EC2 for compute, S3 for storage, Lambda for event-driven "
                           "functions, IAM for access control, and auto-scaling/load balancers for reliability.",
         required_concepts=["EC2", "S3", "Lambda", "IAM", "auto-scaling", "load balancer"]),
    dict(question_id="Q3", question_text="How would you handle an imbalanced dataset when training a classifier?",
         skill="Machine Learning", difficulty="Intermediate", expected_proficiency="Advanced",
         reference_answer="Techniques include resampling (oversampling minority/undersampling majority), class "
                           "weighting, using appropriate metrics like F1 or PR-AUC instead of accuracy, and "
                           "algorithms such as XGBoost with scale_pos_weight.",
         required_concepts=["resampling", "class weight", "F1", "precision", "recall", "XGBoost"]),
    dict(question_id="Q4", question_text="What is the difference between a load balancer and auto-scaling group on AWS?",
         skill="AWS", difficulty="Basic", expected_proficiency="Intermediate",
         reference_answer="A load balancer distributes incoming traffic across multiple instances, while an "
                           "auto-scaling group automatically adds or removes instances based on demand.",
         required_concepts=["load balancer", "auto-scaling", "traffic distribution", "demand"]),
]

COUNTERFACTUAL_TEMPLATE = dict(
    skill="Machine Learning",
    followup_question_id="Q3F",
    perturbed_assumption="the minority class becomes only 1% of the dataset instead of 30%",
    expected_new_concepts=["SMOTE", "oversampling", "class weight", "evaluation metric change", "anomaly detection framing"],
    expected_retained_concepts=["precision", "recall", "F1"],
)


def qa(qid, latency, duration, answer):
    base = next(q for q in BASE_QUESTIONS if q["question_id"] == qid).copy()
    base["candidate_answer"] = answer
    base["response_latency_sec"] = latency
    base["answer_duration_sec"] = duration
    return base


def speech(qid, pitch, loud, rate, pause, ratio):
    return dict(question_id=qid, pitch_hz=pitch, loudness_db=loud, speech_rate_wpm=rate, pause_ratio=pause, speech_silence_ratio=ratio)


# ---------------------------------------------------------------------------
# Candidate A — Genuine Strong
# ---------------------------------------------------------------------------
genuine = dict(
    candidate_name="Ananya Rao",
    scenario_label="Genuine Strong",
    expected_outcome="LOW risk, HIGH confidence, strong skill verification, normal behavior.",
    job_description=JD,
    resume_text="Designed and deployed AWS architecture for a production recommendation system using EC2, S3 and "
                "Lambda. Experience with XGBoost for imbalanced classification. Familiar with Kubernetes.",
    resume_claims=[
        {"skill": "AWS", "claim_text": "Designed and deployed AWS architecture for a production recommendation system.", "claim_strength": "strong"},
        {"skill": "Machine Learning", "claim_text": "Experience with XGBoost for imbalanced classification.", "claim_strength": "medium"},
        {"skill": "Kubernetes", "claim_text": "Familiar with Kubernetes.", "claim_strength": "weak"},
    ],
    qa_list=[
        qa("Q1", 1.8, 40, "Overfitting happens when a model memorizes the training data, including its noise, so it "
                          "performs great on training data but poorly on unseen validation or test data. You usually "
                          "catch it by watching the gap between training and validation loss."),
        qa("Q2", 2.1, 55, "I deployed the recommendation service on EC2 behind a load balancer, used auto-scaling to "
                          "handle traffic spikes, stored artifacts in S3, ran event-driven feature updates with "
                          "Lambda, and scoped access with IAM roles per service."),
        qa("Q3", 2.4, 60, "I'd use resampling — oversampling the minority class or undersampling the majority — and "
                          "switch to class weighting in the loss. I'd also stop trusting accuracy and use F1 or "
                          "PR-AUC instead. XGBoost's scale_pos_weight handles this natively."),
        qa("Q4", 1.9, 38, "A load balancer distributes incoming traffic across healthy instances, while an "
                          "auto-scaling group changes the number of instances based on demand."),
    ],
    speech_features=[
        speech("Q1", 172, -18, 128, 0.14, 0.78),
        speech("Q2", 175, -17, 132, 0.15, 0.80),
        speech("Q3", 170, -18, 125, 0.16, 0.77),
        speech("Q4", 174, -18, 130, 0.14, 0.79),
    ],
    behavior_signals=dict(gaze_off_screen_pct=0.06, head_pose_variance="normal", multi_person_detected=False, phone_detected=False),
    identity_signals=dict(min_frame_similarity_to_anchor=0.98, face_lost_pct=0.01),
    av_signals=dict(av_offset_frames=1, sync_confidence=0.92),
    counterfactuals=[{**COUNTERFACTUAL_TEMPLATE,
        "candidate_followup_answer": "At 1% minority class it's more like anomaly detection than classification — "
                                      "I'd lean on SMOTE or targeted oversampling, push class weighting further, and "
                                      "swap the evaluation metric to precision/recall or PR-AUC since F1 alone gets "
                                      "unstable at that skew."}],
)

# ---------------------------------------------------------------------------
# Candidate B — Genuine Weak (real person, weak skills, no cheating signals)
# ---------------------------------------------------------------------------
genuine_weak = dict(
    candidate_name="Rohit Mehta",
    scenario_label="Genuine Weak",
    expected_outcome="LOW integrity risk, LOW capability score, no false cheating accusation.",
    job_description=JD,
    resume_text="Familiar with AWS. Some exposure to machine learning through coursework.",
    resume_claims=[
        {"skill": "AWS", "claim_text": "Familiar with AWS.", "claim_strength": "weak"},
        {"skill": "Machine Learning", "claim_text": "Some exposure to machine learning through coursework.", "claim_strength": "weak"},
    ],
    qa_list=[
        qa("Q1", 4.2, 30, "Overfitting is like... when the model is too specific to the data you gave it and doesn't "
                          "work well on new data. I think it's related to too many parameters maybe."),
        qa("Q2", 5.1, 25, "I've used EC2 a little bit for a class project. I'm not too familiar with the other "
                          "services honestly."),
        qa("Q3", 6.0, 28, "I would probably just get more data for the smaller class if I could. I'm not sure about "
                          "other techniques."),
        qa("Q4", 3.8, 20, "I think a load balancer spreads out traffic. Not sure exactly how auto-scaling is "
                          "different from that."),
    ],
    speech_features=[
        speech("Q1", 160, -20, 110, 0.22, 0.70),
        speech("Q2", 162, -21, 108, 0.24, 0.69),
        speech("Q3", 159, -20, 105, 0.25, 0.68),
        speech("Q4", 161, -20, 112, 0.21, 0.71),
    ],
    behavior_signals=dict(gaze_off_screen_pct=0.10, head_pose_variance="normal", multi_person_detected=False, phone_detected=False),
    identity_signals=dict(min_frame_similarity_to_anchor=0.97, face_lost_pct=0.02),
    av_signals=dict(av_offset_frames=1, sync_confidence=0.88),
    counterfactuals=[{**COUNTERFACTUAL_TEMPLATE,
        "candidate_followup_answer": "I guess I'd still just try to get more data. I'm not sure what else would "
                                      "change if it was that extreme."}],
)

# ---------------------------------------------------------------------------
# Candidate C — AI-assisted simulation
# ---------------------------------------------------------------------------
ai_assisted = dict(
    candidate_name="Karan Verma",
    scenario_label="AI-assisted Simulation",
    expected_outcome="MEDIUM/HIGH integrity risk: resume/answer mismatch, latency anomalies, behavior anomalies.",
    job_description=JD,
    resume_text="Designed and deployed AWS architecture at scale. Led migration to Kubernetes. Deep experience with "
                "XGBoost and imbalanced classification in production.",
    resume_claims=[
        {"skill": "AWS", "claim_text": "Designed and deployed AWS architecture at scale.", "claim_strength": "strong"},
        {"skill": "Kubernetes", "claim_text": "Led migration to Kubernetes.", "claim_strength": "strong"},
        {"skill": "Machine Learning", "claim_text": "Deep experience with XGBoost and imbalanced classification in production.", "claim_strength": "strong"},
    ],
    qa_list=[
        qa("Q1", 1.6, 42, "Overfitting occurs when a model learns the training data too closely, including noise, "
                          "and therefore generalizes poorly to unseen data."),
        qa("Q2", 2.0, 50, "I used EC2, S3, Lambda, IAM, auto-scaling and load balancers to build a highly available "
                          "recommendation platform serving millions of requests."),
        qa("Q3", 8.9, 65, "To handle an imbalanced dataset you can use resampling techniques such as SMOTE for "
                          "oversampling the minority class, apply class weighting during training, and evaluate "
                          "using F1-score or precision-recall AUC rather than raw accuracy, since accuracy is "
                          "misleading under class imbalance."),
        qa("Q4", 1.7, 30, "A load balancer distributes traffic, auto-scaling changes instance count with demand."),
    ],
    speech_features=[
        speech("Q1", 168, -18, 126, 0.15, 0.78),
        speech("Q2", 170, -18, 128, 0.15, 0.79),
        speech("Q3", 152, -14, 178, 0.05, 0.94),  # sudden change: faster, louder, fewer pauses = read-aloud pattern
        speech("Q4", 169, -18, 127, 0.15, 0.78),
    ],
    behavior_signals=dict(gaze_off_screen_pct=0.34, head_pose_variance="elevated", multi_person_detected=False, phone_detected=False),
    identity_signals=dict(min_frame_similarity_to_anchor=0.95, face_lost_pct=0.05),
    av_signals=dict(av_offset_frames=2, sync_confidence=0.75),
    counterfactuals=[{**COUNTERFACTUAL_TEMPLATE,
        "candidate_followup_answer": "Um, I think you'd still just use more data collection. I'm not sure the "
                                      "approach would change that much honestly."}],
)

# ---------------------------------------------------------------------------
# Candidate D — Synthetic deepfake / identity simulation
# ---------------------------------------------------------------------------
deepfake = dict(
    candidate_name="Deep Fake Test Subject",
    scenario_label="Deepfake / Identity Simulation",
    expected_outcome="HIGH A/V risk, possibly HIGH overall risk.",
    job_description=JD,
    resume_text="Designed and deployed AWS architecture. Strong ML background using XGBoost.",
    resume_claims=[
        {"skill": "AWS", "claim_text": "Designed and deployed AWS architecture.", "claim_strength": "strong"},
        {"skill": "Machine Learning", "claim_text": "Strong ML background using XGBoost.", "claim_strength": "strong"},
    ],
    qa_list=[
        qa("Q1", 2.0, 40, "Overfitting is when a model fits the training set too closely, including its noise, and "
                          "fails to generalize to new data."),
        qa("Q2", 2.3, 48, "I used EC2, S3, Lambda and IAM roles, with load balancers and auto-scaling for "
                          "reliability."),
        qa("Q3", 2.6, 52, "Resampling, class weighting, and using F1 or PR-AUC instead of accuracy are the standard "
                          "approaches, along with tools like XGBoost's scale_pos_weight."),
        qa("Q4", 2.1, 36, "Load balancers distribute traffic; auto-scaling groups adjust instance counts."),
    ],
    speech_features=[
        speech("Q1", 171, -18, 129, 0.15, 0.78),
        speech("Q2", 173, -17, 131, 0.14, 0.79),
        speech("Q3", 170, -18, 127, 0.15, 0.78),
        speech("Q4", 172, -18, 130, 0.15, 0.79),
    ],
    behavior_signals=dict(gaze_off_screen_pct=0.08, head_pose_variance="normal", multi_person_detected=False, phone_detected=False),
    identity_signals=dict(min_frame_similarity_to_anchor=0.71, face_lost_pct=0.18),  # significant drift = manipulated media
    av_signals=dict(av_offset_frames=9, sync_confidence=0.35),                      # large offset, low confidence
    counterfactuals=[{**COUNTERFACTUAL_TEMPLATE,
        "candidate_followup_answer": "At that skew you'd lean on SMOTE, adjust class weights further, and evaluate "
                                      "with precision/recall or PR-AUC rather than F1 alone."}],
)

# ---------------------------------------------------------------------------
# Candidate E — Nervous Genuine Candidate
# ---------------------------------------------------------------------------
nervous_genuine = dict(
    candidate_name="Priya Nair",
    scenario_label="Nervous Genuine Candidate",
    expected_outcome="Speech/gaze anomaly may read HIGH, but overall integrity risk should NOT automatically be HIGH.",
    job_description=JD,
    resume_text="Designed and deployed AWS-based systems including EC2, S3, and Lambda. Applied XGBoost for "
                "imbalanced classification problems.",
    resume_claims=[
        {"skill": "AWS", "claim_text": "Designed and deployed AWS-based systems including EC2, S3, and Lambda.", "claim_strength": "strong"},
        {"skill": "Machine Learning", "claim_text": "Applied XGBoost for imbalanced classification problems.", "claim_strength": "medium"},
    ],
    qa_list=[
        qa("Q1", 3.4, 44, "Sorry, um, overfitting is — okay — it's when the model basically memorizes the training "
                          "data, noise included, so it does badly on data it hasn't seen before."),
        qa("Q2", 3.9, 58, "I, uh, deployed things on EC2 behind a load balancer, used auto-scaling for traffic, kept "
                          "artifacts in S3, and used Lambda for some event triggers, with IAM roles scoping access."),
        qa("Q3", 4.6, 63, "Sorry, let me think — you'd resample the data, maybe oversample the minority class, use "
                          "class weights, and definitely not trust plain accuracy — F1 or precision/recall instead."),
        qa("Q4", 3.1, 34, "A load balancer spreads traffic across instances, auto-scaling changes how many "
                          "instances you have based on demand."),
    ],
    # Nervousness builds over the interview relative to the candidate's OWN early
    # baseline (Q1/Q2) rather than any absolute "normal voice" threshold — pitch
    # rises, speech slows, and pauses lengthen noticeably by Q3/Q4.
    speech_features=[
        speech("Q1", 178, -19, 122, 0.17, 0.75),
        speech("Q2", 181, -19, 119, 0.18, 0.74),
        speech("Q3", 214, -25, 88, 0.38, 0.55),
        speech("Q4", 219, -26, 84, 0.41, 0.52),
    ],
    behavior_signals=dict(gaze_off_screen_pct=0.28, head_pose_variance="elevated", multi_person_detected=False, phone_detected=False),
    identity_signals=dict(min_frame_similarity_to_anchor=0.97, face_lost_pct=0.03),
    av_signals=dict(av_offset_frames=1, sync_confidence=0.9),
    counterfactuals=[{**COUNTERFACTUAL_TEMPLATE,
        "candidate_followup_answer": "Oh — okay, at 1%, I think it becomes more like anomaly detection, so I'd use "
                                      "SMOTE-style oversampling, push the class weights harder, and switch to "
                                      "precision/recall or PR-AUC since F1 would be shaky at that skew."}],
)

FIXTURES = {
    "genuine": genuine,
    "genuine_weak": genuine_weak,
    "ai_assisted": ai_assisted,
    "deepfake": deepfake,
    "nervous_genuine": nervous_genuine,
}

if __name__ == "__main__":
    for key, data in FIXTURES.items():
        path = os.path.join(OUT_DIR, f"{key}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"wrote {path}")
