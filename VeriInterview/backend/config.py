"""
Central configuration for VeriInterview.

All risk weights and thresholds live here (never hard-coded inline in modules),
per the master spec's "keep all risk weights and thresholds configurable" rule.
"""
import os
from pydantic import BaseModel

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "synthetic")
STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage")
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage", "veriinterview.db")

LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "none")  # "none" -> deterministic fallback, no external call


class FusionWeights(BaseModel):
    """Evidence fusion weights (Build Spec section 4). Must sum to 1.0."""
    identity_risk: float = 0.20          # InsightFace adapter
    sync_risk: float = 0.15              # SyncNet adapter
    behavior_risk: float = 0.15          # Proctoring-AI adapter (gaze/head-pose/person-phone)
    speech_behavior_risk: float = 0.15   # openSMILE adapter
    knowledge_ownership_risk: float = 0.20  # Claim graph / CCST (original module)
    profile_consistency_risk: float = 0.15  # Resume-Matcher + expected-answer match


class FusionCaps(BaseModel):
    """Cap on any single module's contribution to the final score (0-1 scale of its own weight)."""
    max_single_module_contribution: float = 0.40  # no module may alone push risk above 40% of total


class RiskThresholds(BaseModel):
    low_max: int = 30       # 0-30 LOW
    medium_max: int = 60    # 31-60 MEDIUM
    # 61-100 HIGH


class SpeechBehaviorWeights(BaseModel):
    """Voice_Prosody spec Section 3.4 heuristic v1 formula."""
    pitch_deviation: float = 0.25
    loudness_deviation: float = 0.20
    speech_rate_deviation: float = 0.25
    pause_deviation: float = 0.20
    speech_ratio_deviation: float = 0.10


# ---------------------------------------------------------------------------
# M3 Deepfake Vision (lip-sync, face identity, gaze/proctoring, deepfake fusion).
# Every value is a prototype default. The comment names the task that calibrates it;
# calibration results live in docs/m3/calibration/ and docs/m3/M3_PROGRESS.md.
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(STORAGE_DIR, "models")      # weights (scripts/download_models.py), gitignored
FEATURES_DIR = os.path.join(STORAGE_DIR, "features")  # derived-feature cache by video sha256 (D6), gitignored


class VisionBudget(BaseModel):
    """How much of a video each analyzer looks at (task N-4 tunes for speed)."""
    max_seconds_analyzed: float = 1800.0   # cap per interview video (30 min)
    max_frame_width: int = 960             # downscale wider frames before analysis
    face_det_size: int = 320               # SCRFD input; 320 finds the same faces as 640 here at 4x speed (G-5)
    identity_fps: float = 1.0              # InsightFace embeddings + pose (N-4: 2 -> 1; switches need >= 3 s presence anyway)
    landmarks_fps: float = 10.0            # MediaPipe gaze/head-pose/blink (blinks last ~100-400 ms)
    objects_fps: float = 1.0               # YOLO person/phone (OEP phone-use episodes: median 4 s, so keep 1 fps)
    yolo_input: int = 416                  # YOLOv3 input size (multiple of 32)
    syncnet_fps: float = 25.0              # fixed by the SyncNet model
    syncnet_min_segment_s: float = 2.0     # shortest speech segment worth scoring
    syncnet_max_window_s: float = 10.0     # longer speech is split into windows of at most this length
    syncnet_max_seconds_total: float = 30.0  # N-4: 60 -> 30 s. SyncNet costs ~1.5-3x real time on CPU; spread over the interview
    syncnet_det_every: int = 5             # detect faces on every Nth 25-fps frame, interpolate between (upstream interpolates too)
    syncnet_crop_scale: float = 0.40       # syncnet_python run_pipeline.py --crop_scale


class VadSettings(BaseModel):
    """Energy VAD fallback (task G-3, calibrated on GRID: docs/m3/calibration/vad_grid.md)."""
    rel_threshold: float = 0.55
    min_snr_db: float = 6.0
    merge_gap_s: float = 0.30
    min_speech_s: float = 0.20


class FaceQuality(BaseModel):
    """Frames failing these are excluded (not scored) so pose/blur never turns into risk (R1 rule 4)."""
    min_det_score: float = 0.60
    min_face_px: int = 64                  # shorter side of the face box
    max_abs_yaw_deg: float = 45.0
    # Blur: variance of Laplacian on the face crop resized to 112 px. Absolute sharpness differs ~6x
    # between cameras/codecs (DeepfakeTIMIT ~50, OEP webcam ~300), and uniform blur is itself a
    # deepfake artifact, so only frames much blurrier than the candidate's own median (motion blur)
    # are excluded, plus a very low absolute floor for unusable crops. (G-5 measurement)
    min_sharpness_abs: float = 15.0
    min_sharpness_rel: float = 0.40


class IdentityThresholds(BaseModel):
    """InsightFace / ArcFace identity consistency (tasks I-1, I-2 calibrate on LFW + splices)."""
    same_person_cosine: float = 0.30       # buffalo_l cosine; LFW: impostor max 0.233, FAR 0%, FRR 0.4% (identity_lfw.md)
    enrollment_seconds: float = 10.0       # anchor built from the first good frames in this window
    enrollment_min_frames: int = 8
    cluster_cosine: float = 0.40           # agglomerative clustering link threshold
    min_identity_presence_s: float = 3.0   # a cluster must last this long to count as a person
    below_threshold_medium_pct: float = 0.05
    below_threshold_high_pct: float = 0.15
    face_lost_medium_pct: float = 0.10     # lowers confidence, not risk
    # Reference photo (task I-4): photo-vs-video is the LFW setting (different photos, years, lighting),
    # so this threshold comes straight from the LFW calibration (docs/m3/calibration/identity_lfw.md).
    reference_match_cosine: float = 0.30   # session face vs uploaded reference photo (LFW-calibrated, same setting)
    reference_uncertain_margin: float = 0.10  # within +margin above the threshold = MEDIUM "weak match"


class SyncThresholds(BaseModel):
    """SyncNet lip-sync (task S-3 calibrates on GRID shifts/swaps + AVLips)."""
    vshift: int = 15                       # +/- frames searched (syncnet_python default)
    bonafide_offset_frames: int = 2        # |offset| <= this is normal
    high_offset_frames: int = 6            # |offset| >= this is clearly desynchronised
    lse_c_good: float = 6.0                # GRID genuine mean 5.88 (S-3); LRS2 ~7-8. Maps to confidence 1.0
    lse_c_low: float = 2.0                 # S-3: audio swaps 1.01-1.57; 240p genuine 1.17-5.39 (but stable offset)
    min_offset_stability: float = 0.75     # S-3/N-4: genuine + 240p = 1.00 in all 96 runs; swaps 0.25-0.75. With the 30 s budget only ~3 windows, so 0.67 must count as unstable
    min_speech_seconds: float = 4.0        # less analysed speech gives INSUFFICIENT_EVIDENCE


class GazeThresholds(BaseModel):
    """Proctoring-AI port (tasks P-2..P-6). Eye-ratio cut-offs are the original repo's values."""
    eye_x_ratio_right: float = 3.0
    eye_x_ratio_left: float = 0.33
    eye_y_ratio_up: float = 0.33
    head_yaw_off_deg: float = 30.0         # true Euler yaw beyond this = facing away from screen
    head_pitch_off_deg: float = 25.0
    min_off_screen_s: float = 0.5          # debounce: shorter glances are ignored
    head_std_elevated_deg: float = 8.0     # yaw/pitch std -> head_pose_variance "elevated"
    head_std_high_deg: float = 15.0        # -> "high"
    yolo_score: float = 0.4                # P-5 COCO eval (full YOLOv3): multi-person P 0.97 / R 0.84, phone P 0.98 / R 0.50, 0 false phones on 200 negatives (upstream 0.6: phone R 0.40)
    yolo_nms_iou: float = 0.5
    object_persist_samples: int = 3        # OEP webcam: 1.7% single-frame false phone rate -> 3 in a row ~0.01 false alerts per 30 min at 1 fps (2 would be ~0.5)


class EyeAuthenticityThresholds(BaseModel):
    """Eye-region cues that feed the deepfake score (R1: one-sided, nervousness-proof). Task P-4 / F-4."""
    blink_closed_score: float = 0.5        # MediaPipe eyeBlink blendshape above this = eye closed
    # P-4 finding: a genuine GRID speaker (s4) did not blink at all in 36 s while reading (blendshape max 0.07,
    # EAR never dipped). Blink inhibition while reading/concentrating is normal, so 'too few blinks' is weak,
    # corroborative evidence only (deepfake_fusion caps eye cues at MEDIUM), judged over >= 2 min of visible eyes.
    min_eye_visible_s: float = 120.0       # need this much visible-eye time to judge blinking at all
    too_few_blinks_per_min: float = 2.0    # ONLY low/absent blinking is suspicious; high rate adds 0 risk
    metronomic_ibi_cv: float = 0.15        # inter-blink-interval coefficient of variation below this = unnaturally regular
    mouth_mismatch_window_s: float = 1.0   # speech with a closed mouth (or vice versa) must last this long to count
    mouth_mismatch_medium_pct: float = 0.15


class DeepfakeWeights(BaseModel):
    """Deepfake sub-score fusion (task F-1). Must sum to 1.0. Gaze-off-screen is NOT an input (D3)."""
    identity: float = 0.45
    sync: float = 0.40
    eye_authenticity: float = 0.15
    corroboration_boost: float = 0.15      # per extra HIGH input, as in evidence_fusion
    single_input_floor: float = 0.75       # sub-score >= 75% of the strongest input, so one clear finding is not averaged away (F-1 e2e: audio swap 65 -> 49 MEDIUM)
    single_signal_cap: float = 60.0        # one input alone can reach at most MEDIUM ...
    strong_identity_switch_floor: float = 75.0  # ... except a confident identity switch (>= 2 people)


FUSION_WEIGHTS = FusionWeights()
FUSION_CAPS = FusionCaps()
RISK_THRESHOLDS = RiskThresholds()
SPEECH_WEIGHTS = SpeechBehaviorWeights()

VISION_BUDGET = VisionBudget()
VAD_SETTINGS = VadSettings()
FACE_QUALITY = FaceQuality()
IDENTITY_THRESHOLDS = IdentityThresholds()
SYNC_THRESHOLDS = SyncThresholds()
GAZE_THRESHOLDS = GazeThresholds()
EYE_AUTH_THRESHOLDS = EyeAuthenticityThresholds()
DEEPFAKE_WEIGHTS = DeepfakeWeights()

# answer-evaluation weighting (semantic similarity vs concept coverage)
ANSWER_EVAL_SEMANTIC_WEIGHT = 0.5
ANSWER_EVAL_CONCEPT_WEIGHT = 0.5

CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
