# M3 — Data Sources for Every Task

Companion to [`M3_TASKS.md`](M3_TASKS.md) · tracked in [`M3_PROGRESS.md`](M3_PROGRESS.md)
Researched 2026-10-12. **Only public / online data is used. The local `data/samples/` videos are not used.**

> ⚠️ **Privacy and spec rule (Master Spec §25, §29).** Every dataset below contains **real people's faces and voices**. These datasets are used **only offline**, to calibrate thresholds and run tests. They live **outside git** in `datasets/` (gitignored). They are never committed, never shown in the demo, and never copied into `data/synthetic/`. The demo stays fully synthetic. Only derived numbers go into the repo: thresholds, CSV summaries without images, and pass/fail tables.

---

## 1. Summary — what to get, in what order

| Tier | Meaning | Datasets |
|---|---|---|
| **A — open, download today** | No form, or an automatic loader | **GRID** (CC BY 4.0, Zenodo) · **LFW** via `sklearn.fetch_lfw_pairs` · **DeepfakeTIMIT** (Zenodo) · **OEP proctoring videos** (Kaggle) · **Student suspicious-behaviour features** (Mendeley, CC BY 4.0) · **RT-BENE** (Zenodo, CC BY-NC-SA) · **COCO val2017** subset · **AVLips val split** (Google Drive) |
| **B — request form / EULA, apply now** (approval takes days) | Gated, but free for research | **FakeAVCeleb** (Google form, CC BY 4.0) · **AV-Deepfake1M** (HF gated + EULA, CC BY-NC) · **Celeb-DF v2** (form) · **FaceForensics++** (form) · **VoxCeleb2** (request, CC BY 4.0) · **MPIIFaceGaze** (CC BY-NC-SA) |
| **C — optional / heavy / uncertain** | Only if time allows | YouTube Faces (~25 GB, registration) · DFDC (huge) · BIWI head pose (~11 GB) · Columbia Gaze · VCF video-conference deepfakes (release not found) · **LRS2/LRS3 — currently unavailable** (official pages down, per mirror note) |

**Minimum viable data kit (everything in Tier A, about 5 GB if subsetted):**
GRID 4 talkers (~1.7 GB) + DeepfakeTIMIT (227 MB) + LFW (auto, ~200 MB) + OEP (Kaggle) + RT-BENE subset + COCO val2017 person/phone subset (~1 GB) + AVLips val split.
This kit unblocks **every MVP task**. Tier B datasets make the validation stronger.

---

## 2. Task → data map

| Task | What the data is for | Primary data | Backup / extra |
|---|---|---|---|
| **G-0** *(redefined)* Acquire datasets + build `datasets/` manifest | — | Everything in Tier A | Submit Tier B forms on day 1 |
| **G-3** media.py (decode, WAV, VAD) | Varied containers and codecs, silence and speech segments | GRID `.mpg` + 25 kHz WAV; DeepfakeTIMIT `.avi`; OEP `.avi` | FakeAVCeleb `.mp4` |
| **G-5** face_track.py (detect, track, quality gating) | Frontal talking heads, webcam quality, multiple faces, faces leaving frame | GRID (clean frontal), OEP (real webcam, lighting/pose variety) | Celeb-DF / FF++ (in-the-wild interviews) |
| **I-1** identity anchor + similarity trace | **ArcFace same-person threshold** calibration | **LFW pairs** (`fetch_lfw_pairs(subset="10_folds", resize=1.0, color=True)`): genuine and impostor cosine distributions → pick a threshold at a target FAR | VoxCeleb2 / YouTube Faces (video-level, pose variation) |
| **I-2** identity-switch detection | Clips where the identity changes mid-video | **Generated:** splice GRID talker A → talker B (recipe §4). **Real face swaps:** DeepfakeTIMIT (16 swapped pairs, VidTIMIT source) | FakeAVCeleb *FakeVideo* categories, Celeb-DF v2, FF++ |
| **I-3** insightface_adapter calibration | Genuine sessions must be LOW, swaps must be HIGH | GRID genuine vs splices; DeepfakeTIMIT real (VidTIMIT) vs fake | FakeAVCeleb real vs face-swap |
| **S-1** reproduce SyncNet reference | Known expected output | `example.avi` from the syncnet download (offset 3, conf ≈ 10.0) | — |
| **S-2** per-segment SyncNet runner | Synced talking faces with speech | **GRID** (frontal, 1000 sentences/talker, word alignments give exact speech segments) | VoxCeleb2 test, AVLips real |
| **S-3** SyncNet calibration (offset bands, LSE-C → 0–1) | Controlled positives and negatives | **Generated from GRID:** audio shifted ±2/±5/±10 frames, audio swapped from another talker, silent audio (recipe §4) | **AVLips** (Wav2Lip/TalkLip/SadTalker/MakeItTalk fakes) and **FakeAVCeleb** *RealVideo-FakeAudio / FakeVideo-FakeAudio (Wav2Lip)*, to document what SyncNet catches and what it misses (LIMITATIONS) |
| **S-4** syncnet_adapter | Genuine LOW / desync HIGH / no-speech gives INSUFFICIENT | Same as S-3 + GRID silence segments | — |
| **P-1** MediaPipe landmarker coverage | Landmark success rate | OEP webcam videos, GRID | MPIIFaceGaze |
| **P-2** head pose (solvePnP → Euler) | Ground-truth yaw/pitch | **BIWI** (Tier C) or **AFLW2000-3D**; otherwise cross-check against the MediaPipe transformation matrix only | — |
| **P-3** iris gaze rule + debounce | Ground-truth gaze direction or "looking away" | **OEP** (webcam exam videos with `gt.txt` cheating intervals: looking at notes or elsewhere) | MPIIFaceGaze, Columbia Gaze |
| **P-4** blink rate + mouth-vs-speech | Blink ground truth; speaking vs not speaking | **RT-BENE** (eye open/closed labels); mouth-vs-speech: **GRID** alignments (exact word timings) | Eyeblink8 (download source not found), AVA-ActiveSpeaker |
| **P-5** YOLO person/phone | Person (COCO class 0) and cell phone (class 67) precision at score ≥ 0.6 | **COCO val2017** filtered to `person` / `cell phone` | OEP (phone use during exams), Student suspicious-behaviour dataset (features only) |
| **P-6** proctoring_adapter | Calibrate weak-evidence cap; check nervous ≠ cheating | **OEP** cheating vs non-cheating intervals | Student suspicious-behaviour features (CC BY 4.0, 5,500 labelled rows) as a sanity check on our feature ranges |
| **F-1/F-3** deepfake_fusion | End-to-end mixed cases | Generated GRID set (genuine / desync / splice / both) + DeepfakeTIMIT + (FakeAVCeleb 4 categories) | AV-Deepfake1M (localised AV manipulations) |
| **F-4** nervous-vs-fake (R1) | Real *stressed but genuine* people vs real fakes | **OEP** (students under exam pressure) + **UBFC-Phys** (56 people filmed at rest vs speech and arithmetic stress tasks) vs DeepfakeTIMIT / AVLips / generated set | ChaLearn First Impressions V2, MIT Interview Dataset |
| **N-4** performance | 3–5 min interview-like clips | Concatenated GRID sentences of one talker (~3 min) | OEP videos (several minutes each) |
| **T-2** model-backed tests | Small committed **manifest** (paths + expected outcome), no media in git | 6 clips: 2 genuine GRID, 1 shifted, 1 audio-swap, 1 splice, 1 DeepfakeTIMIT fake | — |

---

## 3. Dataset catalogue

### Tier A — open now

| Dataset | What it is | Size | License / terms | Access | We use |
|---|---|---|---|---|---|
| **GRID audiovisual corpus** (⚠️ use the **Sheffield** per-talker alignments: the Zenodo `alignments.zip` has mismatched talker folders) | 34 talkers × 1000 sentences, frontal face video + audio + word alignments. Video at 360×288 (normal) or 720×576 (HQ), `.mpg`; 25 kHz / 50 kHz audio | 16.2 GB total; ~400 MB per talker | **CC BY 4.0** (Zenodo record). The Sheffield page says "freely available for research use" | [Zenodo 10.5281/zenodo.3625687](https://zenodo.org/records/3625687) · [Sheffield page](https://spandh.dcs.shef.ac.uk/gridcorpus/) | 4–6 talkers (mixed gender) as the **backbone for SyncNet, identity-splice and mouth-vs-speech** calibration. Alignments give exact speech timing |
| **LFW (pairs)** | Face verification pairs, labelled same/different | ~200 MB, auto-download | Research use (no explicit license; cite Huang et al.) | `sklearn.datasets.fetch_lfw_pairs` ([docs](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_pairs.html)) | ArcFace cosine **threshold** for `IdentityThresholds` (use `resize=1.0`, colour, then re-detect and align with InsightFace) |
| **DeepfakeTIMIT** | 32 subjects with face-swaps (16 look-alike pairs) at 64×64 and 128×128 GAN quality, built from VidTIMIT. Audio untouched | 227 MB | Must cite Korshunov & Marcel; VidTIMIT copyright terms apply (read before use) | [Zenodo 4068245](https://zenodo.org/records/4068245) · [Idiap page](https://idiap.ch/dataset/deepfaketimit) · [VidTIMIT](https://conradsanderson.id.au/vidtimit) | Real face-swap positives for identity drift + fusion. Original VidTIMIT gives the real negatives |
| **OEP — Online Exam Proctoring (MSU)** | Webcam videos of students in exams, with `gt.txt` cheating intervals | Several GB | Cite Atoum et al., IEEE TMM 2017. License not stated; research use | [MSU CV Lab page](https://cvlab.cse.msu.edu/oep-dataset.html) · [Kaggle copy](https://www.kaggle.com/datasets/raajanwankhade/oep-dataset) (via [this repo](https://github.com/AnasMultani17/Cheating-Detection-From-Video)) | **The closest public match to interview proctoring.** Gaze-off-screen, phone, and person validation, plus false-positive checks on non-cheating intervals |
| **Student suspicious-behaviours (Mendeley, 2025)** | 5,500 labelled per-frame feature rows (face, hands, head pose, phone, gaze), cheating vs not | Small (tabular) | **CC BY 4.0** | [Mendeley 10.17632/39xs8th543.1](https://data.mendeley.com/datasets/39xs8th543/1) | Sanity check on feature ranges and the head-pose/gaze thresholds. **No video** |
| **RT-BENE** | 200k+ eye images with open/closed (blink) labels | Several GB (subset is fine) | **CC BY-NC-SA 4.0** (non-commercial) | [Zenodo 3685316](https://zenodo.org/records/3685316) | Validate the EAR / blendshape blink classifier (P-4) |
| **COCO val2017** | Object detection, 80 classes | ~1 GB images + annotations | Annotations CC BY 4.0; images under Flickr terms | cocodataset.org (filter with FiftyOne or pycocotools) | Precision/recall of YOLOv3(-tiny) for `person` (0) and `cell phone` (67) at the Proctoring-AI thresholds |
| **AVLips (LipFD)** | Lip-sync forgeries: MakeItTalk, Wav2Lip, TalkLip, SadTalker + perturbations | Full ~60 GB preprocessed. **Use the val split only** | No license stated in the README; research | [AaronComo/LipFD](https://github.com/AaronComo/LipFD) (Google Drive links) | Measure how often SyncNet is **fooled** by SyncNet-trained generators, and document it in LIMITATIONS. Also feeds stretch X-2 |

### Tier B — submit forms now

| Dataset | What it is | Size | License / terms | Access | We use |
|---|---|---|---|---|---|
| **FakeAVCeleb** | 500 real (VoxCeleb2-sourced) + 19,500 fakes. Fakes are video and/or cloned audio across 4 categories (RealV-RealA, RealV-FakeA, FakeV-RealA, FakeV-FakeA) | ~20k clips (download a subset) | **CC BY 4.0** data, MIT code; DUA via Google form, manually screened | [DASH-Lab/FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb) · [paper](https://arxiv.org/pdf/2108.05080) | **Best single benchmark for M3**: it covers identity + lip-sync + audio. Used for the final fusion evaluation |
| **AV-Deepfake1M** | 1M+ clips with **localised** audio/video manipulations (word-level edits) | Very large (multi-volume 7z); subset only | EULA + **CC BY-NC 4.0** | [HF ControlNet/AV-Deepfake1M](https://huggingface.co/datasets/ControlNet/AV-Deepfake1M) · [paper](https://arxiv.org/html/2311.15308v2) | Per-segment SyncNet: can we localise *which answer* was manipulated? |
| **Celeb-DF v2** | 590 real celebrity **interview** videos + 5,639 face-swaps | ~10 GB | Celeb-DF terms of use; form | [danmohaha/celeb-deepfakeforensics](https://github.com/danmohaha/celeb-deepfakeforensics) | Identity drift on interview-style footage |
| **FaceForensics++** | 1,000 YouTube videos × 4 manipulation methods, plus the Google DeepFakeDetection subset (consented actors) | Up to TBs; c23 subset ~10 GB | FF++ Terms of Use; form | [ondyari/FaceForensics](https://github.com/ondyari/faceforensics) | Optional extra identity-swap positives |
| **VoxCeleb2** | 1M+ utterances, 6,112 speakers, cropped face videos | Large; test split only | **CC BY 4.0** (official license text); copyright remains with video owners | [VoxCeleb site](https://mm.kaist.ac.kr/datasets/voxceleb/) | Video-level identity stability across pose/lighting; in-the-wild SyncNet negatives |
| **MPIIFaceGaze** | 15 participants, laptop-webcam images with 2-D gaze labels | ~940 MB | **CC BY-NC-SA 4.0** | [MPIIFaceGaze page](https://perceptualui.org/research/datasets/MPIIFaceGaze) | Quantitative check that the iris-ratio rule separates on-screen from off-screen |

### Nervous-but-genuine data (for requirement R1 / task F-4)

Background research: stress **raises blink rate** ([PMC12649463](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12649463/), [review arXiv 2202.03033](https://arxiv.org/pdf/2202.03033)) and **increases gaze aversion** ([vTSST gaze-cortisol dataset](https://b2find.eudat.eu/dataset/c9b227ee-6b29-59b9-9503-f0ed24aa6cb1)). This is why the deepfake blink cue is one-sided (see `M3_TASKS.md` R1).

| Dataset | What it is | Access | We use |
|---|---|---|---|
| **OEP** (Tier A, downloading) | Real students on webcam under exam pressure | free | Stressed-real negatives: the deepfake score must stay LOW |
| **UBFC-Phys** | 56 participants filmed during rest (T1), speech (T2) and arithmetic (T3) stress tasks, with anxiety questionnaires | [IEEE DataPort 10.21227/5da0-7344](https://ieee-dataport.org/node/3658) (free account) · [UBFC portal](https://search-data.ubfc.fr/FR-18008901306731-2022-05-05) · [project page](https://lead.ube.fr/?p=9490) | Paired test: the deepfake score at rest vs under stress should show no shift |
| **ChaLearn First Impressions V2** (optional) | 10k × 15 s clips of people talking to camera (YouTube) | [ChaLearn LAP](https://chalearnlap.cvc.uab.cat/dataset/24/data/41/description/) (sign-in) | Varied real talking heads (in-the-wild false-positive check) |
| **MIT Interview Dataset** (optional) | 138 mock job interviews (69 MIT students) | [ROC-HCI page](https://roc-hci.com/past-projects/automated-prediction-of-job-interview-performances/) (request; sensitive) | The most interview-like real data, if granted |

### Tier C — optional

| Dataset | Notes | Link |
|---|---|---|
| YouTube Faces DB | 3,425 videos / 1,595 people, ~25 GB, registration form | [TAU page](https://www.cs.tau.ac.il/~wolf/ytfaces/) |
| DFDC | 100k+ clips, 3,426 consented paid actors; very large | [Meta DFDC](https://ai.facebook.com/datasets/dfdc) · [paper](https://arxiv.org/abs/2006.07397v1) |
| BIWI Kinect Head Pose | ~15k frames, 20 people, ±75° yaw ground truth; ~11 GB; non-commercial | [HF ETHZurich/biwi_kinect_head_pose](https://huggingface.co/datasets/ETHZurich/biwi_kinect_head_pose) |
| Columbia Gaze | 5,880 images / 56 people; non-commercial | [Columbia CAVE](https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/) |
| VCF (video-conference deepfakes, 2025) | Most relevant setting (H.264, call resolutions), but **no public release URL found** | [paper](https://isprs-archives.copernicus.org/articles/XLVIII-2-W9-2025/169/2025/) |
| LRS2 / LRS3 | The standard SyncNet benchmark, but **official downloads are currently unavailable** | [KAIST mirror note](https://mm.kaist.ac.kr/datasets/lip_reading) |

---

## 4. Generated test sets (built from GRID; reproducible script)

There is no public dataset of interview deepfakes, so we generate **controlled manipulations** from CC-BY GRID clips. `scripts/build_m3_testset.py` (part of S-3) writes them to `datasets/m3_generated/` together with a `manifest.csv` (clip, manipulation, parameters, expected outcome).

| Set | How | Expected M3 outcome |
|---|---|---|
| `genuine_long` | Concatenate ~40 sentences from one talker (~3 min) → "interview" | identity LOW, sync LOW, fusion LOW |
| `av_shift_{2,5,10}` | ffmpeg `-itsoffset` on the audio by N/25 s | offset recovered ±1 frame. 2 → LOW, 5 → MEDIUM, 10 → HIGH |
| `av_shift_partial` | Shift only the middle third (simulates one manipulated answer) | the affected segment is flagged and the other segments are clean |
| `audio_swap` | Talker A video + talker B audio (same sentences) | low LSE-C → sync MEDIUM/HIGH, identity LOW |
| `identity_splice` | Talker A for the first half, then talker B with the same audio track | `distinct_identities=2`, identity HIGH, drift event at the cut |
| `splice_plus_shift` | Both of the above | fusion HIGH (corroboration rule) |
| `silent_segment` | Replace 30 s of audio with silence | that segment gives INSUFFICIENT_EVIDENCE, not risk |
| `face_occluded` | Black out the face region for 20 s | `face_lost_pct` rises; low confidence, not high risk |
| `low_quality` | Re-encode at 240p / CRF 40 | confidence goes down; risk stays roughly the same |

---

## 5. Model weights (all downloaded by `scripts/download_models.py` into gitignored `storage/models/`)

| Model | Source | Size | License |
|---|---|---|---|
| SyncNet `syncnet_v2.model` | `http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model` | ~50 MB | Code MIT. Check the weight terms with VGG |
| SyncNet reference clip `example.avi` | same host, `/data/example.avi` | small | — |
| InsightFace `buffalo_l` (SCRFD-10GF + ArcFace R50) | auto-download by `FaceAnalysis(name="buffalo_l", root=storage/models)` | ~326 MB | **Non-commercial research only** |
| MediaPipe Face Landmarker | `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task` | ~4 MB | Apache-2.0 |
| YOLOv3-tiny (default) | `yolov3-tiny.cfg` / `.weights` from the Darknet project (pjreddie / AlexeyAB releases) | ~34 MB | Darknet: public domain / YOLO license |
| YOLOv3 (optional, more accurate) | `yolov3.weights` (Darknet) | ~248 MB | same |

*Not needed:* S3FD `sfd_face.pth` (replaced by SCRFD per D2), Proctoring-AI TF landmark model, dlib.

---

## 6. Storage layout (none of it committed)

```
datasets/                      # gitignored, outside the app's storage/
  grid/ s1 s2 s3 s4 ... alignments/
  lfw/                         # sklearn cache (data_home=datasets/lfw)
  deepfaketimit/  vidtimit/
  oep/
  rt_bene/  coco_person_phone/
  avlips_val/
  fakeavceleb/  av_deepfake1m_subset/  ...   # when approved
  m3_generated/  manifest.csv
storage/models/                # weights
docs/m3/calibration/*.csv      # COMMITTED: numbers only, no media or embeddings
```

Disk estimate: Tier A minimal kit is about 5–8 GB. With FakeAVCeleb and AV-Deepfake1M subsets, about 30 GB.

---

## 7. Action items for you (owner M3)

1. **Today:** submit access forms for **FakeAVCeleb**, **Celeb-DF v2**, **AV-Deepfake1M** (Hugging Face login + EULA), **VoxCeleb2**, and optionally **FaceForensics++**. Some of these need an institutional email.
2. Have a **Kaggle account** ready (OEP dataset).
3. Confirm about 10 GB of free disk for Tier A.
4. Decide whether non-commercial (NC) datasets are acceptable. They are fine for a research hackathon, and none of them ship in the product.
