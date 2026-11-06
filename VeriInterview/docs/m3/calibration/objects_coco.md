# Person / phone detection on COCO val2017 (task P-5)

Subset: 914 images (214 with a cell phone, 250 single-person, 250 multi-person, 200 with neither; `datasets/coco/person_phone/subset.json`). Image-level decision: >= 2 persons = multi-person, any phone = phone. `backend.vision.objects.detect` via cv2.dnn, NMS 0.5.

## YOLOv3-tiny (35 MB, 147 ms/image)

| threshold | multi-person P | R | phone P | R | phone false alarms / 200 negatives |
|---|---|---|---|---|---|
| 0.2 | 0.91 | 0.46 | 0.95 | 0.17 | 0 |
| 0.4 | 0.89 | 0.26 | 1.00 | 0.11 | 0 |
| 0.6 (upstream) | 0.85 | 0.15 | 1.00 | 0.07 | 0 |

## YOLOv3 full (248 MB, ~1.6 s/image on CPU with other jobs running)

| threshold | multi-person precision | recall | phone precision | recall | phone false alarms on 200 negatives |
|---|---|---|---|---|---|
| 0.2 | 0.96 | 0.88 | 0.96 | 0.64 | 0 |
| 0.3 | 0.97 | 0.87 | 0.98 | 0.56 | 0 |
| **0.4 (chosen)** | **0.97** | **0.84** | **0.98** | **0.50** | **0** |
| 0.5 | 0.98 | 0.80 | 0.99 | 0.47 | 0 |
| 0.6 (upstream) | 0.98 | 0.78 | 0.99 | 0.40 | 0 |

**Chosen:** full YOLOv3 as default, score 0.4 (multi-person P 0.97 / R 0.84, phone P 0.98 / R 0.50, no false phone on any negative), plus the 2-consecutive-sample persistence rule. COCO phones are often tiny or occluded, so webcam phone recall should be higher; this is a pessimistic bound. Speed (1.6 s/image) → sampling rate is set in N-4.

## Real webcam check on OEP phone-use intervals (`scripts/eval_objects_oep.py`)

Full YOLOv3, score 0.4, single frames at 1 fps sampled inside the labelled phone-use (type 5) intervals and from the same number of non-cheating moments:

| subject | phone-use frames | phone detected | extra person | normal frames | phone detected | extra person |
|---|---|---|---|---|---|---|
| subject1 | 33 | 55% | 0% | 33 | 0% | 0% |
| subject6 | 19 | 5% | 0% | 19 | 0% | 0% |
| subject9 | 40 | 80% | 0% | 40 | 5% | 0% |
| subject24 | 24 | 50% | 0% | 24 | 0% | 0% |

Phone visible in 63/116 phone-use frames (54%) vs 2/116 normal frames (1.7%). Extra person: 0 in phone-use frames, 0 in normal frames.

Phone use is seen in about half of the frames (better than COCO's 0.50 bound, as expected for a webcam). The 1.7 % single-frame false rate would give ~0.5 spurious alerts per 30 min at 1 fps with a 2-sample rule, so **`object_persist_samples` is 3** (~0.01). Real phone use lasts many seconds, so it is still caught.
