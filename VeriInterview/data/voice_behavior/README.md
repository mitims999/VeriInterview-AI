# Voice behavior evaluation fixtures

Run `scripts/generate_voice_fixtures.py --source synthetic-speech.wav` from the
project directory. Source audio must be synthetic/consented mono 16 kHz WAV.
Outputs are ignored under `storage/audio_analysis/fixtures/`; metadata includes
condition, duration, expected measurement behavior and synthetic speaker ID.

Ten controlled cases cover baseline, normal, gain shift, combined rate/pitch
shift, inserted pause, noise, clipping, short audio and silence. The generator
does not create real candidate recordings or integrity/cheating labels.

These cases verify extraction and failure handling. The current fixtures use
one synthetic voice and cannot validate generalization, false-positive rates,
reading detection, or real interview-integrity decisions. A difficult-question
scenario needs actual consented recordings with question context; a transform
must not be presented as a recording of someone experiencing stress.

Future acoustic validation sources:

- [RAVDESS](https://zenodo.org/records/1188976)
- [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)

Review dataset licenses before downloading/reusing recordings. These datasets
label acted emotions, not cheating. If training a model later, split by speaker;
never distribute transformed copies of a source speaker across train and test.
