"""
M3 Deepfake Vision internals (lip-sync, face identity, gaze/proctoring).

These modules do the raw media work; the public contract stays in
backend/adapters/{syncnet,insightface,proctoring}_adapter.py. Every module
here imports its heavy dependencies lazily so the core app runs without
requirements-vision.txt installed.
"""
