"""Multimodal scanner data system (Dr. Zhang's spec).

A unified local-MongoDB store for samples, scans, and per-modality artifacts
(projector/fringe, kinect, laser, fusion, calibration), plus a Streamlit GUI.

Nothing in this package imports hardware or opens a Mongo connection at import
time; the Mongo client is built lazily. See docs in zhang_system_design.
"""
