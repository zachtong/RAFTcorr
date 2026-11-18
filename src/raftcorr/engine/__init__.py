"""Inference engine: one pair (backend), memory model, tiling, sequence pipeline, field mapping.

Submodules are imported explicitly (``from raftcorr.engine.pipeline import Pipeline``); this
package module stays import-free so that light consumers (results, views) do not pull in torch.
"""
