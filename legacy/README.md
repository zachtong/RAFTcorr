# RAFTcorr 1.x (legacy)

The Flask + React application that RAFTcorr 2.0 replaces, moved here unchanged on 2026-11-18
with `git mv` (history is preserved). It is kept for reference only and is not maintained:
the 1.x network code cannot load the RAFTcorr-s1/s2/s4/s8 models.

- `server/`, `frontend/`, `raft_dic_gui/`, `core/`, `alt_cuda_corr/`: the 1.x application
- `tests/`, `pytest.ini`, `ci_v1.yml`: its tests and CI workflow
- `run*.py`, `run.bat`, `install.bat`, `setup.py`: its launchers and installer
- `notebooks/`: the 1.x Colab notebook
- `UNCOMMITTED_2026-11-18_streaming_export.patch`: the uncommitted streaming-export work that
  was in the working tree on 2026-11-18 (`git apply` it at the 1.x layout to restore it)

Once 2.0 is accepted this folder can be removed in one commit; the code stays in git history.
