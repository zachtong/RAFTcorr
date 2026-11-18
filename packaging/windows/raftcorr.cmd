@echo off
rem The `raftcorr` command of a Windows installation: the installer's Python runs RAFTcorr.
rem Only this folder (bin) goes on PATH, so the private Python, pip and the other tools stay off it.
"%~dp0..\python\python.exe" -m raftcorr %*
