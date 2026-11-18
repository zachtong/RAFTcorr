"""Entry script of the frozen macOS application (RAFTcorr.app).

Without arguments it opens the application; with a command it is the ``raftcorr`` command line, so
``RAFTcorr.app/Contents/MacOS/RAFTcorr run images/ ...`` works from a terminal. Kept outside the
package, as pyALDIC's launcher is, so the frozen entry point is never imported as a module.
"""

import multiprocessing
import sys


def _argv() -> list[str]:
    # Older macOS passes a process serial number when Finder opens an app: not a RAFTcorr argument.
    return [a for a in sys.argv[1:] if not a.startswith("-psn_")]


if __name__ == "__main__":
    multiprocessing.freeze_support()     # a frozen app re-runs this script in child processes
    from raftcorr.cli.main import main

    sys.exit(main(_argv()))
