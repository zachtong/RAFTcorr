"""Frame order, sign and axis conventions. Binding for every module of the package.

These mirror the research repository's ``CONVENTION.md``; the trained models follow them.

Frames
    ``reference`` is the undeformed image (frame 0 of a sequence, or the current reference
    frame of an incremental schedule); ``deformed`` is the image being measured. The network is
    always called as ``model(reference, deformed)``.

Displacement
    The reference -> deformed displacement ``(u, v)`` of a material point: a point at reference
    pixel ``X`` appears in the deformed image at ``x = X + d(X)``.

    * ``+u`` points right (increasing column index, image ``x``);
    * ``+v`` points down (increasing row index, image ``y``);
    * arrays are indexed ``[row, column]`` = ``[y, x]``; flow tensors have channel 0 = u,
      channel 1 = v;
    * no sign flip is applied anywhere.

Coordinates of results
    Every field of a sequence is stored in frame-0 (Lagrangian, reference) coordinates: the value
    at ``[y, x]`` belongs to the material point that sits at pixel ``(x, y)`` in frame 0. Fields
    drawn "on the deformed frame" are mapped there for display only.

Images
    Grey levels as float in ``[0, 255]``; colour images are converted to luminance; images with
    more than 8 bits are scaled by their effective bit depth (see :mod:`raftcorr.io.images`).
    The network tiles grey to three channels and normalises to ``[-1, 1]`` internally.

Strain
    Engineering shear is never used: ``exy`` is the tensor component. Green-Lagrange strain
    ``E = (F^T F - I) / 2`` with ``F = I + grad(u)`` in reference coordinates.
"""

from __future__ import annotations

#: Channel order of a displacement array with a trailing component axis.
U_CHANNEL = 0
V_CHANNEL = 1

#: Every released model pads its inputs to a multiple of this.
PAD_MULTIPLE = 8
