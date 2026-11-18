"""Canonical field names of the GUI (pyALDIC's ``core.fields``) and their RAFTcorr keys.

The GUI widgets ported from pyALDIC name quantities ``disp_u``, ``strain_exx`` ...; RAFTcorr's
:class:`raftcorr.viz.fields.FieldProvider` and :mod:`raftcorr.post.strain` call them ``u``,
``exx`` .... :data:`TO_RAFTCORR` maps the first to the second; use :func:`provider_key`.

Displacements carry a length unit and follow the physical-unit setting. Strains are
dimensionless and are never scaled by it. Rotation and the principal angle are in degrees.
"""

from __future__ import annotations

#: Displacement components, in the reference (frame-1) coordinate system.
DISP_FIELDS: frozenset[str] = frozenset(["disp_u", "disp_v", "disp_magnitude"])

#: Strain components produced by the strain stage (pyALDIC names + RAFTcorr extras).
STRAIN_FIELDS: frozenset[str] = frozenset([
    "strain_exx", "strain_eyy", "strain_exy",
    "strain_principal_max", "strain_principal_min",
    "strain_maxshear", "strain_von_mises", "strain_rotation",
    "strain_principal_angle", "strain_area_change",
])

#: Everything a probe or an export may be asked for.
ALL_FIELDS: frozenset[str] = DISP_FIELDS | STRAIN_FIELDS

_ANGLE_FIELDS: frozenset[str] = frozenset(["strain_rotation", "strain_principal_angle"])

#: GUI name -> RAFTcorr FieldProvider / StrainSeries key.
TO_RAFTCORR: dict[str, str] = {
    "disp_u": "u",
    "disp_v": "v",
    "disp_magnitude": "magnitude",
    "disp_mag": "magnitude",  # projects saved before 2.0.0 settled the name
    "residual": "residual",
    "strain_exx": "exx",
    "strain_eyy": "eyy",
    "strain_exy": "exy",
    "strain_principal_max": "e1",
    "strain_principal_min": "e2",
    "strain_maxshear": "max_shear",
    "strain_von_mises": "von_mises",
    "strain_rotation": "rotation",
    "strain_principal_angle": "principal_angle",
    "strain_area_change": "area_change",
}
FROM_RAFTCORR: dict[str, str] = {v: k for k, v in TO_RAFTCORR.items() if k != "disp_mag"}


def strain_symbol(name: str) -> str:
    """The label of a strain field (GUI name, e.g. ``strain_maxshear``): ``post.strain.COMPONENT_LABELS``,
    the one name the strain window, the analysis tab, the export dialog and the files all use."""
    from raftcorr.post.strain import COMPONENT_LABELS

    return COMPONENT_LABELS[TO_RAFTCORR[name]]


def provider_key(name: str) -> str:
    """RAFTcorr key of a GUI field name (unknown names pass through unchanged)."""
    return TO_RAFTCORR.get(name, name)


def is_strain_field(name: str) -> bool:
    """True when *name* is a strain component rather than a displacement."""
    return name in STRAIN_FIELDS


def field_unit(name: str, length_unit: str) -> str:
    """Unit string for *name* given the configured *length_unit*."""
    validate_field(name)
    if name in _ANGLE_FIELDS:
        return "°"
    if is_strain_field(name):
        return ""
    return length_unit


def validate_field(name: str) -> None:
    """Raise ``ValueError`` unless *name* is a known field (listing the near-misses)."""
    if name in ALL_FIELDS:
        return
    prefix = name.split("_")[0] if "_" in name else name
    near = sorted(f for f in ALL_FIELDS if f.startswith(prefix))
    raise ValueError(f"Unknown field {name!r}. Known fields: {', '.join(near or sorted(ALL_FIELDS))}")
