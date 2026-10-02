"""Judge whether a CIF crystal structure belongs to a named structure type.

The CIF is parsed on the client: symmetry operations are applied to the asymmetric unit to get
the full cell, then composition, Z, and the first coordination shell of every site are computed.
These features (plus the raw CIF) go to Clef together with several questions that look at the
structure from different angles. The answers are combined into a verdict and cross-checked
against the parsed facts.

    uv run client/examples/cif_structure.py client/examples/data/NaCl.cif --structure rocksalt
    uv run client/examples/cif_structure.py client/examples/data/*.cif -s perovskite --json
    uv run client/examples/cif_structure.py my.cif -s "Ruddlesden-Popper" --hint "A(n+1)B(n)O(3n+1) layered perovskite"
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import sys
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from functools import reduce
from pathlib import Path
from typing import Any

from clef_client import ClefClient, ClefError, choice, noul, score

# Short definitions of common structure types. Sent as a hint so the model judges against the
# same definition every time. Unknown names still work; pass --hint to describe them.
STRUCTURE_HINTS = {
    "rocksalt": "NaCl type, AB, Fm-3m (225), two interpenetrating fcc lattices, both A and B octahedral (CN 6).",
    "cscl": "CsCl type, AB, Pm-3m (221), simple cubic with B at the body center, both A and B cubic (CN 8).",
    "zincblende": "ZnS sphalerite type, AB, F-43m (216), both A and B tetrahedral (CN 4), cubic stacking.",
    "wurtzite": "ZnS wurtzite type, AB, P6_3mc (186), both A and B tetrahedral (CN 4), hexagonal stacking.",
    "fluorite": "CaF2 type, AB2, Fm-3m (225), A cubic (CN 8), B tetrahedral (CN 4).",
    "antifluorite": "Li2O type, A2B, Fm-3m (225), A tetrahedral (CN 4), B cubic (CN 8).",
    "rutile": "TiO2 rutile type, AB2, P4_2/mnm (136), A octahedral (CN 6), B trigonal planar (CN 3), edge-sharing chains.",
    "perovskite": (
        "ABX3 perovskite, aristotype Pm-3m (221), corner-sharing BX6 octahedra, A in the 12-coordinate cavity. "
        "Tilted/distorted variants (e.g. GdFeO3-type Pnma, R-3c) are perovskite derivatives."
    ),
    "spinel": "AB2X4 spinel, Fd-3m (227), A tetrahedral (CN 4), B octahedral (CN 6), cubic close-packed X.",
    "corundum": "Al2O3 type, A2B3, R-3c (167), A octahedral (CN 6) in 2/3 of octahedral holes of hcp B.",
    "nias": "NiAs type, AB, P6_3/mmc (194), A octahedral (CN 6), B trigonal prismatic (CN 6).",
    "diamond": "Diamond type, single element, Fd-3m (227), tetrahedral (CN 4).",
    "fcc": "Cu type, single element, Fm-3m (225), CN 12 cubic close packing.",
    "bcc": "W type, single element, Im-3m (229), CN 8 (+6).",
    "hcp": "Mg type, single element, P6_3/mmc (194), CN 12 hexagonal close packing.",
}
ALIASES = {"nacl": "rocksalt", "rock salt": "rocksalt", "rock-salt": "rocksalt", "sphalerite": "zincblende", "caf2": "fluorite"}

CRYSTAL_SYSTEMS = {
    "triclinic": "Space groups 1-2",
    "monoclinic": "Space groups 3-15",
    "orthorhombic": "Space groups 16-74",
    "tetragonal": "Space groups 75-142",
    "trigonal": "Space groups 143-167",
    "hexagonal": "Space groups 168-194",
    "cubic": "Space groups 195-230",
}

MAX_ATOMS_FOR_GEOMETRY = 600
SHELL_TOLERANCE = 1.15  # neighbors within 15% of the shortest distance form the first shell


# --------------------------------------------------------------------------------------------
# CIF parsing
# --------------------------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Split CIF text into tokens; ``;``-delimited text fields become one token."""
    tokens: list[str] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith(";"):
            block = [line[1:]]
            index += 1
            while index < len(lines) and not lines[index].startswith(";"):
                block.append(lines[index])
                index += 1
            tokens.append("\n".join(block).strip())
        else:
            stripped = line.split("#", 1)[0] if "'" not in line and '"' not in line else line
            try:
                tokens.extend(shlex.split(stripped, comments=False, posix=True))
            except ValueError:
                tokens.extend(stripped.split())
        index += 1
    return tokens


def parse_cif(text: str) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Return (single data items, loops) of the first data block."""
    items: dict[str, str] = {}
    loops: list[dict[str, list[str]]] = []
    tokens = _tokenize(text)
    index = 0
    seen_block = False
    while index < len(tokens):
        token = tokens[index]
        lower = token.lower()
        if lower.startswith("data_"):
            if seen_block:
                break
            seen_block = True
            index += 1
        elif lower == "loop_":
            index += 1
            names: list[str] = []
            while index < len(tokens) and tokens[index].startswith("_"):
                names.append(tokens[index].lower())
                index += 1
            values: list[str] = []
            while index < len(tokens) and not tokens[index].startswith("_") and tokens[index].lower() not in ("loop_",) and not tokens[index].lower().startswith("data_"):
                values.append(tokens[index])
                index += 1
            loops.append({name: values[column :: len(names)] for column, name in enumerate(names)})
        elif token.startswith("_") and index + 1 < len(tokens):
            items[lower] = tokens[index + 1]
            index += 2
        else:
            index += 1
    rows = [
        [dict(zip(loop, row)) for row in zip(*loop.values())]
        for loop in loops
    ]
    return items, [row for loop in rows for row in loop]


def number(value: str | None) -> float | None:
    """Parse '5.6402(3)' -> 5.6402; '?' and '.' -> None."""
    if value is None or value in ("?", "."):
        return None
    try:
        return float(re.sub(r"\(\d+\)$", "", value))
    except ValueError:
        return None


def first(items: dict[str, str], *names: str) -> str | None:
    for name in names:
        if items.get(name) not in (None, "?", "."):
            return items[name]
    return None


# --------------------------------------------------------------------------------------------
# Symmetry and geometry
# --------------------------------------------------------------------------------------------


def parse_symop(operation: str) -> tuple[list[list[Fraction]], list[Fraction]]:
    """'-x+1/2, y-x, z+1/3' -> (3x3 rotation, translation)."""
    rotation: list[list[Fraction]] = []
    translation: list[Fraction] = []
    for component in operation.lower().replace(" ", "").split(","):
        row = [Fraction(0)] * 3
        shift = Fraction(0)
        for term in re.findall(r"[+-]?[^+-]+", component):
            sign = -1 if term.startswith("-") else 1
            term = term.lstrip("+-")
            axis = next((axis for axis in "xyz" if axis in term), None)
            if axis:
                coefficient = term.replace(axis, "").rstrip("*") or "1"
                row["xyz".index(axis)] += sign * Fraction(coefficient)
            else:
                shift += sign * Fraction(term)
        rotation.append(row)
        translation.append(shift)
    return rotation, translation


def lattice_matrix(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> list[list[float]]:
    """Rows are the Cartesian lattice vectors."""
    al, be, ga = (math.radians(angle) for angle in (alpha, beta, gamma))
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return [[a, 0.0, 0.0], [b * math.cos(ga), b * math.sin(ga), 0.0], [cx, cy, cz]]


def to_cartesian(fractional: tuple[float, float, float], lattice: list[list[float]]) -> tuple[float, ...]:
    return tuple(sum(fractional[i] * lattice[i][k] for i in range(3)) for k in range(3))


def crystal_system(space_group_number: int | None) -> str | None:
    if not space_group_number:
        return None
    bounds = [(2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"), (142, "tetragonal"), (167, "trigonal"), (194, "hexagonal"), (230, "cubic")]
    return next((name for upper, name in bounds if space_group_number <= upper), None)


def element(symbol: str) -> str:
    match = re.match(r"[A-Z][a-z]?", symbol.strip().capitalize())
    return match.group(0) if match else symbol


def reduced_formula(counts: Counter[str]) -> tuple[str, int]:
    rounded = {el: round(n) for el, n in counts.items() if round(n) > 0}
    divisor = reduce(math.gcd, rounded.values()) if rounded else 1
    formula = "".join(f"{el}{'' if n // divisor == 1 else n // divisor}" for el, n in sorted(rounded.items()))
    return formula, divisor


@dataclass
class Structure:
    path: str
    formula_sum: str | None
    space_group: str | None
    space_group_number: int | None
    cell: dict[str, float | None]
    sites: list[dict[str, Any]]
    symops: list[str]
    warnings: list[str] = field(default_factory=list)
    cell_composition: dict[str, float] = field(default_factory=dict)
    reduced_formula: str | None = None
    z: int | None = None
    coordination: list[dict[str, Any]] = field(default_factory=list)

    @property
    def crystal_system(self) -> str | None:
        return crystal_system(self.space_group_number)


def load_structure(path: Path) -> Structure:
    items, rows = parse_cif(path.read_text(errors="replace"))
    cell = {
        key: number(items.get(f"_cell_{name}"))
        for key, name in [
            ("a", "length_a"), ("b", "length_b"), ("c", "length_c"),
            ("alpha", "angle_alpha"), ("beta", "angle_beta"), ("gamma", "angle_gamma"),
            ("volume", "volume"),
        ]
    }
    space_group_number = number(first(items, "_space_group_it_number", "_symmetry_int_tables_number"))
    symops = [
        row.get("_space_group_symop_operation_xyz") or row.get("_symmetry_equiv_pos_as_xyz")
        for row in rows
        if row.get("_space_group_symop_operation_xyz") or row.get("_symmetry_equiv_pos_as_xyz")
    ]
    sites = []
    for row in rows:
        if "_atom_site_fract_x" not in row:
            continue
        label = row.get("_atom_site_label", "?")
        sites.append({
            "label": label,
            "element": element(row.get("_atom_site_type_symbol") or label),
            "wyckoff": (row.get("_atom_site_symmetry_multiplicity") or "") + (row.get("_atom_site_wyckoff_symbol") or ""),
            "xyz": tuple(number(row.get(f"_atom_site_fract_{axis}")) or 0.0 for axis in "xyz"),
            "occupancy": number(row.get("_atom_site_occupancy")) or 1.0,
        })
    structure = Structure(
        path=str(path),
        formula_sum=first(items, "_chemical_formula_sum"),
        space_group=first(items, "_space_group_name_h-m_alt", "_symmetry_space_group_name_h-m"),
        space_group_number=int(space_group_number) if space_group_number else None,
        cell=cell,
        sites=sites,
        symops=symops,
    )
    if not sites:
        structure.warnings.append("no _atom_site_fract_x/y/z loop")
    if not symops:
        structure.warnings.append("no symmetry operations; only the asymmetric unit is used")
        structure.symops = ["x,y,z"]
    if any(site["occupancy"] < 0.999 for site in sites):
        structure.warnings.append("partial occupancy present")
    if None in (cell["a"], cell["b"], cell["c"], cell["alpha"], cell["beta"], cell["gamma"]):
        structure.warnings.append("incomplete cell parameters; geometry skipped")
        return structure
    analyse_geometry(structure)
    return structure


def analyse_geometry(structure: Structure) -> None:
    """Expand the asymmetric unit, then compute cell composition, Z, and first coordination shells."""
    operations = [parse_symop(op) for op in structure.symops]
    atoms: list[tuple[str, int, tuple[float, float, float], float]] = []  # element, site index, xyz, occupancy
    for site_index, site in enumerate(structure.sites):
        seen: list[tuple[float, float, float]] = []
        for rotation, translation in operations:
            image = tuple(
                float(sum(rotation[i][k] * Fraction(site["xyz"][k]).limit_denominator(10**6) for k in range(3)) + translation[i]) % 1.0
                for i in range(3)
            )
            if not any(all(min(abs(p - q), 1 - abs(p - q)) < 1e-3 for p, q in zip(image, other)) for other in seen):
                seen.append(image)
        site["multiplicity"] = len(seen)
        atoms.extend((site["element"], site_index, xyz, site["occupancy"]) for xyz in seen)

    composition: Counter[str] = Counter()
    for el, _, _, occupancy in atoms:
        composition[el] += occupancy
    structure.cell_composition = {el: round(n, 3) for el, n in sorted(composition.items())}
    structure.reduced_formula, structure.z = reduced_formula(composition)

    if len(atoms) > MAX_ATOMS_FOR_GEOMETRY:
        structure.warnings.append(f"{len(atoms)} atoms in cell; coordination skipped")
        return
    cell = structure.cell
    lattice = lattice_matrix(cell["a"], cell["b"], cell["c"], cell["alpha"], cell["beta"], cell["gamma"])
    images = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    for site_index, site in enumerate(structure.sites):
        center = to_cartesian(site["xyz"], lattice)
        distances: list[tuple[float, str]] = []
        for el, _, xyz, _ in atoms:
            for shift in images:
                position = to_cartesian(tuple(xyz[n] + shift[n] for n in range(3)), lattice)
                distance = math.dist(center, position)
                if distance > 0.5:
                    distances.append((distance, el))
        if not distances:
            continue
        shortest = min(d for d, _ in distances)
        shell = sorted((d, el) for d, el in distances if d <= shortest * SHELL_TOLERANCE)
        structure.coordination.append({
            "site": site["label"],
            "element": site["element"],
            "wyckoff": site["wyckoff"] or None,
            "multiplicity": site["multiplicity"],
            "coordination_number": len(shell),
            "neighbors": dict(Counter(el for _, el in shell)),
            "distance_range_angstrom": [round(shell[0][0], 3), round(shell[-1][0], 3)],
        })


# --------------------------------------------------------------------------------------------
# Clef request and verdict
# --------------------------------------------------------------------------------------------


def build_state(structure: Structure, name: str, hint: str | None, raw_cif: str | None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "task": f"Decide whether this crystal structure belongs to the {name} structure type.",
        "target_structure_type": {"name": name, "definition": hint or "not provided; use general crystallographic knowledge"},
        "parsed_structure": {
            "formula_sum": structure.formula_sum,
            "reduced_formula_from_sites": structure.reduced_formula,
            "formula_units_per_cell_z": structure.z,
            "space_group": structure.space_group,
            "space_group_number": structure.space_group_number,
            "crystal_system": structure.crystal_system,
            "cell": structure.cell,
            "cell_composition": structure.cell_composition,
            "sites": [
                {k: site[k] for k in ("label", "element", "wyckoff", "xyz", "occupancy", "multiplicity") if k in site}
                for site in structure.sites
            ],
            "first_coordination_shells": structure.coordination,
            "parser_warnings": structure.warnings,
        },
    }
    if raw_cif:
        state["raw_cif"] = raw_cif
    return state


def build_questions(name: str) -> dict[str, dict[str, Any]]:
    return {
        "is_target": noul(
            f"Does this structure belong to the {name} structure type? Chemical substitution is allowed; "
            "judge the arrangement of atoms, not the elements.",
            true=f"Same structure type as {name}, including small distortions of it.",
            false=f"A different structure type from {name}.",
        ),
        "relation": choice(
            {
                "prototype": f"Ideal {name} type: same space group, Wyckoff sites, and coordination.",
                "distorted": f"A distorted or tilted derivative of {name} in a subgroup symmetry.",
                "ordered_superstructure": f"An ordered (e.g. double / layered) superstructure built from {name} units.",
                "related_family": f"Shares building units or stoichiometry with {name} but is a different type.",
                "unrelated": f"No meaningful structural relation to {name}.",
            },
            f"How is this structure related to the {name} structure type?",
        ),
        "match_level": score(
            [
                "Clearly a different structure type",
                "Same stoichiometry or motifs only",
                "Derivative of the target type",
                "Same type with a distortion",
                "Ideal match to the target type",
            ],
            f"How closely does the structure match {name}?",
        ),
        "stoichiometry_ok": noul(f"Is the reduced formula compatible with the stoichiometry of the {name} type?"),
        "symmetry_ok": noul(f"Is the space group that of the {name} type or one of its subgroups?"),
        "coordination_ok": noul(f"Do the coordination numbers and neighbor types match the {name} type?"),
        "crystal_system": choice(CRYSTAL_SYSTEMS, "Which crystal system does this structure have?"),
        "data_quality": choice(
            {
                "complete": "Cell, symmetry, and all sites are present and ordered.",
                "disordered": "Partial occupancy or mixed sites make the decision less certain.",
                "insufficient": "Key information is missing; the decision is unreliable.",
            },
            "How reliable is the input for this decision?",
        ),
    }


def verdict(structure: Structure, answers: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Combine the answers into a decision and flag disagreements with the parsed facts."""
    p_target = answers["is_target"]["noul"]
    checks = {key: answers[key]["noul"] for key in ("stoichiometry_ok", "symmetry_ok", "coordination_ok")}
    decision = p_target >= threshold
    notes: list[str] = []
    failed = [key for key, value in checks.items() if value < 0.5]
    if decision and failed:
        notes.append(f"positive verdict but sub-checks disagree: {', '.join(failed)}")
    if not decision and not failed:
        notes.append("negative verdict although all sub-checks pass")
    relation = answers["relation"]["choice"]
    if decision and relation in ("related_family", "unrelated"):
        notes.append(f"positive verdict but relation is '{relation}'")
    predicted_system = answers["crystal_system"]["choice"]
    if structure.crystal_system and predicted_system != structure.crystal_system:
        notes.append(f"model read crystal system as {predicted_system}, CIF says {structure.crystal_system}")
    if answers["data_quality"]["choice"] != "complete" or structure.warnings:
        notes.append("input quality: " + ", ".join([answers["data_quality"]["choice"], *structure.warnings]))
    return {
        "decision": decision,
        "confidence": round(p_target if decision else 1 - p_target, 4),
        "relation": relation,
        "match_level": answers["match_level"]["score"],
        "checks": checks,
        "reliable": not notes,
        "notes": notes,
    }


def print_report(structure: Structure, name: str, result: dict[str, Any], latency_ms: float | None) -> None:
    mark = "YES" if result["decision"] else "NO"
    print(f"\n{structure.path}")
    print(f"  {structure.reduced_formula or structure.formula_sum}  {structure.space_group} (#{structure.space_group_number})  Z={structure.z}")
    for shell in structure.coordination:
        neighbors = " ".join(f"{el}x{n}" for el, n in shell["neighbors"].items())
        low, high = shell["distance_range_angstrom"]
        print(f"    {shell['site']:<6} CN{shell['coordination_number']:<2} {neighbors:<14} {low:.3f}-{high:.3f} Å")
    print(f"  {name}? {mark}  (confidence {result['confidence']:.2f}, relation={result['relation']}, match {result['match_level']:.2f}/4)")
    print("  checks: " + "  ".join(f"{key.removesuffix('_ok')}={value:.2f}" for key, value in result["checks"].items()))
    for note in result["notes"]:
        print(f"  ! {note}")
    if latency_ms is not None:
        print(f"  ({latency_ms:.0f} ms)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cif", nargs="+", type=Path, help="CIF file(s)")
    parser.add_argument("--structure", "-s", required=True, help="structure type name, e.g. rocksalt, perovskite, spinel")
    parser.add_argument("--hint", help="definition of the structure type (built-in for common names)")
    parser.add_argument("--threshold", type=float, default=0.5, help="probability needed for a YES (default 0.5)")
    parser.add_argument("--no-raw", action="store_true", help="send only parsed features, not the raw CIF")
    parser.add_argument("--max-raw-chars", type=int, default=12000, help="skip the raw CIF above this size")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a report")
    args = parser.parse_args()

    key = ALIASES.get(args.structure.lower(), args.structure.lower())
    hint = args.hint or STRUCTURE_HINTS.get(key)
    client = ClefClient(args.url)
    questions = build_questions(args.structure)
    results = []
    for path in args.cif:
        structure = load_structure(path)
        raw = None if args.no_raw else path.read_text(errors="replace")
        if raw and len(raw) > args.max_raw_chars:
            structure.warnings.append(f"raw CIF ({len(raw)} chars) not sent")
            raw = None
        try:
            response = client.systemone(build_state(structure, args.structure, hint, raw), questions)
        except ClefError as error:
            print(f"{path}: {error}", file=sys.stderr)
            return 1
        result = verdict(structure, response["answers"], args.threshold)
        if args.json:
            results.append({"cif": str(path), "structure": args.structure, **result, "answers": response["answers"]})
        else:
            print_report(structure, args.structure, result, response["usage"].get("latency_ms"))
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
