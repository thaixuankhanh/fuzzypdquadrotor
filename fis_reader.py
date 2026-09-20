"""
Minimal reader/evaluator for MATLAB Fuzzy Logic Toolbox ``.fis`` files
(FIS ASCII format, Version 2.0, Mamdani-type systems).

This reimplements just enough of MATLAB's ``readfis`` / ``evalfis`` to
reproduce the behaviour used by FuzzyPDController.m:

    fis1 = readfis('delKp.fis');
    delKpx = evalfis(fis1, [ex, ex_dot]);

Only the pieces exercised by delKp.fis / delKd.fis are implemented:
  - triangular membership functions ('trimf')
  - AndMethod = 'min', OrMethod = 'max'
  - ImpMethod = 'min', AggMethod = 'max'
  - DefuzzMethod = 'centroid'

The evaluator is written generically (rule antecedent index 0 = "don't
care", negative index = negated MF) so it will also work with other
simple two-input/one-output Mamdani FIS files using the functions
above.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MembershipFunction:
    name: str
    mftype: str
    params: list


@dataclass
class Variable:
    name: str
    range: tuple
    mfs: list = field(default_factory=list)


@dataclass
class Rule:
    antecedents: list  # signed 1-based MF indices per input, 0 = not used
    consequents: list  # signed 1-based MF indices per output, 0 = not used
    weight: float
    connective: int  # 1 = AND (min), 2 = OR (max)


@dataclass
class FIS:
    name: str
    and_method: str
    or_method: str
    imp_method: str
    agg_method: str
    defuzz_method: str
    inputs: list
    outputs: list
    rules: list


def _membership(mf: MembershipFunction, x: np.ndarray) -> np.ndarray:
    if mf.mftype == "trimf":
        a, b, c = mf.params
        x = np.asarray(x, dtype=float)
        y = np.zeros_like(x)
        if b > a:
            left = (x > a) & (x <= b)
            y[left] = (x[left] - a) / (b - a)
        if c > b:
            right = (x > b) & (x < c)
            y[right] = (c - x[right]) / (c - b)
        y[x == b] = 1.0
        return y
    if mf.mftype == "trapmf":
        a, b, c, d = mf.params
        x = np.asarray(x, dtype=float)
        y = np.zeros_like(x)
        if b > a:
            left = (x > a) & (x < b)
            y[left] = (x[left] - a) / (b - a)
        flat = (x >= b) & (x <= c)
        y[flat] = 1.0
        if d > c:
            right = (x > c) & (x < d)
            y[right] = (d - x[right]) / (d - c)
        return y
    if mf.mftype == "gaussmf":
        sigma, c = mf.params
        return np.exp(-0.5 * ((np.asarray(x, dtype=float) - c) / sigma) ** 2)
    raise NotImplementedError(f"Membership type '{mf.mftype}' not supported")


_SECTION_RE = re.compile(r"^\[(?P<name>.+)\]$")
_KV_RE = re.compile(r"^(?P<key>\w+)=(?P<value>.*)$")
_MF_RE = re.compile(
    r"^MF\d+='(?P<name>[^']*)':'(?P<type>\w+)',\[(?P<params>[^\]]*)\]$"
)
_RULE_RE = re.compile(
    r"^(?P<ants>[-\d\s]+),\s*(?P<cons>[-\d\s]+)\s*\((?P<weight>[-\d.]+)\)\s*:\s*(?P<conn>\d+)$"
)


def read_fis(path: str) -> FIS:
    """Parse a MATLAB ``.fis`` (Mamdani) file."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f]

    section = None
    sys_kv = {}
    inputs: dict[int, Variable] = {}
    outputs: dict[int, Variable] = {}
    rules: list[Rule] = []
    cur_var: Variable | None = None

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("%"):
            continue

        m = _SECTION_RE.match(line)
        if m:
            section = m.group("name")
            if section.startswith("Input"):
                idx = int(section.replace("Input", ""))
                cur_var = Variable(name=f"input{idx}", range=(0.0, 0.0))
                inputs[idx] = cur_var
            elif section.startswith("Output"):
                idx = int(section.replace("Output", ""))
                cur_var = Variable(name=f"output{idx}", range=(0.0, 0.0))
                outputs[idx] = cur_var
            else:
                cur_var = None
            continue

        mf_match = _MF_RE.match(line)
        if mf_match and cur_var is not None:
            params = [float(p) for p in mf_match.group("params").split()]
            cur_var.mfs.append(
                MembershipFunction(mf_match.group("name"), mf_match.group("type"), params)
            )
            continue

        kv = _KV_RE.match(line)
        if kv and section == "System":
            sys_kv[kv.group("key")] = kv.group("value").strip("'")
            continue

        if kv and cur_var is not None and kv.group("key") in ("Name", "Range", "NumMFs"):
            if kv.group("key") == "Range":
                lo, hi = [float(v) for v in kv.group("value").strip("[]").split()]
                cur_var.range = (lo, hi)
            continue

        if section == "Rules":
            rm = _RULE_RE.match(line)
            if rm:
                ants = [int(v) for v in rm.group("ants").split()]
                cons = [int(v) for v in rm.group("cons").split()]
                rules.append(
                    Rule(
                        antecedents=ants,
                        consequents=cons,
                        weight=float(rm.group("weight")),
                        connective=int(rm.group("conn")),
                    )
                )

    return FIS(
        name=sys_kv.get("Name", ""),
        and_method=sys_kv.get("AndMethod", "min"),
        or_method=sys_kv.get("OrMethod", "max"),
        imp_method=sys_kv.get("ImpMethod", "min"),
        agg_method=sys_kv.get("AggMethod", "max"),
        defuzz_method=sys_kv.get("DefuzzMethod", "centroid"),
        inputs=[inputs[k] for k in sorted(inputs)],
        outputs=[outputs[k] for k in sorted(outputs)],
        rules=rules,
    )


def _mf_degree(var: Variable, mf_index: int, x: float) -> float:
    """Signed 1-based MF index -> membership degree of crisp value x."""
    if mf_index == 0:
        return None  # variable not used in this rule
    idx = abs(mf_index) - 1
    mf = var.mfs[idx]
    deg = float(_membership(mf, np.array([x]))[0])
    if mf_index < 0:
        deg = 1.0 - deg
    return deg


def evalfis(fis: FIS, inputs: list, resolution: int = 101) -> np.ndarray:
    """Evaluate a Mamdani FIS for one crisp input vector.

    Mirrors MATLAB's ``evalfis(fis, inputs)`` for the subset of options
    used by delKp.fis / delKd.fis (min AND, min implication, max
    aggregation, centroid defuzzification). ``resolution`` is the number
    of points used to discretize each output universe, matching
    MATLAB's default of 101 points.
    """
    n_out = len(fis.outputs)
    outputs = np.zeros(n_out)

    for oi, out_var in enumerate(fis.outputs):
        lo, hi = out_var.range
        x = np.linspace(lo, hi, resolution)
        aggregated = np.zeros_like(x)

        for rule in fis.rules:
            degrees = []
            for vi, ant_idx in enumerate(rule.antecedents):
                if ant_idx == 0:
                    continue
                degrees.append(_mf_degree(fis.inputs[vi], ant_idx, inputs[vi]))
            if not degrees:
                continue
            if rule.connective == 2:  # OR
                firing = max(degrees) if fis.or_method == "max" else min(degrees)
            else:  # AND (default)
                firing = min(degrees) if fis.and_method == "min" else max(degrees)
            firing *= rule.weight
            if firing <= 0:
                continue

            cons_idx = rule.consequents[oi]
            if cons_idx == 0:
                continue
            mf = out_var.mfs[abs(cons_idx) - 1]
            mf_y = _membership(mf, x)
            if cons_idx < 0:
                mf_y = 1.0 - mf_y

            # Implication (min -> clip)
            implied = np.minimum(mf_y, firing) if fis.imp_method == "min" else mf_y * firing

            # Aggregation (max)
            aggregated = np.maximum(aggregated, implied) if fis.agg_method == "max" else aggregated + implied

        area = np.trapz(aggregated, x)
        if area <= 1e-12:
            outputs[oi] = 0.5 * (lo + hi)
        else:
            outputs[oi] = np.trapz(aggregated * x, x) / area

    return outputs
