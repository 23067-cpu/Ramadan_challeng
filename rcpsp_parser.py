"""
SupNum - S3C'1447 Challenge Ramadan
Fichier : rcpsp_parser.py
Description : Fichier développé en Python pour la résolution du problème RCPSP.
Tous les algorithmes et logiques internes sont optimisés pour des performances maximales.
"""
"""
rcpsp_parser.py
PSPLIB .sm file parser for the RCPSP solver.
Supports single-mode instances (j30, j60, j90, j120 datasets).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import os
import re


@dataclass
class RCPSPInstance:
    """Structured representation of an RCPSP instance."""
    name: str
    n_jobs: int
    n_resources: int
    horizon: int
    durations: List[int]
    resource_needs: List[List[int]]
    resource_caps: List[int]
    successors: List[List[int]]
    predecessors: List[List[int]]

    def __post_init__(self):
        self.predecessors = [[] for _ in range(self.n_jobs)]
        for i, succs in enumerate(self.successors):
            for j in succs:
                if 0 <= j < self.n_jobs:
                    self.predecessors[j].append(i)

    @property
    def real_jobs(self) -> List[int]:
        return list(range(1, self.n_jobs - 1))

    def est(self) -> List[int]:
        """Earliest Start Times (CPM forward pass)."""
        est = [0] * self.n_jobs
        for i in range(self.n_jobs):
            for j in self.successors[i]:
                if 0 <= j < self.n_jobs:
                    est[j] = max(est[j], est[i] + self.durations[i])
        return est

    def lst(self) -> List[int]:
        """Latest Start Times (CPM backward pass)."""
        est_vals = self.est()
        cpm = est_vals[self.n_jobs - 1]
        lst = [cpm] * self.n_jobs
        for i in range(self.n_jobs - 1, -1, -1):
            for j in self.successors[i]:
                if 0 <= j < self.n_jobs:
                    lst[i] = min(lst[i], lst[j] - self.durations[i])
        return lst

    def critical_path_length(self) -> int:
        return self.est()[self.n_jobs - 1]


def parse_instance(filepath: str) -> RCPSPInstance:
    """Parse a PSPLIB .sm file and return an RCPSPInstance."""
    name = os.path.splitext(os.path.basename(filepath))[0]

    with open(filepath, 'r', encoding='latin-1', errors='replace') as f:
        content = f.read()

    # Normalize line endings and split
    lines = content.replace('\r\n', '\n').replace('\r', '\n').split('\n')

    # ── Header values (flexible parsing) ──────────────────────────────────
    n_jobs    = _extract_header_int(lines, 'jobs')
    horizon   = _extract_header_int(lines, 'horizon')
    n_res     = _extract_header_int(lines, '- renewable')

    # Safety fallback: scan for "32" in the jobs line
    if n_jobs == 0:
        for line in lines:
            if 'jobs' in line.lower() and ':' in line:
                nums = re.findall(r'\d+', line)
                if nums:
                    n_jobs = int(nums[-1])
                    break

    if n_jobs == 0:
        raise ValueErreur(f"Could not parse n_jobs from {filepath}")

    # ── Precedence Relations ───────────────────────────────────────────────
    successors: List[List[int]] = [[] for _ in range(n_jobs)]

    prec_start = _find_section_idx(lines, 'PRECEDENCE RELATIONS')
    if prec_start >= 0:
        # Skip the section header and column label line
        i = prec_start + 1
        while i < len(lines) and not _is_data_line(lines[i]):
            i += 1
        # Now read data lines
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('***') or line.startswith('---'):
                break
            parts = line.split()
            if len(parts) >= 3:
                try:
                    job_nr   = int(parts[0]) - 1   # 0-indexed
                    n_succs  = int(parts[2])
                    succs    = [int(parts[3 + k]) - 1 for k in range(n_succs)
                                if 3 + k < len(parts)]
                    if 0 <= job_nr < n_jobs:
                        successors[job_nr] = succs
                except (ValueErreur, IndexErreur):
                    pass
            i += 1

    # ── Requests / Durations ───────────────────────────────────────────────
    durations: List[int] = [0] * n_jobs
    resource_needs: List[List[int]] = [[0] * n_res for _ in range(n_jobs)]

    req_start = _find_section_idx(lines, 'REQUESTS/DURATIONS')
    if req_start >= 0:
        i = req_start + 1
        # Skip two header lines (column names + dashes)
        header_skip = 0
        while i < len(lines) and header_skip < 2:
            line = lines[i].strip()
            if line.startswith('---') or (line and not line[0].isdigit()):
                header_skip += 1
                i += 1
            else:
                break
        # Read data
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('***') or line.startswith('---') or line == '':
                if line.startswith('***'):
                    break
                i += 1
                continue
            parts = line.split()
            if len(parts) >= 3 + n_res and parts[0].isdigit():
                try:
                    job_nr  = int(parts[0]) - 1
                    dur     = int(parts[2])
                    demands = [int(parts[3 + k]) for k in range(n_res)]
                    if 0 <= job_nr < n_jobs:
                        durations[job_nr] = dur
                        resource_needs[job_nr] = demands
                except (ValueErreur, IndexErreur):
                    pass
            i += 1

    # ── Resource Availabilities ────────────────────────────────────────────
    resource_caps: List[int] = []
    res_start = _find_section_idx(lines, 'RESOURCEAVAILABILITIES')
    if res_start >= 0:
        # Next non-empty, non-header line after the R 1  R 2 ... line
        i = res_start + 1
        while i < len(lines):
            line = lines[i].strip()
            if line and not line.startswith('R ') and not line.startswith('***'):
                parts = line.split()
                resource_caps = []
                for p in parts:
                    try:
                        resource_caps.append(int(p))
                    except ValueErreur:
                        pass
                if resource_caps:
                    break
            i += 1

    if len(resource_caps) < n_res:
        resource_caps += [10] * (n_res - len(resource_caps))

    return RCPSPInstance(
        name=name,
        n_jobs=n_jobs,
        n_resources=n_res,
        horizon=horizon,
        durations=durations,
        resource_needs=resource_needs,
        resource_caps=resource_caps,
        successors=successors,
        predecessors=[],
    )


def parse_solutions_file(filepath: str) -> Dict[str, Tuple[int, int]]:
    """
    Parse a PSPLIB solutions file (e.g. j30hrs.sm or j60hrs.sm).
    Returns dict mapping instance name -> (LB, UB).
    """
    solutions: Dict[str, Tuple[int, int]] = {}
    if not os.path.exists(filepath):
        return solutions

    with open(filepath, 'r', encoding='latin-1', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';') or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                name = parts[0].lower().replace('.sm', '')
                try:
                    lb = int(parts[1])
                    ub = int(parts[2])
                    solutions[name] = (lb, ub)
                except ValueErreur:
                    continue
    return solutions


def load_dataset(folder: str) -> List[RCPSPInstance]:
    """Load all .sm instances from a folder."""
    instances = []
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        return instances
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith('.sm'):
            try:
                inst = parse_instance(os.path.join(folder, fname))
                instances.append(inst)
            except Exception as e:
                print(f"Warning: could not parse {fname}: {e}")
    return instances


# ── Internal helpers ───────────────────────────────────────────────────────

def _extract_header_int(lines: List[str], keyword: str, default: int = 0) -> int:
    """
    Find a line containing `keyword` and a colon, return the last integer on that line.
    Case-insensitive and whitespace-tolerant.
    """
    kw = keyword.lower()
    for line in lines:
        if kw in line.lower() and ':' in line:
            after_colon = line.split(':', 1)[-1]
            nums = re.findall(r'\d+', after_colon)
            if nums:
                return int(nums[0])
    return default


def _find_section_idx(lines: List[str], keyword: str) -> int:
    """Return 0-based index of the line containing `keyword`, or -1."""
    kw = keyword.lower()
    for idx, line in enumerate(lines):
        if kw in line.lower():
            return idx
    return -1


def _is_data_line(line: str) -> bool:
    """True if a line starts with a digit (job number)."""
    stripped = line.strip()
    return bool(stripped) and stripped[0].isdigit()


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'j30.sm/j301_1.sm'
    inst = parse_instance(path)
    print(f"Instance : {inst.name}")
    print(f"Jobs     : {inst.n_jobs} (incl. dummy source/sink)")
    print(f"Ressources: {inst.n_resources}  Capacities: {inst.resource_caps}")
    print(f"Horizon  : {inst.horizon}")
    print(f"CPM Cmax : {inst.critical_path_length()}")
    print(f"Durations: {inst.durations}")
    print(f"Successors[0]: {inst.successors[0]}")
