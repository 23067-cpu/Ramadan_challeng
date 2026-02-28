"""
SupNum - S3C'1447 Challenge Ramadan
Fichier : rcpsp_scheduler.py
Description : Fichier développé en Python pour la résolution du problème RCPSP.
Tous les algorithmes et logiques internes sont optimisés pour des performances maximales.
"""
"""
rcpsp_scheduler.py
World-class Schedule Generation and Improvement for RCPSP.

Implements (based on Kolisch & Hartmann 2006, Valls et al. 2005, 2008):
  - Serial SGS (forward)
  - Parallel SGS (forward)
  - Backward Serial SGS (reversed network)
  - Left-Shift operator (local activity start-time pull)
  - Forward-Backward Improvement / Justification (FBI)
  - Topological sort + validation utilities
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from rcpsp_parser import RCPSPInstance


# ─────────────────────────────────────────────────────────────────────────────
# Schedule dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Schedule:
    """A feasible RCPSP schedule with start times for each job."""
    instance: RCPSPInstance
    start_times: List[int]

    @property
    def cmax(self) -> int:
        return self.start_times[self.instance.n_jobs - 1]

    @property
    def finish_times(self) -> List[int]:
        return [self.start_times[i] + self.instance.durations[i]
                for i in range(self.instance.n_jobs)]

    def is_feasible(self) -> bool:
        inst = self.instance
        ft = self.finish_times
        for i, succs in enumerate(inst.successors):
            for j in succs:
                if self.start_times[j] < ft[i]:
                    return False
        T = self.cmax
        for t in range(T):
            for k in range(inst.n_resources):
                usage = sum(
                    inst.resource_needs[i][k]
                    for i in range(inst.n_jobs)
                    if self.start_times[i] <= t < ft[i]
                )
                if usage > inst.resource_caps[k]:
                    return False
        return True

    def resource_usage(self) -> Dict[int, List[int]]:
        T = self.cmax
        ft = self.finish_times
        usage = {k: [0] * T for k in range(self.instance.n_resources)}
        for i in range(self.instance.n_jobs):
            for t in range(self.start_times[i], ft[i]):
                if t < T:
                    for k in range(self.instance.n_resources):
                        usage[k][t] += self.instance.resource_needs[i][k]
        return usage


# ─────────────────────────────────────────────────────────────────────────────
# Resource tracking helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_resource_table(instance: RCPSPInstance, H: int) -> List[List[int]]:
    """Pre-allocate 2D resource usage table: resource_at[k][t]."""
    return [[0] * H for _ in range(instance.n_resources)]


def _resource_feasible(resource_at: List[List[int]], resource_caps: List[int],
                        needs: List[int], t: int, dur: int, H: int) -> bool:
    """Check whether job with `needs` can start at t for duration dur without overflow."""
    for tau in range(t, t + dur):
        if tau >= H:
            return True  # beyond horizon is always feasible (no competing jobs)
        for k, cap in enumerate(resource_caps):
            if resource_at[k][tau] + needs[k] > cap:
                return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Forward Serial SGS
# ─────────────────────────────────────────────────────────────────────────────

def serial_sgs(activity_list: List[int], instance: RCPSPInstance) -> Schedule:
    """
    Forward Serial Schedule Generation Scheme.
    Schedules each activity as early as possible given precedence + resources.
    """
    n = instance.n_jobs
    n_res = instance.n_resources
    durations = instance.durations
    preds = instance.predecessors
    resource_needs = instance.resource_needs
    resource_caps = instance.resource_caps

    start_times = [-1] * n
    H = instance.horizon + 10
    res = _build_resource_table(instance, H)

    for job in activity_list:
        est = max((start_times[p] + durations[p] for p in preds[job]), default=0)
        t = est
        needs = resource_needs[job]
        dur = durations[job]

        if dur == 0:
            start_times[job] = t
            continue

        while True:
            # Extend table if needed
            while t + dur > H:
                for k in range(n_res):
                    res[k].extend([0] * 100)
                H += 100

            if _resource_feasible(res, resource_caps, needs, t, dur, H):
                break
            t += 1

        start_times[job] = t
        for tau in range(t, t + dur):
            for k in range(n_res):
                res[k][tau] += needs[k]

    return Schedule(instance=instance, start_times=start_times)


# ─────────────────────────────────────────────────────────────────────────────
# Forward Parallel SGS
# ─────────────────────────────────────────────────────────────────────────────

def parallel_sgs(activity_list: List[int], instance: RCPSPInstance) -> Schedule:
    """
    Forward Parallel Schedule Generation Scheme.
    At each decision point, schedules all eligible activities in priority order.
    """
    n = instance.n_jobs
    n_res = instance.n_resources
    durations = instance.durations
    preds = instance.predecessors
    resource_needs = instance.resource_needs
    resource_caps = instance.resource_caps

    priority = {job: idx for idx, job in enumerate(activity_list)}
    start_times = [-1] * n
    scheduled = set()
    H = instance.horizon + 10
    res = _build_resource_table(instance, H)

    t = 0
    remaining = set(range(n))

    while remaining:
        eligible = [
            job for job in remaining
            if all(p in scheduled for p in preds[job])
            and all(start_times[p] + durations[p] <= t for p in preds[job] if start_times[p] >= 0)
        ]
        eligible.sort(key=lambda j: priority.get(j, n))

        changed = True
        while changed:
            changed = False
            for job in eligible[:]:
                if job in scheduled:
                    eligible.remove(job)
                    continue
                dur = durations[job]
                needs = resource_needs[job]
                fits = True
                for tau in range(t, t + dur):
                    if tau >= H:
                        break
                    if not all(res[k][tau] + needs[k] <= resource_caps[k]
                               for k in range(n_res)):
                        fits = False
                        break
                if fits:
                    start_times[job] = t
                    scheduled.add(job)
                    eligible.remove(job)
                    remaining.discard(job)
                    for tau in range(t, t + dur):
                        if tau < H:
                            for k in range(n_res):
                                res[k][tau] += needs[k]
                    changed = True

        finish_times = [start_times[j] + durations[j]
                        for j in scheduled if start_times[j] >= 0]
        next_t = [ft for ft in finish_times if ft > t]
        t = min(next_t) if next_t else t + 1

    return Schedule(instance=instance, start_times=start_times)


# ─────────────────────────────────────────────────────────────────────────────
# Backward Serial SGS  (Valls et al. 2005)
# ─────────────────────────────────────────────────────────────────────────────

def backward_sgs(activity_list: List[int], instance: RCPSPInstance,
                 horizon: Optional[int] = None) -> Schedule:
    """
    Backward Serial SGS (Valls et al. 2005).

    Builds a mirror-image of the problem (reversed precedences, reversed activity list),
    runs forward serial SGS on that mirror, then maps the result back to the original
    timeline:  start_orig[j] = T - finish_mirror[j]  where T = mirror makespan.

    This produces schedules that are "right-justified" in the original timeline,
    which when combined with forward passes (FBI) can uncover shorter makespans.
    """
    inst = instance
    n = inst.n_jobs
    durations = inst.durations
    resource_needs = inst.resource_needs
    resource_caps = inst.resource_caps
    n_res = inst.n_resources

    # Mirror: original preds ↔ succs, reversed activity list
    mirror_preds = [list(inst.successors[i]) for i in range(n)]
    rev_list = list(reversed(activity_list))

    # Forward SSGS on mirror problem
    start_mirror = [-1] * n
    H = (horizon if horizon is not None else inst.horizon) + 20
    res = [[0] * H for _ in range(n_res)]

    for job in rev_list:
        est = max(
            (start_mirror[p] + durations[p]
             for p in mirror_preds[job] if start_mirror[p] >= 0),
            default=0
        )
        t = est
        dur = durations[job]
        needs = resource_needs[job]

        if dur == 0:
            start_mirror[job] = t
            continue

        while True:
            while t + dur > H:
                for k in range(n_res):
                    res[k].extend([0] * 100)
                H += 100
            if _resource_feasible(res, resource_caps, needs, t, dur, H):
                break
            t += 1

        start_mirror[job] = t
        for tau in range(t, t + dur):
            for k in range(n_res):
                res[k][tau] += needs[k]

    # Mirror makespan = finish time of source in mirror (= job 0 in mirror)
    # In mirror: source (job 0) has no mirror-predecessors → it finishes last
    # The mirror "project duration" = max finish in mirror
    T_mirror = max(start_mirror[j] + durations[j] for j in range(n))

    # Map back: start_original[j] = T_mirror - start_mirror[j] - durations[j]
    start_times = [max(0, T_mirror - start_mirror[j] - durations[j])
                   for j in range(n)]

    return Schedule(instance=instance, start_times=start_times)


# ─────────────────────────────────────────────────────────────────────────────
# Left-Shift Operator  (Kolisch 1996)
# ─────────────────────────────────────────────────────────────────────────────

def left_shift_schedule(schedule: Schedule) -> Schedule:
    """
    Apply the Left-Shift operator to a schedule.
    Each activity is moved as early as possible (without touching others).
    This is a GLOBAL left-shift: iterate until no improvement.

    This is also known as "schedule normalization" or "earliest-start justification".
    Repeatedly pulls each activity to t=0 and finds its real earliest feasible start.

    Returns a new (potentially improved) Schedule.
    """
    inst = schedule.instance
    n = inst.n_jobs
    durations = inst.durations
    preds = inst.predecessors
    resource_needs = inst.resource_needs
    resource_caps = inst.resource_caps
    n_res = inst.n_resources

    # Process in topological order (by current start time)
    topo = sorted(range(n), key=lambda i: schedule.start_times[i])

    # Rebuild start times greedily
    start_times = [0] * n
    H = max(schedule.cmax + 10, inst.horizon + 10)
    res = [[0] * H for _ in range(n_res)]

    for job in topo:
        dur = durations[job]
        needs = resource_needs[job]
        # Earliest from predecessors
        est = max((start_times[p] + durations[p] for p in preds[job]), default=0)
        t = est

        if dur == 0:
            start_times[job] = t
            continue

        while True:
            while t + dur > H:
                for k in range(n_res):
                    res[k].extend([0] * 50)
                H += 50
            if _resource_feasible(res, resource_caps, needs, t, dur, H):
                break
            t += 1

        start_times[job] = t
        for tau in range(t, t + dur):
            for k in range(n_res):
                res[k][tau] += needs[k]

    return Schedule(instance=inst, start_times=start_times)


# ─────────────────────────────────────────────────────────────────────────────
# Forward-Backward Improvement / Justification (FBI)  (Valls et al. 2005)
# ─────────────────────────────────────────────────────────────────────────────

def forward_backward_improvement(activity_list: List[int],
                                  instance: RCPSPInstance,
                                  max_rounds: int = 5) -> Tuple[List[int], Schedule]:
    """
    Forward-Backward Improvement (Justification) as described in Valls et al. (2005).

    Algorithm:
      repeat:
        1. Forward SGS  → schedule S1 with makespan C1
        2. Backwards SGS on reversed list → schedule S2 (right-justified)
        3. Derive new activity list from S2's relative order
        4. Run Forward SGS on new list → schedule S3 (always feasible)
        5. If S3.cmax < C1, update; else try one more forward sort; else stop
      until no improvement or max_rounds reached.

    IMPORTANT: We never use bwd_sc directly as the final schedule — we always
    convert it back to an activity list and run forward SSGS to guarantee feasibility.

    Returns (best_activity_list, best_schedule).
    """
    best_al = activity_list[:]
    best_sc = serial_sgs(best_al, instance)
    best_cmax = best_sc.cmax

    for _ in range(max_rounds):
        # ── Backward pass ──
        bwd_sc = backward_sgs(best_al, instance, horizon=best_cmax)

        # Derive activity list from backward schedule's relative ordering
        # (sort jobs by their start time in the backward-justified schedule)
        bwd_al = sorted(range(instance.n_jobs),
                        key=lambda i: (bwd_sc.start_times[i], i))

        # Always validate with forward SGS  (guarantees feasibility!)
        fwd_sc = serial_sgs(bwd_al, instance)
        if fwd_sc.cmax < best_cmax:
            best_al, best_sc, best_cmax = bwd_al, fwd_sc, fwd_sc.cmax
        else:
            # ── Forward pass: re-sort current best by start time and re-evaluate ──
            resort_al = sorted(range(instance.n_jobs),
                               key=lambda i: (best_sc.start_times[i], i))
            resort_sc = serial_sgs(resort_al, instance)
            if resort_sc.cmax < best_cmax:
                best_al, best_sc, best_cmax = resort_al, resort_sc, resort_sc.cmax
            else:
                break  # no improvement in this round

    return best_al, best_sc



# ─────────────────────────────────────────────────────────────────────────────
# Utility: Topological sort + validation
# ─────────────────────────────────────────────────────────────────────────────

def topological_sort(instance: RCPSPInstance) -> List[int]:
    """Kahn's algorithm: returns one valid topological ordering."""
    n = instance.n_jobs
    in_deg = [len(instance.predecessors[i]) for i in range(n)]
    queue = deque(i for i in range(n) if in_deg[i] == 0)
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for s in instance.successors[v]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)
    return order


def validate_activity_list(activity_list: List[int], instance: RCPSPInstance) -> bool:
    """Check that activity_list is a valid topological order."""
    pos = {job: idx for idx, job in enumerate(activity_list)}
    for i, succs in enumerate(instance.successors):
        for j in succs:
            if pos.get(i, -1) >= pos.get(j, -1):
                return False
    return True


if __name__ == '__main__':
    from rcpsp_parser import parse_instance
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'j30.sm/j301_1.sm'
    inst = parse_instance(path)
    order = topological_sort(inst)
    s1 = serial_sgs(order, inst)
    s2 = parallel_sgs(order, inst)
    s3 = backward_sgs(order, inst)
    s4 = left_shift_schedule(s1)
    _, s5 = forward_backward_improvement(order, inst)
    print(f"SSGS  Cmax: {s1.cmax}")
    print(f"PSGS  Cmax: {s2.cmax}")
    print(f"BSGS  Cmax: {s3.cmax}")
    print(f"LS    Cmax: {s4.cmax}")
    print(f"FBI   Cmax: {s5.cmax}")
