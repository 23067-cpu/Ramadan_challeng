"""
SupNum - S3C'1447 Challenge Ramadan
Fichier : rcpsp_bounds.py
Description : Fichier développé en Python pour la résolution du problème RCPSP.
Tous les algorithmes et logiques internes sont optimisés pour des performances maximales.
"""
"""
rcpsp_bounds.py
Lower Bound computation for RCPSP instances.

Computes:
  LB1 - Critical Path Method (CPM) ignoring resources
  LB2 - Resource-based bound using total work / capacity
  LB   = max(LB1, LB2)
"""

from __future__ import annotations
import math
from rcpsp_parser import RCPSPInstance


def lb_critical_path(instance: RCPSPInstance) -> int:
    """
    LB1: Critical path length (CPM) ignoring resource constraints.
    This is the longest path from source to sink in the precedence graph
    weighted by activity durations.
    """
    return instance.critical_path_length()


def lb_resource_based(instance: RCPSPInstance) -> int:
    """
    LB2: Resource-based lower bound.
    
    For each resource k, the total work required divided by capacity
    gives a minimum project duration just due to that resource.
    LB2 = max over all k of: ceil( sum_i(duration_i * demand_ik) / capacity_k )
    
    Also supports a refined version that excludes jobs not needing resource k.
    """
    n = instance.n_jobs
    n_res = instance.n_resources
    durations = instance.durations
    resource_needs = instance.resource_needs
    resource_caps = instance.resource_caps

    lb2 = 0
    for k in range(n_res):
        if resource_caps[k] == 0:
            continue
        total_work = sum(durations[i] * resource_needs[i][k] for i in range(n))
        bound_k = math.ceil(total_work / resource_caps[k])
        lb2 = max(lb2, bound_k)

    return lb2


def lb_resource_window(instance: RCPSPInstance) -> int:
    """
    LB3: Improved resource-based bound using earliest/latest finish times.
    
    For each resource k and each time window [a, b], the work content
    that MUST be done in [a, b] gives a refined bound.
    This is a simplified version of the Demeulemeester & Herroelen bound.
    """
    n = instance.n_jobs
    n_res = instance.n_resources
    durations = instance.durations
    resource_needs = instance.resource_needs
    resource_caps = instance.resource_caps

    est_vals = instance.est()
    lst_vals = instance.lst()
    cp = instance.critical_path_length()

    lb3 = cp
    for k in range(n_res):
        if resource_caps[k] == 0:
            continue
        # Try windows [0, T] for T in range of plausible makespan values
        for T in range(cp, cp + 1):  # simplified: only check at cp
            # Work that can be postponed AFTER T
            work_after = 0
            for i in range(n):
                if resource_needs[i][k] == 0:
                    continue
                # How much of job i's work can be shifted after T?
                earliest_end = est_vals[i] + durations[i]
                latest_start = lst_vals[i]
                # Minimum work in [0, T]
                work_in = max(0, min(T, earliest_end) - est_vals[i]) * resource_needs[i][k]
                work_after += max(0, durations[i] * resource_needs[i][k] - work_in)

            # Bound: T + ceil(work_after / capacity_k)
            if resource_caps[k] > 0:
                bound = T + math.ceil(work_after / resource_caps[k])
                lb3 = max(lb3, bound)

    return lb3


def compute_lower_bound(instance: RCPSPInstance) -> dict:
    """
    Compute all lower bounds and return the best (maximum).
    
    Returns a dict with:
        'LB1': critical path
        'LB2': resource-based
        'LB3': resource-window
        'LB' : max of all
    """
    lb1 = lb_critical_path(instance)
    lb2 = lb_resource_based(instance)
    lb3 = lb_resource_window(instance)
    lb = max(lb1, lb2, lb3)

    return {
        'LB1': lb1,
        'LB2': lb2,
        'LB3': lb3,
        'LB': lb,
    }


if __name__ == '__main__':
    from rcpsp_parser import parse_instance
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'j30.sm/j301_1.sm'
    inst = parse_instance(path)
    bounds = compute_lower_bound(inst)
    print(f"Instance : {inst.name}")
    for k, v in bounds.items():
        print(f"  {k}: {v}")
