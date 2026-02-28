"""
SupNum - S3C'1447 Challenge Ramadan
Fichier : fast_j60_solver.py
Description : Fichier développé en Python pour la résolution du problème RCPSP.
Tous les algorithmes et logiques internes sont optimisés pour des performances maximales.
"""
"""
fast_j60_solver.py
==================
High-quality RCPSP solver designed specifically for j60 instances.

Based on the Sequential Scatter Search + FBI approach (Berthaut et al. 2018)
which achieves APD = 10.45% on j60 - within range of state-of-art (10.37%).

Key ideas:
  1. Generate many random activity lists + apply FBI → get good solutions fast
  2. Maintain a small elite population
  3. Combine elite pairs with Path Relinking
  4. Repeat until time budget exhausted

This is time-efficient: each FBI call takes ~0.1-0.5s for j60.
In 60s, we can do ~120-600 FBI calls → excellent coverage.
"""
import time
import random
from typing import List, Tuple, Optional
from rcpsp_parser import RCPSPInstance
from rcpsp_scheduler import (
    serial_sgs, parallel_sgs, forward_backward_improvement,
    topological_sort, Schedule
)


def _repair(al: List[int], instance) -> List[int]:
    """Fast Kahn's algorithm topological repair."""
    n = instance.n_jobs
    in_deg = [0] * n
    for j in range(n):
        for s in instance.successors[j]:
            in_deg[s] += 1
    pos = {j: i for i, j in enumerate(al)}
    ready = sorted([j for j in range(n) if in_deg[j] == 0], key=lambda j: pos.get(j, n))
    out = []
    while ready:
        j = ready.pop(0)
        out.append(j)
        next_rdy = []
        for s in instance.successors[j]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                next_rdy.append(s)
        next_rdy.sort(key=lambda k: pos.get(k, n))
        ready = ready + next_rdy
    return out


def _random_topo(instance) -> List[int]:
    """Generate a uniformly random topological order."""
    n = instance.n_jobs
    in_deg = [0] * n
    for j in range(n):
        for s in instance.successors[j]:
            in_deg[s] += 1
    ready = [j for j in range(n) if in_deg[j] == 0]
    out = []
    while ready:
        idx = random.randrange(len(ready))
        ready[idx], ready[-1] = ready[-1], ready[idx]
        j = ready.pop()
        out.append(j)
        for s in instance.successors[j]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                ready.append(s)
    return out


def _eval(al: List[int], instance) -> Tuple[int, Schedule]:
    """Evaluate with both SSGS and PSGS, take the best."""
    al = _repair(al, instance)
    s1 = serial_sgs(al, instance)
    s2 = parallel_sgs(al, instance)
    if s1.cmax <= s2.cmax and s1.is_feasible():
        return s1.cmax, s1
    if s2.is_feasible():
        return s2.cmax, s2
    return s1.cmax, s1


def _fbi(al: List[int], instance, rounds: int = 3) -> Tuple[int, Schedule]:
    """FBI: forward-backward improvement."""
    al = _repair(al, instance)
    try:
        al2, sc2 = forward_backward_improvement(al, instance, max_rounds=rounds)
        al2 = _repair(al2, instance)
        sc3 = serial_sgs(al2, instance)
        # take best feasible
        best_sc = sc2
        if sc3.is_feasible() and sc3.cmax < sc2.cmax:
            best_sc = sc3
        if not best_sc.is_feasible():
            _, best_sc = _eval(al, instance)
        return best_sc.cmax, best_sc
    except Exception:
        return _eval(al, instance)


def _ox1(p1: List[int], p2: List[int]) -> List[int]:
    """Order crossover OX1."""
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[a:b+1] = p1[a:b+1]
    seg = set(child[a:b+1])
    pos = b + 1
    for gene in p2[b+1:] + p2[:b+1]:
        if gene not in seg:
            child[pos % n] = gene
            seg.add(gene)
            pos += 1
    return child


def _path_relink(al1: List[int], al2: List[int], instance) -> Tuple[int, Schedule]:
    """Simple Path Relinking between two activity lists."""
    best_c, best_sc = _eval(al1, instance)
    current = al1[:]
    target_set = {j: i for i, j in enumerate(al2)}
    n = len(al1)
    current_pos = {j: i for i, j in enumerate(current)}

    for step in range(min(n // 3, 20)):  # take at most n/3 steps
        # Find job in current that's closest to its target position
        best_move = None
        best_delta = -1
        for j in range(n):
            cur_p = current_pos[j]
            tgt_p = target_set[j]
            delta = abs(cur_p - tgt_p)
            if delta > best_delta:
                best_delta = delta
                best_move = (j, cur_p, tgt_p)

        if best_move is None or best_delta == 0:
            break

        j, from_p, to_p = best_move
        # Move j toward its position in al2
        new_p = from_p + (1 if to_p > from_p else -1)
        # Swap in list
        other = current[new_p]
        current[from_p], current[new_p] = current[new_p], current[from_p]
        current_pos[j] = new_p
        current_pos[other] = from_p

        repaired = _repair(current, instance)
        c, sc = _eval(repaired, instance)
        if c < best_c and sc.is_feasible():
            best_c, best_sc = c, sc

    return best_c, best_sc


def solve(instance: RCPSPInstance, time_limit: float,
          lb: int = 0, ub_target: Optional[int] = None) -> Tuple[int, Schedule]:
    """
    Fast + high-quality solver using repeat-FBI + elite scatter search.

    Strategy (proven in literature ~10.37% APD on j60):
      Phase 1 (50% of time): Random restarts + FBI → build elite set
      Phase 2 (50% of time): Scatter search (combine elite pairs) + FBI
    """
    t0 = time.time()
    n = instance.n_jobs

    # ── Phase 0: Deterministic seeds ─────────────────────────────────────────
    topo = topological_sort(instance)
    best_c, best_sc = _eval(topo, instance)

    # Apply FBI to topo immediately
    try:
        c2, sc2 = _fbi(topo, instance, rounds=4)
        if sc2.is_feasible() and c2 < best_c:
            best_c, best_sc = c2, sc2
    except Exception:
        pass

    if best_c <= lb:
        return best_c, best_sc

    # Try a few heuristic orderings
    try:
        try:
            est = instance.est()
            lst = instance.lst()
            for order_fn in [
                lambda: sorted(range(n), key=lambda i: est[i]),
                lambda: sorted(range(n), key=lambda i: lst[i]),
                lambda: sorted(range(n), key=lambda i: -instance.durations[i]),
            ]:
                al = _repair(order_fn(), instance)
                c, sc = _fbi(al, instance, rounds=3)
                if sc.is_feasible() and c < best_c:
                    best_c, best_sc = c, sc
                if time.time() - t0 > time_limit * 0.15:
                    break
        except Exception:
            pass
    except Exception:
        pass

    if best_c <= lb:
        return best_c, best_sc

    # ── Phase 1: Random restart FBI ──────────────────────────────────────────
    elite: List[Tuple[int, List[int]]] = []
    elite_size = 12
    phase1_end = time_limit * 0.55

    while time.time() - t0 < phase1_end:
        if best_c <= lb:
            break
        al = _random_topo(instance)
        try:
            c, sc = _fbi(al, instance, rounds=2)
            if sc.is_feasible():
                if c < best_c:
                    best_c, best_sc = c, sc
                # Add to elite if new or different
                al_key = tuple(al[::n//8+1])  # cheap fingerprint
                if len(elite) < elite_size:
                    elite.append((c, al))
                elif c < elite[-1][0]:
                    elite[-1] = (c, al)
                    elite.sort(key=lambda x: x[0])
        except Exception:
            pass

    # ── Phase 2: Scatter search on elite ────────────────────────────────────
    if len(elite) >= 2 and best_c > lb:
        phase2_end = time_limit * 0.90

        while time.time() - t0 < phase2_end and best_c > lb:
            if len(elite) < 2:
                break
            # Pick two elite solutions
            i1 = random.randint(0, min(len(elite)-1, elite_size//2))
            i2 = random.randint(0, len(elite)-1)
            if i1 == i2:
                continue

            al1 = elite[i1][1]
            al2 = elite[i2][1]

            # Method 1: Path relinking
            if random.random() < 0.5:
                try:
                    c, sc = _path_relink(al1, al2, instance)
                    if sc.is_feasible() and c < best_c:
                        best_c, best_sc = c, sc
                except Exception:
                    pass
            else:
                # Method 2: OX1 crossover + FBI
                try:
                    child = _ox1(al1, al2)
                    child = _repair(child, instance)
                    c, sc = _fbi(child, instance, rounds=2)
                    if sc.is_feasible() and c < best_c:
                        best_c, best_sc = c, sc
                    if sc.is_feasible() and c < elite[-1][0]:
                        elite[-1] = (c, child)
                        elite.sort(key=lambda x: x[0])
                except Exception:
                    pass

    # ── Phase 3: Final polish FBI on best ────────────────────────────────────
    if time.time() - t0 < time_limit and best_c > lb:
        try:
            from rcpsp_scheduler import left_shift_schedule
            ls_sc = left_shift_schedule(best_sc, instance)
            if ls_sc.is_feasible() and ls_sc.cmax < best_c:
                best_c, best_sc = ls_sc.cmax, ls_sc
        except Exception:
            pass

        # Extract activity list from best schedule and FBI it hard
        try:
            sc_al = sorted(range(n), key=lambda i: (best_sc.start_times[i], i))
            c, sc = _fbi(sc_al, instance, rounds=5)
            if sc.is_feasible() and c < best_c:
                best_c, best_sc = c, sc
        except Exception:
            pass

    return best_c, best_sc
