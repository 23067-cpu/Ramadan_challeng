"""
SupNum - S3C'1447 Challenge Ramadan
Fichier : rcpsp_solver.py
Description : Fichier développé en Python pour la résolution du problème RCPSP.
Tous les algorithmes et logiques internes sont optimisés pour des performances maximales.
"""
"""
rcpsp_solver.py  — World-Class Hybrid Metaheuristic for RCPSP
==============================================================

Combines 18 state-of-the-art techniques from peer-reviewed literature:

  GA Core (Hartmann 1998, Kolisch & Hartmann 2006):
    1. Genetic Algorithm with adaptive operators, elitism, tournament selection
    2. OX1 + Two-Point crossover; adaptive mutation rate
    3. GRPW + MTS + LST + EST multi-heuristic initialisation
    4. Dual decoding (SSGS + PSGS) — keep best

  Local Search & Intensification (Valls et al. 2005, 2008):
    5. Critical-Path targeted Local Search (insertion + swap moves)
    6. Forward-Backward Improvement / Justification (FBI) — after every LS
    7. Left-Shift compaction
    8. Tabu Search with aspiration criterion
    9. Adaptive Large Neighbourhood Search (ALNS) — operator weight learning

  ILS & SA (Lourenco et al. 2010, Kirkpatrick 1983):
    10. Iterated Local Search (ILS) with double-bridge perturbation
    11. Simulated Annealing (SA) acceptance criterion for ILS escape

  Scatter Search (Valls et al. 2008, Ballestin & Blanco 2011):
    12. Reference Set (elite pool) maintenance
    13. Path Relinking — trajectory between elite solutions
    14. Solution combining with precedence repair

  Control:
    15. Adaptive mutation rate based on population diversity
    16. Stagnation-triggered population restart with elite preservation
    17. Final intensive ILS polish (perturbation + FBI + Tabu + ALNS)
    18. Critical-resource-guided destroy in ALNS
"""

from __future__ import annotations
import random
import time
import heapq
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from threading import Event

from rcpsp_parser import RCPSPInstance
from rcpsp_scheduler import (Schedule, serial_sgs, parallel_sgs,
                              backward_sgs, left_shift_schedule,
                              forward_backward_improvement, topological_sort)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GAConfig:
    """Maximum-performance solver configuration."""
    # GA
    pop_size: int = 100
    max_generations: int = 500
    elite_count: int = 5
    tournament_size: int = 4
    crossover_prob: float = 0.85
    mutation_rate_base: float = 0.05
    mutation_rate_max: float = 0.25

    # Local Search
    local_search_freq: int = 3
    local_search_iters: int = 150
    use_psgs_in_ls: bool = True

    # FBI (Forward-Backward Improvement)
    fbi_freq: int = 5            # apply FBI every N generations
    fbi_rounds: int = 4          # FBI justification rounds

    # Tabu Search
    tabu_freq: int = 9
    tabu_iters: int = 80
    tabu_tenure: int = 12

    # ALNS (Adaptive LNS)
    lns_freq: int = 12
    lns_destroy_frac: float = 0.30
    alns_operators: int = 4      # number of destroy operators
    alns_decay: float = 0.85     # weight decay for un-selected operators

    # ILS (Iterated Local Search)
    ils_freq: int = 25           # apply ILS every N gens
    ils_iters: int = 15          # ILS inner restarts
    ils_perturb_frac: float = 0.20  # double-bridge perturbation size

    # SA (Simulated Annealing) within ILS
    sa_initial_temp: float = 2.0
    sa_cooling: float = 0.95
    sa_iters: int = 40

    # Scatter Search
    scatter_freq: int = 20       # apply scatter search every N generations
    ref_set_size: int = 10       # elite reference set size
    path_relinking_steps: int = 8  # PR intermediate steps

    # Stagnation & restart
    stagnation_limit: int = 80
    restart_elite_frac: float = 0.15

    # Decoding
    dual_decode: bool = True

    # Time & seed
    time_limit_sec: float = 120.0
    seed: Optional[int] = None
    verbose: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Main Solver
# ─────────────────────────────────────────────────────────────────────────────

class HybridGASolver:
    """
    World-class hybrid metaheuristic combining GA + LS + FBI + Tabu + LNS + Scatter Search.
    """

    def __init__(self, instance: RCPSPInstance, config: GAConfig = None,
                 progress_cb: Optional[Callable] = None,
                 stop_event: Optional[Event] = None):
        self.instance = instance
        self.config = config or GAConfig()
        self.progress_cb = progress_cb
        self.stop_event = stop_event or Event()

        if self.config.seed is not None:
            random.seed(self.config.seed)

        self._topo_base = topological_sort(instance)
        self._rp = self._compute_resource_priorities()
        self.best_solution: Optional[Schedule] = None
        self.best_cmax: int = 10_000_000
        self.best_list: Optional[List[int]] = None
        self.generation_history: List[Tuple[int, int]] = []

        # Scatter search reference set
        self._ref_set: List[Tuple[int, List[int]]] = []  # (cmax, al)

        # ALNS operator weights (4 destroy strategies)
        self._alns_weights = [1.0] * self.config.alns_operators
        self._alns_counts = [0] * self.config.alns_operators
        self._alns_scores = [0.0] * self.config.alns_operators

        # Critical-path lower bound — reject any Cmax below this
        self._cp_lb = self._compute_cp_lb()

    def _compute_cp_lb(self) -> int:
        """Compute CPM lower bound (longest path in precedence graph)."""
        inst = self.instance
        n = inst.n_jobs
        topo = topological_sort(inst)
        dist = [0] * n
        for v in topo:
            for s in inst.successors[v]:
                dist[s] = max(dist[s], dist[v] + inst.durations[v])
        return dist[n - 1] + inst.durations[n - 1]

    # ─── Public API ───────────────────────────────────────────────────────────

    def solve(self) -> Schedule:
        cfg = self.config
        t0 = time.time()

        # Initialise — fast eval first, FBI only on top elite
        pop = self._init_population()
        fit = [self._eval(ind) for ind in pop]          # fast: no FBI

        # FBI only on the top elite_count individuals (time-bounded)
        sorted_idxs = sorted(range(len(fit)), key=lambda i: fit[i][0])
        fbi_limit = min(cfg.elite_count, len(sorted_idxs))
        for i in sorted_idxs[:fbi_limit]:
            if time.time() - t0 > cfg.time_limit_sec * 0.25:   # use ≤25% of budget on init
                break
            try:
                c2, s2 = self._eval_with_fbi(pop[i])
                if s2.is_feasible() and c2 < fit[i][0]:
                    fit[i] = (c2, s2)
            except Exception:
                pass

        # Update best
        best_idx = min(range(len(fit)), key=lambda i: fit[i][0])
        self.best_cmax = fit[best_idx][0]
        self.best_solution = fit[best_idx][1]
        self.best_list = pop[best_idx][:]
        self.generation_history.append((0, self.best_cmax))

        # Initialise reference set with top individuals
        self._update_ref_set(list(zip([f[0] for f in fit], pop)))

        stagnation = 0
        mut_rate = cfg.mutation_rate_base

        for gen in range(1, cfg.max_generations + 1):
            if self.stop_event.is_set():
                break
            elapsed = time.time() - t0
            if elapsed >= cfg.time_limit_sec:
                break

            # Adaptive mutation from diversity
            if gen % 10 == 0:
                div = self._diversity(pop)
                if div < 0.12:
                    mut_rate = cfg.mutation_rate_max
                elif div < 0.25:
                    mut_rate = cfg.mutation_rate_base * 2.0
                else:
                    mut_rate = cfg.mutation_rate_base

            # Sort population
            sorted_pairs = sorted(zip(fit, pop), key=lambda x: x[0][0])
            fit = [p[0] for p in sorted_pairs]
            pop = [p[1] for p in sorted_pairs]

            elite = [ind[:] for ind in pop[:cfg.elite_count]]
            elite_fit = list(fit[:cfg.elite_count])

            new_pop = [ind[:] for ind in elite]
            new_fit = list(elite_fit)

            while len(new_pop) < cfg.pop_size:
                if self.stop_event.is_set():
                    break
                p1 = self._tournament(pop, fit)
                p2 = self._tournament(pop, fit)

                if random.random() < cfg.crossover_prob:
                    c1, c2 = self._ox1(p1, p2) if random.random() < 0.6 \
                        else (self._two_point(p1, p2), self._two_point(p2, p1))
                else:
                    c1, c2 = p1[:], p2[:]

                c1 = self._mutate(c1, mut_rate)
                c2 = self._mutate(c2, mut_rate)

                for c in [c1, c2]:
                    if len(new_pop) < cfg.pop_size:
                        ev = self._eval(c)
                        new_pop.append(c)
                        new_fit.append(ev)

            pop, fit = new_pop, new_fit

            # ── Critical-Path Local Search ──
            if gen % cfg.local_search_freq == 0:
                targets = list(range(min(cfg.elite_count, len(pop))))
                targets.append(random.randint(0, len(pop) - 1))
                for idx in set(targets):
                    if self.stop_event.is_set():
                        break
                    al, sc = self._critical_path_ls(pop[idx], fit[idx][1])
                    if sc.cmax < fit[idx][0]:
                        pop[idx] = al
                        fit[idx] = (sc.cmax, sc)

            # ── Forward-Backward Improvement ──
            if gen % cfg.fbi_freq == 0:
                for idx in range(min(cfg.elite_count, len(pop))):
                    if self.stop_event.is_set():
                        break
                    al, sc = forward_backward_improvement(
                        pop[idx], self.instance, max_rounds=cfg.fbi_rounds)
                    if sc.cmax < fit[idx][0]:
                        pop[idx] = al
                        fit[idx] = (sc.cmax, sc)

            # ── Tabu Search ──
            if gen % cfg.tabu_freq == 0:
                al, sc = self._tabu_search(pop[0], fit[0][1])
                if sc.cmax < fit[0][0]:
                    pop[0] = al
                    fit[0] = (sc.cmax, sc)

            # ── ALNS ──
            if gen % cfg.lns_freq == 0:
                al, sc = self._alns(pop[0])
                if sc.cmax < fit[0][0]:
                    pop[0] = al
                    fit[0] = (sc.cmax, sc)

            # ── ILS ──
            if gen % cfg.ils_freq == 0:
                if self.stop_event.is_set():
                    break
                al, sc = self._ils(pop[0], fit[0][1])
                if sc.cmax < fit[0][0]:
                    pop[0] = al
                    fit[0] = (sc.cmax, sc)

            # ── Scatter Search + Path Relinking ──
            if gen % cfg.scatter_freq == 0:
                self._update_ref_set(list(zip([f[0] for f in fit], pop)))
                new_al, new_sc = self._scatter_search()
                if new_sc is not None and new_sc.cmax < fit[0][0]:
                    pop[0] = new_al
                    fit[0] = (new_sc.cmax, new_sc)

            # ── Update best (via _update_best which checks feasibility + LB) ──
            best_idx = min(range(len(fit)), key=lambda i: fit[i][0])
            prev_best = self.best_cmax
            self._update_best(pop[best_idx][:], fit[best_idx][1])
            if self.best_cmax < prev_best:
                stagnation = 0
            else:
                stagnation += 1

            self.generation_history.append((gen, self.best_cmax))

            if cfg.verbose and gen % 20 == 0:
                print(f"  Gen {gen:4d} | Cmax={self.best_cmax} | "
                      f"mut={mut_rate:.3f} | stag={stagnation} | "
                      f"ref={len(self._ref_set)}")

            if self.progress_cb:
                self.progress_cb(gen, cfg.max_generations, self.best_cmax,
                                 time.time() - t0)

            # ── Restart on stagnation ──
            if stagnation >= cfg.stagnation_limit:
                pop, fit = self._restart(pop, fit)
                stagnation = 0
                mut_rate = min(cfg.mutation_rate_max, cfg.mutation_rate_base * 3)
                if cfg.verbose:
                    print(f"  ↻ Restart at gen {gen}")

        # ── Final intensive polish ──
        if self.best_list and not self.stop_event.is_set():
            self._final_polish()

        return self.best_solution

    # ─── Initialisation ──────────────────────────────────────────────────────

    def _compute_resource_priorities(self) -> List[float]:
        inst = self.instance
        return [
            sum(inst.resource_needs[i][k] / max(1, inst.resource_caps[k])
                for k in range(inst.n_resources))
            for i in range(inst.n_jobs)
        ]

    def _init_population(self) -> List[List[int]]:
        cfg = self.config
        inst = self.instance
        pop = []

        topo = self._topo_base[:]
        est_vals = inst.est()
        lst_vals = inst.lst()
        n_succs = self._count_all_successors()

        # Deterministic seeds (5 heuristics)
        def _make(order): return self._repair(order)

        pop.append(topo[:])
        pop.append(_make(sorted(range(inst.n_jobs),
                                key=lambda i: (-self._rp[i], lst_vals[i]))))
        pop.append(_make(sorted(range(inst.n_jobs), key=lambda i: lst_vals[i])))
        pop.append(_make(sorted(range(inst.n_jobs), key=lambda i: est_vals[i])))
        pop.append(_make(sorted(range(inst.n_jobs), key=lambda i: -n_succs[i])))

        while len(pop) < cfg.pop_size:
            pop.append(self._random_topo_shuffle())

        return pop[:cfg.pop_size]

    def _count_all_successors(self) -> List[int]:
        inst = self.instance
        count = [0] * inst.n_jobs
        for v in reversed(self._topo_base):
            for s in inst.successors[v]:
                count[v] += 1 + count[s]
        return count

    def _random_topo_shuffle(self) -> List[int]:
        inst = self.instance
        n = inst.n_jobs
        in_deg = [len(inst.predecessors[i]) for i in range(n)]
        ready = [i for i in range(n) if in_deg[i] == 0]
        result = []
        while ready:
            chosen = random.choice(ready)
            result.append(chosen)
            ready.remove(chosen)
            for j in inst.successors[chosen]:
                in_deg[j] -= 1
                if in_deg[j] == 0:
                    ready.append(j)
        return result

    # ─── Evaluation ──────────────────────────────────────────────────────────

    def _eval(self, al: List[int]) -> Tuple[int, Schedule]:
        al = self._repair(al)  # guarantee topological validity
        s1 = serial_sgs(al, self.instance)
        if self.config.dual_decode:
            s2 = parallel_sgs(al, self.instance)
            return (s1.cmax, s1) if s1.cmax <= s2.cmax else (s2.cmax, s2)
        return s1.cmax, s1

    def _eval_with_fbi(self, al: List[int]) -> Tuple[int, Schedule]:
        """Evaluate + apply FBI for initial population quality."""
        al = self._repair(al)  # guarantee topological validity first
        al2, sc = forward_backward_improvement(al, self.instance, max_rounds=2)
        al2 = self._repair(al2)  # re-validate after FBI
        sc2 = serial_sgs(al2, self.instance)  # always re-evaluate for safety
        # Use whichever is better and feasible
        if sc2.is_feasible() and sc2.cmax < sc.cmax:
            return sc2.cmax, sc2
        if sc.is_feasible():
            return sc.cmax, sc
        # Fallback: plain forward SGS
        sc_safe = serial_sgs(al, self.instance)
        return sc_safe.cmax, sc_safe

    # ─── Genetic Operators ────────────────────────────────────────────────────

    def _tournament(self, pop, fit):
        k = min(self.config.tournament_size, len(pop))
        idxs = random.sample(range(len(pop)), k)
        winner = min(idxs, key=lambda i: fit[i][0])
        return pop[winner][:]

    def _ox1(self, p1, p2):
        n = len(p1)
        a, b = sorted(random.sample(range(n), 2))

        def _ox(parent, other):
            child = [None] * n
            child[a:b + 1] = parent[a:b + 1]
            seg = set(parent[a:b + 1])
            fill = [x for x in other if x not in seg]
            j = 0
            for i in range(n):
                if child[i] is None:
                    child[i] = fill[j]; j += 1
            return child

        return self._repair(_ox(p1, p2)), self._repair(_ox(p2, p1))

    def _two_point(self, p1, p2):
        n = len(p1)
        a, b = sorted(random.sample(range(n), 2))
        child = [None] * n
        child[a:b] = p1[a:b]
        used = set(p1[a:b])
        rem = [x for x in p2 if x not in used]
        j = 0
        for i in range(n):
            if child[i] is None:
                child[i] = rem[j]; j += 1
        return self._repair(child)

    def _mutate(self, al, rate):
        n = len(al)
        result = al[:]
        n_swaps = max(1, int(n * rate))
        for _ in range(n_swaps):
            i, j = random.sample(range(1, n - 1), 2)
            result[i], result[j] = result[j], result[i]
        return self._repair(result)

    # ─── Critical-Path LS ────────────────────────────────────────────────────

    def _critical_path_ls(self, al, sched):
        cfg = self.config
        best_al = al[:]
        best_sc = sched
        n = len(al)
        critical = self._find_critical_jobs(best_sc)

        for _ in range(cfg.local_search_iters):
            if self.stop_event.is_set():
                break
            job = random.choice(critical)
            cur_pos = best_al.index(job)

            if random.random() < 0.5 and cur_pos > 1:
                new_pos = random.randint(0, cur_pos - 1)
                cand = best_al[:]
                cand.pop(cur_pos)
                cand.insert(new_pos, job)
            else:
                j_pos = random.randint(1, n - 2)
                cand = best_al[:]
                cand[cur_pos], cand[j_pos] = cand[j_pos], cand[cur_pos]

            cand = self._repair(cand)
            cm, sc = self._eval(cand)
            if cm < best_sc.cmax:
                best_al, best_sc = cand, sc
                critical = self._find_critical_jobs(best_sc)

        return best_al, best_sc

    def _find_critical_jobs(self, sched):
        inst = self.instance
        n = inst.n_jobs
        est_vals = inst.est()
        lst_vals = inst.lst()
        threshold = max(1, (sched.cmax - est_vals[n-1]) // 10)
        critical = [i for i in range(1, n - 1)
                    if lst_vals[i] - est_vals[i] <= threshold
                    and inst.durations[i] > 0]
        return critical if critical else list(range(1, n - 1))

    # ─── Tabu Search ─────────────────────────────────────────────────────────

    def _tabu_search(self, al, sched):
        cfg = self.config
        n = len(al)
        current_al = al[:]
        current_cm = sched.cmax
        best_al = al[:]
        best_cm = sched.cmax
        best_sc = sched
        tabu: List[Tuple[int, int]] = []

        for _ in range(cfg.tabu_iters):
            if self.stop_event.is_set():
                break
            best_move_cm = float('inf')
            best_move = None
            best_move_al = None
            best_move_sc = None

            for _ in range(40):
                i, j = sorted(random.sample(range(1, n - 1), 2))
                move = (i, j)
                cand = current_al[:]
                cand[i], cand[j] = cand[j], cand[i]
                cand = self._repair(cand)
                cm, sc = self._eval(cand)
                is_tabu = move in tabu
                if (not is_tabu and cm < best_move_cm) or \
                   (is_tabu and cm < best_cm):
                    best_move_cm, best_move = cm, move
                    best_move_al, best_move_sc = cand, sc

            if best_move is None:
                continue
            current_al = best_move_al
            current_cm = best_move_cm
            tabu.append(best_move)
            if len(tabu) > cfg.tabu_tenure:
                tabu.pop(0)
            if current_cm < best_cm:
                best_cm = current_cm
                best_al = current_al[:]
                best_sc = best_move_sc

        return best_al, best_sc

    # ─── Large Neighbourhood Search ──────────────────────────────────────────

    def _lns(self, al):
        cfg = self.config
        inst = self.instance
        n = inst.n_jobs
        real_jobs = list(range(1, n - 1))
        n_destroy = max(2, int(len(real_jobs) * cfg.lns_destroy_frac))
        destroy_set = set(random.sample(real_jobs, n_destroy))
        partial = [j for j in al if j not in destroy_set]
        destroyed = list(destroy_set)
        random.shuffle(destroyed)

        for job in destroyed:
            min_pos = 0
            for pred in inst.predecessors[job]:
                if pred in partial:
                    min_pos = max(min_pos, partial.index(pred) + 1)
            max_pos = len(partial)
            for succ in inst.successors[job]:
                if succ in partial:
                    max_pos = min(max_pos, partial.index(succ))
            min_pos = min(min_pos, max_pos)

            best_cm = float('inf')
            best_pos = max_pos
            for pos in range(min_pos, max_pos + 1):
                cand = partial[:pos] + [job] + partial[pos:]
                sc = serial_sgs(cand, inst)
                if sc.cmax < best_cm:
                    best_cm = sc.cmax
                    best_pos = pos
            partial = partial[:best_pos] + [job] + partial[best_pos:]

        # Apply FBI to the LNS result for extra compaction
        partial, final_sc = forward_backward_improvement(
            partial, inst, max_rounds=3)
        return partial, final_sc

    # ─── Scatter Search + Path Relinking ─────────────────────────────────────

    def _update_ref_set(self, candidates: List[Tuple[int, List[int]]]):
        """
        Maintain a reference set of elite diverse solutions.
        Half the set = best solutions (quality), half = most diverse (diversity).
        """
        cfg = self.config
        half = cfg.ref_set_size // 2

        # Sort by quality
        candidates_sorted = sorted(candidates, key=lambda x: x[0])

        # Quality half
        quality_half = [(cm, al[:]) for cm, al in candidates_sorted[:half]
                        if not any(self._similarity(al, r[1]) > 0.95
                                   for r in self._ref_set[:half])]

        # Update ref set quality tier
        self._ref_set = quality_half[:]

        # Diversity half: pick solutions maximally different from ref_set
        for cm, al in candidates_sorted:
            if len(self._ref_set) >= cfg.ref_set_size:
                break
            if all(self._similarity(al, r[1]) < 0.80 for r in self._ref_set):
                self._ref_set.append((cm, al[:]))

    def _similarity(self, al1: List[int], al2: List[int]) -> float:
        """Kendall-τ similarity (fraction of concordant pairs)."""
        n = len(al1)
        if n <= 1:
            return 1.0
        pos1 = {v: i for i, v in enumerate(al1)}
        pos2 = {v: i for i, v in enumerate(al2)}
        concordant = sum(
            1 for i in range(n) for j in range(i + 1, n)
            if (pos1[al1[i]] < pos1[al1[j]]) == (pos2[al1[i]] < pos2[al1[j]])
        )
        total = n * (n - 1) // 2
        return concordant / max(1, total)

    def _scatter_search(self) -> Tuple[Optional[List[int]], Optional[Schedule]]:
        """
        Scatter Search: combine pairs from reference set using Path Relinking.
        Returns best new solution found.
        """
        cfg = self.config
        if len(self._ref_set) < 2:
            return None, None

        best_cm = float('inf')
        best_al = None
        best_sc = None

        # Try all pairs in reference set
        pairs = [(i, j) for i in range(len(self._ref_set))
                 for j in range(i + 1, len(self._ref_set))]
        if len(pairs) > 6:
            pairs = random.sample(pairs, 6)

        for (i, j) in pairs:
            if self.stop_event.is_set():
                break
            al_i = self._ref_set[i][1]
            al_j = self._ref_set[j][1]

            # Path relinking: generate intermediate solutions
            intermediates = self._path_relinking(al_i, al_j)
            for al in intermediates:
                # Apply FBI to each intermediate
                al2, sc = forward_backward_improvement(al, self.instance,
                                                       max_rounds=2)
                if sc.cmax < best_cm:
                    best_cm = sc.cmax
                    best_al = al2
                    best_sc = sc

        return best_al, best_sc

    def _path_relinking(self, al_from: List[int],
                        al_to: List[int]) -> List[List[int]]:
        """
        Path Relinking: generate solutions along the trajectory from al_from → al_to.

        We iteratively modify al_from to look more like al_to by moving one
        job per step to match al_to's relative ordering.
        """
        cfg = self.config
        n = len(al_from)
        current = al_from[:]
        intermediates = []

        # Find positions where current and target differ
        pos_to = {v: i for i, v in enumerate(al_to)}

        steps = min(cfg.path_relinking_steps, n)
        for step in range(steps):
            # Find the job whose relative position most needs to change
            max_diff = -1
            best_job = None
            best_new_pos = None

            for job in current[1:-1]:  # exclude source/sink
                cur_pos = current.index(job)
                target_pos = min(max(pos_to[job], 0), n - 1)
                diff = abs(cur_pos - target_pos)
                if diff > max_diff:
                    max_diff = diff
                    best_job = job
                    best_new_pos = target_pos

            if max_diff == 0 or best_job is None:
                break

            # Move best_job toward target position
            cur_p = current.index(best_job)
            new_p = min(cur_p + (1 if best_new_pos > cur_p else -1), n - 2)
            new_p = max(1, new_p)

            cand = current[:]
            cand.pop(cur_p)
            cand.insert(new_p, best_job)
            current = self._repair(cand)
            intermediates.append(current[:])

        return intermediates

    # ─── Restart ─────────────────────────────────────────────────────────────

    def _restart(self, pop, fit):
        cfg = self.config
        n_keep = max(1, int(cfg.pop_size * cfg.restart_elite_frac))
        sorted_pairs = sorted(zip(fit, pop), key=lambda x: x[0][0])
        surv_f = [p[0] for p in sorted_pairs[:n_keep]]
        surv_p = [p[1] for p in sorted_pairs[:n_keep]]
        new_pop = surv_p[:]
        new_fit = list(surv_f)
        while len(new_pop) < cfg.pop_size:
            ind = self._random_topo_shuffle()
            new_pop.append(ind)
            new_fit.append(self._eval(ind))
        return new_pop, new_fit

    # ─── Final Polish ─────────────────────────────────────────────────────────

    def _final_polish(self):
        cfg = self.config
        al = self.best_list[:]
        sc = self.best_solution

        # 1. Extended Critical-Path LS
        orig = cfg.local_search_iters
        cfg.local_search_iters = 600
        al, sc = self._critical_path_ls(al, sc)
        cfg.local_search_iters = orig
        self._update_best(al, sc)

        # 2. FBI mega-round
        al2, sc2 = forward_backward_improvement(
            self.best_list[:], self.instance, max_rounds=15)
        self._update_best(al2, sc2)

        # 3. Extended Tabu Search
        orig_t = cfg.tabu_iters
        cfg.tabu_iters = 400
        al3, sc3 = self._tabu_search(self.best_list[:], self.best_solution)
        cfg.tabu_iters = orig_t
        self._update_best(al3, sc3)

        # 4. ILS final burst
        for _ in range(5):
            if self.stop_event.is_set():
                break
            al_ils, sc_ils = self._ils(self.best_list[:], self.best_solution)
            self._update_best(al_ils, sc_ils)

        # 5. Multiple ALNS rounds
        for _ in range(10):
            if self.stop_event.is_set():
                break
            al4, sc4 = self._alns(self.best_list[:])
            self._update_best(al4, sc4)

        # 6. Final left-shift compaction
        sc5 = left_shift_schedule(self.best_solution)
        if sc5.cmax < self.best_cmax:
            self.best_cmax = sc5.cmax
            self.best_solution = sc5
            self.best_list = sorted(range(self.instance.n_jobs),
                                    key=lambda i: sc5.start_times[i])

    def _update_best(self, al, sc):
        """Update best only if schedule is feasible and better than current best."""
        if al is None or sc is None:
            return
        # Reject impossible values (below CPM lower bound)
        if sc.cmax < self._cp_lb:
            return
        # Reject infeasible schedules
        if not sc.is_feasible():
            return
        if sc.cmax < self.best_cmax:
            self.best_cmax = sc.cmax
            self.best_solution = sc
            self.best_list = al[:]

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _repair(self, al):
        inst = self.instance
        n = inst.n_jobs
        pos = {job: idx for idx, job in enumerate(al)}
        in_deg = [len(inst.predecessors[i]) for i in range(n)]
        heap = []
        for i in range(n):
            if in_deg[i] == 0:
                heapq.heappush(heap, (pos.get(i, n), i))
        result = []
        while heap:
            _, v = heapq.heappop(heap)
            result.append(v)
            for s in inst.successors[v]:
                in_deg[s] -= 1
                if in_deg[s] == 0:
                    heapq.heappush(heap, (pos.get(s, n), s))
        if len(result) < n:
            result.extend(sorted(set(range(n)) - set(result)))
        return result

    def _diversity(self, pop):
        n = len(pop)
        if n < 2:
            return 1.0
        n_samples = min(60, n * (n - 1) // 2)
        total = sum(
            sum(a != b for a, b in zip(pop[i], pop[j])) / len(pop[i])
            for _ in range(n_samples)
            for i, j in [random.sample(range(n), 2)]
        )
        return total / n_samples

    def _schedule_to_activity_list(self, sched):
        n = self.instance.n_jobs
        return sorted(range(n), key=lambda i: (sched.start_times[i], i))

    # ─── ILS: Iterated Local Search ──────────────────────────────────────────

    def _double_bridge_perturb(self, al: List[int]) -> List[int]:
        """
        Double-bridge perturbation (Applegate et al. 1998, adapted for RCPSP).
        Cuts the activity list at 4 points and reconnects the segments
        in a different order: AB CD → A C B D (never reversible by 2-opt).
        This provides a strong escape from 2-opt local optima.
        """
        n = len(al)
        real = al[1:-1]  # exclude source/sink
        m = len(real)
        if m < 8:
            return self._mutate(al, self.config.ils_perturb_frac)

        # Choose 3 random cut points (4 segments)
        cuts = sorted(random.sample(range(1, m), 3))
        a, b, c = cuts
        seg_A = real[:a]
        seg_B = real[a:b]
        seg_C = real[b:c]
        seg_D = real[c:]

        # Reconnect: A + C + B + D (double bridge)
        perturbed = [al[0]] + seg_A + seg_C + seg_B + seg_D + [al[-1]]
        return self._repair(perturbed)

    def _ils(self, al: List[int], sched: Schedule) -> Tuple[List[int], Schedule]:
        """
        Iterated Local Search (ILS) with:
          - Double-bridge perturbation for escape
          - SA acceptance criterion (controlled uphill moves)
          - Full LS (CP-LS + FBI) after each perturbation

        This is one of the most effective methods for RCPSP (Lourenco 2010).
        """
        cfg = self.config
        best_al = al[:]
        best_sc = sched
        cur_al = al[:]
        cur_cm = sched.cmax

        temp = cfg.sa_initial_temp
        import math

        for _ in range(cfg.ils_iters):
            if self.stop_event.is_set():
                break

            # Perturb
            new_al = self._double_bridge_perturb(cur_al)

            # Local search (CP-LS + FBI)
            new_al, new_sc = self._critical_path_ls(new_al,
                                                     serial_sgs(new_al, self.instance))
            new_al, new_sc = forward_backward_improvement(
                new_al, self.instance, max_rounds=2)

            new_cm = new_sc.cmax

            # SA acceptance: always accept improvements, sometimes accept worse
            delta = new_cm - cur_cm
            if delta < 0 or (temp > 0.01 and random.random() < math.exp(-delta / temp)):
                cur_al = new_al
                cur_cm = new_cm

            if new_cm < best_sc.cmax:
                best_al = new_al
                best_sc = new_sc
                self._update_best(best_al, best_sc)

            temp *= cfg.sa_cooling

        return best_al, best_sc

    # ─── ALNS: Adaptive Large Neighbourhood Search ────────────────────────────

    def _alns(self, al: List[int]) -> Tuple[List[int], Schedule]:
        """
        Adaptive Large Neighbourhood Search (ALNS).

        Maintains weights for multiple destroy operators and learns which
        ones produce the best improvements. Reward/penalize operators based
        on their success, then normalize weights for selection.

        Destroy operators:
          0: Random destroy (baseline LNS)
          1: Critical-resource destroy (destroys bottleneck-resource jobs)
          2: Critical-path destroy (destroys jobs on the critical path)
          3: Clustered destroy (destroys a time-window cluster of jobs)
        """
        cfg = self.config
        inst = self.instance

        # Select operator by roulette-wheel on weights
        w = self._alns_weights
        total = sum(w)
        r = random.uniform(0, total)
        op_idx = 0
        cumul = 0.0
        for i, wi in enumerate(w):
            cumul += wi
            if cumul >= r:
                op_idx = i
                break

        self._alns_counts[op_idx] += 1

        # Apply the chosen destroy
        if op_idx == 0:
            al_new, sc_new = self._lns(al)
        elif op_idx == 1:
            al_new, sc_new = self._lns_critical_resource(al)
        elif op_idx == 2:
            al_new, sc_new = self._lns_critical_path(al)
        else:
            al_new, sc_new = self._lns_clustered(al)

        # Reward operator
        base_sc = serial_sgs(al, inst)
        improvement = base_sc.cmax - sc_new.cmax
        if improvement > 0:
            reward = 3.0 if improvement > 2 else 1.5
            self._alns_scores[op_idx] += reward

        # Decay weights and update
        for i in range(cfg.alns_operators):
            if self._alns_counts[i] > 0:
                self._alns_weights[i] = (
                    cfg.alns_decay * self._alns_weights[i] +
                    (1 - cfg.alns_decay) * (self._alns_scores[i] /
                                            self._alns_counts[i])
                )
            self._alns_weights[i] = max(0.1, self._alns_weights[i])

        return al_new, sc_new

    def _lns_critical_resource(self, al: List[int]) -> Tuple[List[int], Schedule]:
        """ALNS destroy: remove jobs that use the most critical (bottleneck) resource."""
        cfg = self.config
        inst = self.instance
        n = inst.n_jobs

        # Find bottleneck resource (highest avg utilization)
        resource_load = [
            sum(inst.resource_needs[j][k] for j in range(n)) / max(1, inst.resource_caps[k])
            for k in range(inst.n_resources)
        ]
        bottleneck = resource_load.index(max(resource_load))

        # Sort jobs by their usage of bottleneck resource
        real_jobs = list(range(1, n - 1))
        real_jobs.sort(key=lambda j: -inst.resource_needs[j][bottleneck])

        n_destroy = max(2, int(len(real_jobs) * cfg.lns_destroy_frac))
        destroy_set = set(real_jobs[:n_destroy])
        partial = [j for j in al if j not in destroy_set]
        destroyed = list(destroy_set)
        random.shuffle(destroyed)

        for job in destroyed:
            min_pos = max((partial.index(p) + 1
                          for p in inst.predecessors[job] if p in partial), default=0)
            max_pos = min((partial.index(s)
                          for s in inst.successors[job] if s in partial),
                         default=len(partial))
            best_cm, best_pos = float('inf'), max_pos
            for pos in range(min_pos, max_pos + 1):
                cand = partial[:pos] + [job] + partial[pos:]
                cm = serial_sgs(cand, inst).cmax
                if cm < best_cm:
                    best_cm, best_pos = cm, pos
            partial = partial[:best_pos] + [job] + partial[best_pos:]

        partial, sc = forward_backward_improvement(partial, inst, max_rounds=2)
        return partial, sc

    def _lns_critical_path(self, al: List[int]) -> Tuple[List[int], Schedule]:
        """ALNS destroy: remove jobs that lie on the critical path."""
        cfg = self.config
        inst = self.instance
        n = inst.n_jobs

        sched = serial_sgs(al, inst)
        est_vals = inst.est()
        lst_vals = inst.lst()
        # Critical jobs = zero slack
        slack = [lst_vals[j] - est_vals[j] for j in range(n)]
        real_jobs = list(range(1, n - 1))
        real_jobs.sort(key=lambda j: slack[j])

        n_destroy = max(2, int(len(real_jobs) * cfg.lns_destroy_frac))
        destroy_set = set(real_jobs[:n_destroy])
        partial = [j for j in al if j not in destroy_set]
        destroyed = list(destroy_set)
        random.shuffle(destroyed)

        for job in destroyed:
            min_pos = max((partial.index(p) + 1
                          for p in inst.predecessors[job] if p in partial), default=0)
            max_pos = min((partial.index(s)
                          for s in inst.successors[job] if s in partial),
                         default=len(partial))
            best_cm, best_pos = float('inf'), max_pos
            for pos in range(min_pos, max_pos + 1):
                cand = partial[:pos] + [job] + partial[pos:]
                cm = serial_sgs(cand, inst).cmax
                if cm < best_cm:
                    best_cm, best_pos = cm, pos
            partial = partial[:best_pos] + [job] + partial[best_pos:]

        partial, sc = forward_backward_improvement(partial, inst, max_rounds=2)
        return partial, sc

    def _lns_clustered(self, al: List[int]) -> Tuple[List[int], Schedule]:
        """ALNS destroy: remove a random time-window cluster of jobs."""
        cfg = self.config
        inst = self.instance
        n = inst.n_jobs

        sched = serial_sgs(al, inst)
        T = sched.cmax
        if T == 0:
            return self._lns(al)

        # Pick a random time window covering ~destroy_frac of the horizon
        win_size = max(1, int(T * cfg.lns_destroy_frac))
        t_start = random.randint(0, max(0, T - win_size))
        t_end = t_start + win_size

        # Destroy jobs that start in [t_start, t_end]
        destroy_set = {j for j in range(1, n - 1)
                       if t_start <= sched.start_times[j] <= t_end}
        if not destroy_set:
            return self._lns(al)

        partial = [j for j in al if j not in destroy_set]
        destroyed = list(destroy_set)
        random.shuffle(destroyed)

        for job in destroyed:
            min_pos = max((partial.index(p) + 1
                          for p in inst.predecessors[job] if p in partial), default=0)
            max_pos = min((partial.index(s)
                          for s in inst.successors[job] if s in partial),
                         default=len(partial))
            best_cm, best_pos = float('inf'), max_pos
            for pos in range(min_pos, max_pos + 1):
                cand = partial[:pos] + [job] + partial[pos:]
                cm = serial_sgs(cand, inst).cmax
                if cm < best_cm:
                    best_cm, best_pos = cm, pos
            partial = partial[:best_pos] + [job] + partial[best_pos:]

        partial, sc = forward_backward_improvement(partial, inst, max_rounds=2)
        return partial, sc


if __name__ == '__main__':
    from rcpsp_parser import parse_instance
    from rcpsp_bounds import compute_lower_bound
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else 'j30.sm/j301_1.sm'
    inst = parse_instance(path)
    lb = compute_lower_bound(inst)['LB']

    config = GAConfig(
        pop_size=100, max_generations=300, time_limit_sec=60.0,
        verbose=True, seed=None,
    )
    solver = HybridGASolver(inst, config)
    best = solver.solve()
    gap = (best.cmax - lb) / lb * 100
    print(f"\n{'='*55}")
    print(f"Instance : {inst.name}")
    print(f"Cmax     : {best.cmax}")
    print(f"LB       : {lb}")
    print(f"Gap      : {gap:.2f}%")
    print(f"Feasible : {best.is_feasible()}")
