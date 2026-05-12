import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict


# Mastery threshold
# A student is considered to have "mastered" a skill when
# P(know skill) >= this threshold
MASTERY_THRESHOLD = 0.95


def infer_prerequisite_graph_from_data(sequences, skill_to_idx, n_skills):
    # For each student, find the order of first appearance of each skill
    first_occurrence = {}  # user_id -> {skill_id: position}
    for user_id, seq in sequences.items():
        seen = {}
        for pos, (skill_id, _) in enumerate(seq):
            if skill_id not in seen:
                seen[skill_id] = pos
        first_occurrence[user_id] = seen

    # Count: how often does A appear before B (among students who practiced both)?
    skills = list(skill_to_idx.keys())
    n = len(skills)

    before_count = defaultdict(lambda: defaultdict(int))  # before_count[A][B] = count
    both_count   = defaultdict(lambda: defaultdict(int))  # both_count[A][B] = count

    for user_id, occ in first_occurrence.items():
        skill_list = list(occ.keys())
        for i, a in enumerate(skill_list):
            for b in skill_list:
                if a == b:
                    continue
                both_count[a][b] += 1
                if occ[a] < occ[b]:
                    before_count[a][b] += 1

    # Compute fraction A-before-B
    edges = []
    BEFORE_THRESHOLD = 0.85
    MIN_STUDENTS = 10

    for a in skills:
        for b in skills:
            if a == b:
                continue
            n_both = both_count[a][b]
            if n_both < MIN_STUDENTS:
                continue
            frac = before_count[a][b] / n_both
            if frac >= BEFORE_THRESHOLD:
                edges.append((a, b, frac, n_both))

    print(f"  Found {len(edges)} prerequisite edges.")

    # Build adjacency dict: prereqs[B] = list of A's that must come before B
    prereqs = defaultdict(list)
    for a, b, frac, n in edges:
        prereqs[b].append(a)

    return dict(prereqs), edges


def build_phi(state, skill_to_idx, prereqs, mastery_threshold=MASTERY_THRESHOLD):
    n_skills = len(skill_to_idx)
    admissible = []

    skills = list(skill_to_idx.keys())

    for i, skill_id in enumerate(skills):
        p_know = state[i]

        # Skip already mastered skills
        if p_know >= mastery_threshold:
            continue

        # Check prerequisites
        prereq_ids = prereqs.get(skill_id, [])
        prereqs_met = all(
            state[skill_to_idx[p]] >= mastery_threshold
            for p in prereq_ids
            if p in skill_to_idx
        )

        if prereqs_met:
            admissible.append(i)

    return admissible


def compute_psi(state, skill_id, skill_to_idx, skill_params,
                recent_history=None):
    if skill_id not in skill_to_idx:
        return 0.0

    idx = skill_to_idx[skill_id]
    p_know = state[idx]

    # Zone of proximal development proxy:
    # maximum Psi at p_know = 0.5 (uncertain = most to learn)
    zpd_score = 1.0 - abs(p_know - 0.5) * 2.0  # range [0, 1]

    # Optionally incorporate recency from recent history
    recency_penalty = 0.0
    if recent_history is not None:
        # Penalize skills practiced very recently (last 2 steps)
        recent_skills = [s for s, _ in recent_history[-2:]]
        if skill_id in recent_skills:
            recency_penalty = 0.2

    psi = zpd_score - recency_penalty
    return float(np.clip(psi, 0.0, 1.0))


def check_bdo(psi_value, phi_admissible, current_skill_idx,
              psi_threshold=0.4):
    if psi_value >= psi_threshold:
        return "preserve"
    elif current_skill_idx in phi_admissible or len(phi_admissible) > 0:
        return "update"
    else:
        return "transient"


def analyze_phi(student_states, skill_to_idx, prereqs):
    print("\nPhi (Prerequisite Graph) Analysis")

    admissible_counts = []
    n_skills = len(skill_to_idx)

    for user_id, trajectory in list(student_states.items())[:500]:  # sample
        for step in trajectory:
            state = step["state_before"]
            admissible = build_phi(state, skill_to_idx, prereqs)
            admissible_counts.append(len(admissible))

    admissible_counts = np.array(admissible_counts)
    print(f"  Total skills: {n_skills}")
    print(f"  Admissible actions per step:")
    print(f"    Mean:   {admissible_counts.mean():.1f}")
    print(f"    Median: {np.median(admissible_counts):.1f}")
    print(f"    Min:    {admissible_counts.min()}")
    print(f"    Max:    {admissible_counts.max()}")
    print(f"  Fraction of steps where Phi constrains "
          f"(< all skills): "
          f"{(admissible_counts < n_skills).mean():.3f}")


if __name__ == "__main__":
    with open("data/preprocessed.pkl", "rb") as f:
        data = pickle.load(f)
    with open("data/bkt_results.pkl", "rb") as f:
        bkt = pickle.load(f)

    sequences      = data["sequences"]
    skill_to_idx   = data["skill_to_idx"]
    idx_to_skill   = data["idx_to_skill"]
    student_states = bkt["student_states"]
    skill_params   = bkt["skill_params"]

    n_skills = len(skill_to_idx)

    # Build prerequisite graph from data
    prereqs, edges = infer_prerequisite_graph_from_data(sequences, skill_to_idx, n_skills)

    analyze_phi(student_states, skill_to_idx, prereqs)

    # BDO on one student
    print("\nBDO Demo (first student, first 5 steps)")
    demo_user = list(student_states.keys())[0]
    demo_traj = student_states[demo_user]

    for i, step in enumerate(demo_traj[:5]):
        state     = step["state_before"]
        skill_id  = step["skill_id"]
        correct   = step["correct"]

        idx       = skill_to_idx.get(skill_id)
        admissible = build_phi(state, skill_to_idx, prereqs)
        psi       = compute_psi(state, skill_id, skill_to_idx, skill_params)
        bdo       = check_bdo(psi, admissible, idx)

        print(f"  Step {i+1}: skill={skill_id}, correct={correct}, "
              f"P(know)={state[idx]:.3f}, Psi={psi:.3f}, "
              f"admissible={len(admissible)}, BDO={bdo}")

    with open("data/phi_results.pkl", "wb") as f:
        pickle.dump({
            "prereqs": prereqs,
            "prereq_edges": edges,
        }, f)

    print("\nPhi structure saved to data/phi_results.pkl")
