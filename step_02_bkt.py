import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm

DEFAULT_BKT_PARAMS = {
    "p_init":  0.3,   # prior: P(knows skill at start)
    "p_learn": 0.09,  # P(learns after each attempt)
    "p_slip":  0.1,   # P(wrong answer | knows skill)
    "p_guess": 0.2,   # P(right answer | doesn't know skill)
}

def bkt_update(p_know, correct, params):
    p_slip   = params["p_slip"]
    p_guess  = params["p_guess"]
    p_learn  = params["p_learn"]

    # P(correct | know) and P(correct | not know)
    p_correct_given_know    = 1 - p_slip
    p_correct_given_notknow = p_guess

    # Bayes update: posterior P(know | observation)
    if correct == 1:
        p_know_given_obs = (p_know * p_correct_given_know) / (
            p_know * p_correct_given_know
            + (1 - p_know) * p_correct_given_notknow
        )
    else:
        p_know_given_obs = (p_know * p_slip) / (
            p_know * p_slip
            + (1 - p_know) * (1 - p_guess)
        )

    # Learning update: P(know after this step)
    p_know_next = p_know_given_obs + (1 - p_know_given_obs) * p_learn

    # Clip to valid probability range
    return float(np.clip(p_know_next, 0.0, 1.0))


def fit_bkt_em(skill_interactions, n_iter=20, params_init=None):
    if params_init is None:
        params = DEFAULT_BKT_PARAMS.copy()
    else:
        params = params_init.copy()

    for _ in range(n_iter):
        # E-step: compute expected sufficient statistics
        total_correct_know    = 0.0
        total_wrong_know      = 0.0
        total_correct_notknow = 0.0
        total_wrong_notknow   = 0.0
        total_learn           = 0.0
        total_notlearn        = 0.0
        total_init_know       = 0.0
        total_init_notknow    = 0.0

        for seq in skill_interactions:
            if len(seq) == 0:
                continue
            p_know = params["p_init"]

            for t, correct in enumerate(seq):
                p_slip   = params["p_slip"]
                p_guess  = params["p_guess"]
                p_learn  = params["p_learn"]

                p_c_k  = 1 - p_slip
                p_c_nk = p_guess

                # Posterior P(know | obs)
                if correct == 1:
                    p_k = (p_know * p_c_k) / max(
                        p_know * p_c_k + (1 - p_know) * p_c_nk, 1e-10
                    )
                else:
                    p_k = (p_know * p_slip) / max(
                        p_know * p_slip + (1 - p_know) * (1 - p_guess), 1e-10
                    )

                # Accumulate sufficient statistics
                if correct == 1:
                    total_correct_know    += p_k
                    total_correct_notknow += (1 - p_k)
                else:
                    total_wrong_know      += p_k
                    total_wrong_notknow   += (1 - p_k)

                # For p_learn: expected transitions not-know -> know
                total_learn    += (1 - p_k) * p_learn
                total_notlearn += (1 - p_k) * (1 - p_learn)

                if t == 0:
                    total_init_know    += p_k
                    total_init_notknow += (1 - p_k)

                # Update for next step
                p_know = p_k + (1 - p_k) * p_learn

        # M-step: update parameters
        eps = 1e-10
        params["p_init"]  = total_init_know / max(total_init_know + total_init_notknow, eps)
        params["p_slip"]  = total_wrong_know / max(total_correct_know + total_wrong_know, eps)
        params["p_guess"] = total_correct_notknow / max(total_correct_notknow + total_wrong_notknow, eps)
        params["p_learn"] = total_learn / max(total_learn + total_notlearn, eps)

        # Clip to reasonable bounds
        for k in params:
            params[k] = float(np.clip(params[k], 0.01, 0.99))

    return params


def fit_all_skills(df, skill_to_idx, min_students=5, fit_em=True):
    skill_params = {}

    for skill_id, group in tqdm(df.groupby("skill_id")):
        # Group by student, collect response sequences
        student_seqs = []
        for _, sgroup in group.groupby("user_id"):
            seq = sgroup.sort_values("order_id")["correct"].tolist()
            student_seqs.append(seq)

        if len(student_seqs) < min_students or not fit_em:
            skill_params[skill_id] = DEFAULT_BKT_PARAMS.copy()
        else:
            try:
                skill_params[skill_id] = fit_bkt_em(student_seqs)
            except Exception:
                skill_params[skill_id] = DEFAULT_BKT_PARAMS.copy()

    print(f"  Fitted parameters for {len(skill_params)} skills.")
    return skill_params


def compute_knowledge_states(sequences, skill_to_idx, skill_params):
    n_skills = len(skill_to_idx)
    student_states = {}

    for user_id, seq in tqdm(sequences.items()):
        # Initialize knowledge state: P(know each skill) = p_init for that skill
        state = np.array([
            skill_params.get(sid, DEFAULT_BKT_PARAMS)["p_init"]
            for sid in [
                list(skill_to_idx.keys())[i]
                for i in range(n_skills)
            ]
        ])

        trajectory = []
        for skill_id, correct in seq:
            if skill_id not in skill_to_idx:
                continue

            idx = skill_to_idx[skill_id]
            params = skill_params.get(skill_id, DEFAULT_BKT_PARAMS)

            trajectory.append({
                "skill_id": skill_id,
                "correct": correct,
                "state_before": state.copy(),
            })

            state[idx] = bkt_update(state[idx], correct, params)

        student_states[user_id] = trajectory

    print(f"  Average trajectory length: "
          f"{np.mean([len(t) for t in student_states.values()]):.1f}")
    return student_states


def analyze_knowledge_states(student_states, skill_to_idx, idx_to_skill):
    print("\nKnowledge state analysis")

    # Distribution of final knowledge states
    final_states = []
    for trajectory in student_states.values():
        if trajectory:
            final_states.append(trajectory[-1]["state_before"])

    final_states = np.array(final_states)
    mean_final = final_states.mean(axis=0)

    # Top 10 best and worst known skills at end
    top_idx = np.argsort(mean_final)[-10:][::-1]
    bot_idx = np.argsort(mean_final)[:10]

    print("\nTop 10 best-known skills (end of session):")
    for i in top_idx:
        sid = list(skill_to_idx.keys())[i]
        print(f"  {sid}: P(know) = {mean_final[i]:.3f}")

    print("\nTop 10 least-known skills:")
    for i in bot_idx:
        sid = list(skill_to_idx.keys())[i]
        print(f"  {sid}: P(know) = {mean_final[i]:.3f}")


if __name__ == "__main__":
    with open("data/preprocessed.pkl", "rb") as f:
        data = pickle.load(f)

    df           = data["df"]
    sequences    = data["sequences"]
    skill_to_idx = data["skill_to_idx"]
    idx_to_skill = data["idx_to_skill"]
    skill_names  = data["skill_names"]

    # Fit BKT parameters per skill
    skill_params = fit_all_skills(df, skill_to_idx, fit_em=True)

    # Compute knowledge state trajectories
    student_states = compute_knowledge_states(sequences, skill_to_idx, skill_params)

    analyze_knowledge_states(student_states, skill_to_idx, idx_to_skill)

    with open("data/bkt_results.pkl", "wb") as f:
        pickle.dump({
            "skill_params": skill_params,
            "student_states": student_states,
        }, f)

    print("\nBKT results saved to data/bkt_results.pkl")
