import numpy as np
import pickle
import random
from collections import defaultdict
from tqdm import tqdm

from step_03_prerequisite_graph import (
    build_phi, compute_psi, check_bdo, MASTERY_THRESHOLD
)
from step_02_bkt import bkt_update, DEFAULT_BKT_PARAMS

def reward_baseline(correct, **kwargs):
    return float(correct)


def reward_qa_coherence(correct, psi, bdo_outcome,
                         trajectory_recent=None, **kwargs):
    # Base
    r = float(correct)

    # ZPD alignment: reward exercises at the right difficulty
    r += 0.3 * psi

    # BDO-based adjustments
    if bdo_outcome == "transient":
        r -= 0.5   # Penalize cognitive overload

    # Productive regime: check recent trajectory
    if trajectory_recent is not None and len(trajectory_recent) >= 4:
        recent_bdo = trajectory_recent[-4:]  # last 4 BDO outcomes
        n_preserve = sum(1 for b in recent_bdo if b == "preserve")
        n_update   = sum(1 for b in recent_bdo if b == "update")

        # Healthy ratio: roughly 2-3 preserves per update
        if 1 <= n_update <= 2 and 1 <= n_preserve <= 3:
            r += 0.2  # Productive regime bonus

        # Stagnation: too many preserves, no advancement
        if n_preserve >= 4 and n_update == 0:
            r -= 0.3  # Stagnation penalty

    return float(r)

def discretize_state(knowledge_vector, n_buckets=3):
    boundaries = np.linspace(0, 1, n_buckets + 1)[1:-1]
    buckets = np.digitize(knowledge_vector, boundaries)
    return tuple(buckets.tolist())

class QLearningAgent:

    def __init__(self, n_skills, learning_rate=0.1, discount=0.9,
                 epsilon=0.3, epsilon_decay=0.995, min_epsilon=0.05):
        self.n_skills       = n_skills
        self.lr             = learning_rate
        self.gamma          = discount
        self.epsilon        = epsilon
        self.epsilon_decay  = epsilon_decay
        self.min_epsilon    = min_epsilon
        self.q_table        = defaultdict(lambda: np.zeros(n_skills))

    def get_action(self, state_key, admissible_actions):
        if admissible_actions is None:
            available = list(range(self.n_skills))
        else:
            available = admissible_actions
            if len(available) == 0:
                return random.randint(0, self.n_skills - 1)  # fallback

        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(available)
        else:
            q_values = self.q_table[state_key]
            # Among available actions, pick the one with highest Q
            best = max(available, key=lambda a: q_values[a])
            return best

    def update(self, state_key, action, reward, next_state_key,
               next_admissible=None):
        if next_admissible is None:
            next_available = list(range(self.n_skills))
        else:
            next_available = next_admissible if next_admissible else [action]

        current_q = self.q_table[state_key][action]
        next_q    = max(self.q_table[next_state_key][a] for a in next_available)
        target    = reward + self.gamma * next_q
        self.q_table[state_key][action] += self.lr * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon,
                           self.epsilon * self.epsilon_decay)

def simulate_student_response(state, skill_idx, skill_params,
                               skill_to_idx, idx_to_skill):
    skill_id = idx_to_skill[skill_idx]
    params   = skill_params.get(skill_id, DEFAULT_BKT_PARAMS)

    p_know   = state[skill_idx]
    p_slip   = params["p_slip"]
    p_guess  = params["p_guess"]

    if random.random() < p_know:
        correct = 1 if random.random() > p_slip else 0
    else:
        correct = 1 if random.random() < p_guess else 0

    new_state = state.copy()
    new_state[skill_idx] = bkt_update(p_know, correct, params)

    return correct, new_state


def train_agent(condition, student_states, skill_to_idx, idx_to_skill,
                skill_params, prereqs,
                n_episodes=2000, max_steps=30,
                use_phi=True, use_qa_reward=True):

    n_skills = len(skill_to_idx)
    agent    = QLearningAgent(n_skills)

    # Use actual student trajectories to initialize realistic starting states
    start_states = []
    for traj in student_states.values():
        if traj:
            start_states.append(traj[0]["state_before"].copy())

    print(f"\nTraining condition {condition} "
          f"(Phi={'ON' if use_phi else 'OFF'}, "
          f"QA-reward={'ON' if use_qa_reward else 'OFF'})...")

    episode_rewards  = []
    episode_mastered = []

    for episode in tqdm(range(n_episodes)):
        # Initialize: pick a random starting state from real data
        state = random.choice(start_states).copy()
        state_key = discretize_state(state)

        total_reward = 0.0
        bdo_history  = []

        for step in range(max_steps):
            # Get admissible actions (Phi constraint or not)
            if use_phi:
                admissible = build_phi(state, skill_to_idx, prereqs)
                if len(admissible) == 0:
                    break  # Student has mastered everything available
            else:
                admissible = None  # Unconstrained

            # Agent selects action
            action = agent.get_action(state_key, admissible)
            skill_id = idx_to_skill[action]

            # Compute Psi for this skill
            psi = compute_psi(state, skill_id, skill_to_idx, skill_params)

            # BDO outcome
            current_admissible = admissible if admissible else list(range(n_skills))
            bdo_outcome = check_bdo(psi, current_admissible, action)
            bdo_history.append(bdo_outcome)

            # Simulate student response
            correct, new_state = simulate_student_response(
                state, action, skill_params, skill_to_idx, idx_to_skill
            )

            # Compute reward
            if use_qa_reward:
                reward = reward_qa_coherence(
                    correct=correct,
                    psi=psi,
                    bdo_outcome=bdo_outcome,
                    trajectory_recent=bdo_history
                )
            else:
                reward = reward_baseline(correct=correct)

            # Next state
            next_state_key = discretize_state(new_state)
            next_admissible = (
                build_phi(new_state, skill_to_idx, prereqs)
                if use_phi else None
            )

            # Q-learning update
            agent.update(state_key, action, reward, next_state_key,
                         next_admissible)

            total_reward += reward
            state     = new_state
            state_key = next_state_key

        # Track progress
        n_mastered = int(np.sum(state >= MASTERY_THRESHOLD))
        episode_rewards.append(total_reward)
        episode_mastered.append(n_mastered)
        agent.decay_epsilon()

    print(f"  Final avg reward (last 100 ep): "
          f"{np.mean(episode_rewards[-100:]):.3f}")
    print(f"  Final avg skills mastered (last 100 ep): "
          f"{np.mean(episode_mastered[-100:]):.2f}")

    return agent, episode_rewards, episode_mastered

if __name__ == "__main__":
    with open("data/preprocessed.pkl", "rb") as f:
        data = pickle.load(f)
    with open("data/bkt_results.pkl", "rb") as f:
        bkt = pickle.load(f)
    with open("data/phi_results.pkl", "rb") as f:
        phi_data = pickle.load(f)

    skill_to_idx   = data["skill_to_idx"]
    idx_to_skill   = data["idx_to_skill"]
    student_states = bkt["student_states"]
    skill_params   = bkt["skill_params"]
    prereqs        = phi_data["prereqs"]

    conditions = {
        "A": {"use_phi": False, "use_qa_reward": False,
              "label": "Unconstrained + Baseline reward"},
        "B": {"use_phi": True,  "use_qa_reward": False,
              "label": "QA-constrained (Phi) + Baseline reward"},
        "C": {"use_phi": False, "use_qa_reward": True,
              "label": "Unconstrained + QA-coherence reward"},
        "D": {"use_phi": True,  "use_qa_reward": True,
              "label": "QA-constrained (Phi) + QA-coherence reward"},
    }

    results = {}
    for cond_name, cond_cfg in conditions.items():
        agent, rewards, mastered = train_agent(
            condition=cond_name,
            student_states=student_states,
            skill_to_idx=skill_to_idx,
            idx_to_skill=idx_to_skill,
            skill_params=skill_params,
            prereqs=prereqs,
            use_phi=cond_cfg["use_phi"],
            use_qa_reward=cond_cfg["use_qa_reward"],
            n_episodes=2000,
            max_steps=30,
        )
        results[cond_name] = {
            "agent":    agent,
            "rewards":  rewards,
            "mastered": mastered,
            "label":    cond_cfg["label"],
        }

    save_data = {
        k: {"rewards": v["rewards"], "mastered": v["mastered"],
            "label": v["label"]}
        for k, v in results.items()
    }
    with open("data/mdp_results.pkl", "wb") as f:
        pickle.dump(save_data, f)

    print("\nMDP results saved to data/mdp_results.pkl")
