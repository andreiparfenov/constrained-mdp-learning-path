import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

path = Path("skill_builder_data.csv")


def load_data(path=path):
    df = pd.read_csv(path, encoding="ISO-8859-15", low_memory=False)
    print(f"  Raw shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    if "skill_id" not in df.columns and "list_skill_ids" in df.columns:
        df = df.rename(columns={
            "list_skill_ids": "skill_id",
            "list_skills":    "skill_name",
        })

        df["skill_id"]   = df["skill_id"].astype(str).str.split(";").str[0]
        df["skill_name"] = df["skill_name"].astype(str).str.split(";").str[0]
        df["skill_id"]   = df["skill_id"].replace("nan", np.nan)
        df["skill_name"] = df["skill_name"].replace("nan", np.nan)

    cols = ["order_id", "user_id", "skill_id", "skill_name", "correct"]
    df = df[cols].copy()

    df = df.dropna(subset=["skill_id", "skill_name"])
    df["skill_id"] = df["skill_id"].astype(float).astype(int)
    df["correct"]  = df["correct"].astype(int)

    df = df.sort_values("order_id").reset_index(drop=True)

    df["opportunity"] = df.groupby(["user_id", "skill_id"]).cumcount()

    print(f"  After cleaning: {df.shape}")
    print(f"  Students: {df['user_id'].nunique()}")
    print(f"  Unique skills: {df['skill_id'].nunique()}")
    print(f"  Total interactions: {len(df)}")
    print(f"  Overall accuracy: {df['correct'].mean():.3f}")
    return df


def explore_data(df):
    print("\nDataset overview")

    # Interactions per student
    per_student = df.groupby("user_id").size()
    print(f"\nInteractions per student:")
    print(f"  Mean:   {per_student.mean():.1f}")
    print(f"  Median: {per_student.median():.1f}")
    print(f"  Min:    {per_student.min()}")
    print(f"  Max:    {per_student.max()}")

    # Skills per student
    skills_per_student = df.groupby("user_id")["skill_id"].nunique()
    print(f"\nDistinct skills practiced per student:")
    print(f"  Mean:   {skills_per_student.mean():.1f}")
    print(f"  Median: {skills_per_student.median():.1f}")

    # Most common skills
    print(f"\nTop 10 most practiced skills:")
    top_skills = (
        df.groupby("skill_name")
        .agg(count=("correct", "count"), accuracy=("correct", "mean"))
        .sort_values("count", ascending=False)
        .head(10)
    )
    print(top_skills.to_string())

    return per_student, skills_per_student


def build_student_sequences(df):
    sequences = {}
    for user_id, group in df.groupby("user_id"):
        group = group.sort_values("order_id")
        seq = list(zip(group["skill_id"].tolist(), group["correct"].tolist()))
        sequences[user_id] = seq
    print(f"\nBuilt {len(sequences)} student sequences.")
    return sequences


def filter_students(sequences, min_interactions=10, min_skills=3):
    # Keep only students with enough data to be useful for MDP training
    filtered = {
        uid: seq
        for uid, seq in sequences.items()
        if len(seq) >= min_interactions
        and len(set(s for s, _ in seq)) >= min_skills
    }
    print(
        f"After filtering (>={min_interactions} interactions, "
        f">={min_skills} skills): {len(filtered)} students remain."
    )
    return filtered


def get_skill_index(df):
    unique_skills = sorted(df["skill_id"].unique())
    skill_to_idx = {sid: i for i, sid in enumerate(unique_skills)}
    idx_to_skill = {i: sid for sid, i in skill_to_idx.items()}

    skill_names = df.drop_duplicates("skill_id").set_index("skill_id")["skill_name"].to_dict()

    print(f"\nSkill index built: {len(skill_to_idx)} skills")
    return skill_to_idx, idx_to_skill, skill_names


def plot_basic_stats(df, save_path="results/data_overview.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Interactions per student
    per_student = df.groupby("user_id").size()
    axes[0].hist(per_student, bins=50, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Interactions per student")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of interactions per student")

    # Accuracy per skill (top 30 skills by frequency)
    top30 = df["skill_name"].value_counts().head(30).index
    skill_acc = df[df["skill_name"].isin(top30)].groupby("skill_name")["correct"].mean()
    skill_acc = skill_acc.sort_values()
    axes[1].barh(range(len(skill_acc)), skill_acc.values, color="steelblue")
    axes[1].set_yticks(range(len(skill_acc)))
    axes[1].set_yticklabels(skill_acc.index, fontsize=6)
    axes[1].set_xlabel("Accuracy")
    axes[1].set_title("Accuracy per skill (top 30)")

    # Opportunity vs accuracy (learning curve)
    opp_acc = (
        df[df["opportunity"] <= 20]
        .groupby("opportunity")["correct"]
        .mean()
    )
    axes[2].plot(opp_acc.index, opp_acc.values, "o-", color="steelblue")
    axes[2].set_xlabel("Opportunity (practice count)")
    axes[2].set_ylabel("Average accuracy")
    axes[2].set_title("Learning curve: accuracy vs. practice")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    df = load_data()
    per_student, skills_per_student = explore_data(df)
    sequences = build_student_sequences(df)
    sequences = filter_students(sequences)
    skill_to_idx, idx_to_skill, skill_names = get_skill_index(df)
    plot_basic_stats(df)

    import pickle
    with open("data/preprocessed.pkl", "wb") as f:
        pickle.dump({
            "df": df,
            "sequences": sequences,
            "skill_to_idx": skill_to_idx,
            "idx_to_skill": idx_to_skill,
            "skill_names": skill_names,
        }, f)
    print("\nsaved to data/preprocessed.pkl")
