# Сonstrained Markov Decision Process for adaptive learning path generation

A Quantum Autopoiesis (QA) constrained Markov Decision Process for adaptive learning path generation, evaluated on the ASSISTments 2009–2010 skill-builder dataset.

Every decision point in an adaptive learning pipeline is a _semantic collapse_ – a reduction from a continuous space of possibilities to a discrete, actionable distinction. The QA framework provides the semantics: Φ (deontic potential) constrains which actions are admissible, and Ψ (epistemic potential) describes the reward function toward the Zone of Proximal Development.

## Results

| Condition           | Φ constraint | QA reward | Avg skills mastered |
| ------------------- | ------------ | --------- | ------------------- |
| A — Baseline        | No           | No        | 3.88                |
| B — Φ only          | Yes          | No        | 4.57                |
| C — QA reward only  | No           | Yes       | 4.07                |
| **D — Full QA-MDP** | **Yes**      | **Yes**   | **5.84 (+50.6%)**   |

## Setup

```bash
pip install -r requirements.txt
```

**Dataset:** the [ASSISTments 2009–2010 skill-builder dataset](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010) and save it as `skill_builder_data.csv`.

## Usage

Run the steps in order from the `src/` directory:

```bash
cd src
python step_01_load_data.py
python step_02_bkt.py
python step_03_prerequisite_graph.py
python step_04_mdp.py
python step_05_evaluate.py
```

## File structure

```
├── data/                              # datasets and intermediate results
├── results/                           # output figures
├── src/
│   ├── step_01_load_data.py           # load and clean the dataset, compute opportunity counts
│   ├── step_02_bkt.py                 # fit Bayesian Knowledge Tracing per skill, compute knowledge state vectors
│   ├── step_03_prerequisite_graph.py  # infer prerequisite graph (Φ), compute Ψ and BDO
│   ├── step_04_mdp.py                 # Q-learning agent, four experimental conditions (A/B/C/D)
│   └── step_05_evaluate.py            # comparison table, learning curves, bar charts
├── requirements.txt
└── README.md
```
