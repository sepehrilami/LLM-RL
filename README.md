# Adaptive Information Modulation

**Designing governance mechanisms for multi-agent AI systems by separating the interaction network from the information network.**

This repository contains the framework, experiments, trained policies, and figure code for:

> Qiliang Chen, Sepehr Ilami, Nunzio Lore, and Babak Heydari.
> "Adaptive Information Modulation: Designing Governance Mechanisms for Multi-Agent Artificial Intelligence Systems."
> *ASME Journal of Mechanical Design*, 148(4): 041708, April 2026.
> [doi:10.1115/1.4070755](https://doi.org/10.1115/1.4070755) · Open access (CC-BY 4.0)

---

## The idea

Multi-agent systems implicitly assume information flows along the same pathways as direct interaction: agents observe and learn from those they engage with. That conflation collapses governance into a single lever — to change what agents know, you must rewire who they interact with. In engineered systems the interaction topology is frequently fixed by physics, hardware range, safety protocols, or organizational structure, which makes exactly that lever unavailable.

This framework separates the two layers:

- The **interaction network** `G = (N, E)` — who plays with whom — is fixed and exogenous.
- The **information network** — who observes what, about whom, at which timestep — is a designable, dynamically adaptable control surface.

A reinforcement learning **manager** governs the second layer. At each timestep it observes aggregated agent states and system-level metrics, then selects, per agent per interaction, which slice of contextual and historical information that agent's prompt will contain. Agents retain full decision autonomy; payoffs are untouched; no edge is ever added or removed. The paper calls this *governance through virtual topology reconfiguration*.

The mechanism is validated on a networked repeated Prisoner's Dilemma played by LLM agents, chosen because it isolates the tension between individual rationality and collective welfare while remaining analytically interpretable. The Prisoner's Dilemma is the testbed, not the contribution — the contribution is establishing information transparency as a *dynamic design parameter* that can be embedded into multi-agent system architecture at design time.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  RL Manager  (A2C actor–critic over a POMDP)             │
│  observes  o_t^M = h(aggregate agent states, system obs) │
│  emits     u_t^ij ∈ {LA, LA+AR, LA+NR}   per agent, per  │
│            interaction — decisions may be asymmetric     │
│  objective max E[ Σ_t γ^t W_t ],  W_t = Σ_(i,j)∈E r_ij   │
└────────────────────────┬─────────────────────────────────┘
                         │ selects information content
                         ▼        (prompt injection)
┌──────────────────────────────────────────────────────────┐
│  Strategic LLM Agents  (pretrained, not fine-tuned)      │
│  fixed Erdős–Rényi interaction network                   │
│  pairwise repeated Prisoner's Dilemma with all neighbors │
│  a_t^i = π^LLM(prompt_t^i) ∈ {C, D}                      │
└──────────────────────────────────────────────────────────┘
```

**Manager observation** (15-dim, per directed side of an interaction link): one-hot last action of self against this coplayer (3) ⊕ one-hot last action of the coplayer (3) ⊕ binned own cooperation ratio (3) ⊕ binned coplayer cooperation ratio (3) ⊕ binned neighborhood cooperation ratio (3). The "no prior history" state is encoded explicitly, so the manager can distinguish a fresh pair from a defecting one.

**Manager action space** — three nested information disclosure levels, deliberately coarse to keep training tractable and the learned policy interpretable:

| Action | Name | Information revealed to the agent |
| --- | --- | --- |
| `0` | **LA** | Last action pair between the agent and this coplayer |
| `1` | **LA+AR** | LA, plus long-run cooperation ratios of both the agent and its coplayer |
| `2` | **LA+NR** | LA, plus the agent's own ratio and the aggregate cooperation ratio of all its neighbors |

These parallel real disclosure regimes: `LA` is immediate dyadic feedback, `LA+AR` is a reputation system, `LA+NR` is a community-level norm signal.

**Manager reward.** Normalized joint payoff of the interacting pair, `((r_i + r_j) − 2) / 4`, mapping the four PD outcomes onto `[0, 1]`. Welfare, not individual return — the manager is a social planner over a fixed topology.

**Networks.** Payoffs follow the standard PD form: `CC → (3,3)`, `CD → (0,5)`, `DC → (5,0)`, `DD → (1,1)`. Cooperation ratios are discretized into three qualitative bins before entering a prompt — *rarely* (< 33%), *sometimes* (33–66%), *often* (> 66%) — because LLaMa3-70B responds inconsistently to raw numerical frequencies.

## Results

Averaged over 50 independent Erdős–Rényi network instances (`n = 20`, `p = 0.25`, 20 timesteps per round, LLaMa3-70B at temperature 0.8):

- The RL manager **outperforms every static baseline** (`LA`, `LA+AR`, `LA+NR`, and randomized disclosure) on both social welfare and cooperation rate, converging to near-universal mutual cooperation — `[C,C]` reaches 100% by roughly the 10th timestep, faster than any fixed policy.
- **Uniform reputation disclosure backfires.** `LA+AR` — showing each agent both parties' full cooperation histories — produces the *lowest* welfare and cooperation of any condition. Extending memory of past behavior makes agents less forgiving and more willing to defect against a coplayer with a blemished record. More information is not monotonically better; this is the central negative result motivating adaptive control.
- **The learned policy is two-phase.** In steps 2–6 the manager mixes `LA` and `LA+NR`, including asymmetric assignments within a single interaction link (~20% of interventions), to break early mutual-defection deadlocks. From step 7 onward it converges on `LA+NR` for both agents, maintaining cooperation once it is established. `LA+AR` is never selected at test time.
- **The manager discovers network structure it was never told.** It has no access to the topology, yet high-degree agents receive neighborhood information significantly more often than low-degree agents (mean degree 5.68 vs. 4.64, *p* < 0.001, Cohen's *d* = 0.60) — routing network-level signal to nodes that can amplify it.
- **It also conditions on behavioral history.** Agents receiving `LA+NR` had substantially higher pre-intervention cooperation rates than those receiving `LA` (0.908 vs. 0.801, *p* < 0.001, Cohen's *d* = 2.61). The manager gives simple dyadic feedback to less-cooperative agents as a remedial strategy, and community norm signals to already-cooperative ones.

**Microlevel validation.** Before any system-level claim, agent behavior is validated in isolation (`micro_validation.py`): prompts are sampled ≥ 100 times per condition and the empirical cooperation frequency is treated as the agent's policy. LLaMa3-70B reproduces a **win-stay, lose-shift** pattern — reciprocating cooperation, defecting after exploitation, continuing to exploit when successful — and responds to described coplayer cooperation levels in the expected direction (87% and 99% cooperation following mutual cooperation when the coplayer is described as cooperating *sometimes* and *often*). This establishes that the agents are strategically coherent enough for information modulation to be a meaningful governance lever rather than prompt noise.

## Repository layout

| Path | Contents |
| --- | --- |
| [SI/](SI/) | **Frozen replication package for the published article.** Training and evaluation scripts, trained manager checkpoints, prompt templates, raw output arrays, and figure code as used for the paper. Start here to reproduce published results. |
| [environment_simplified/](environment_simplified/) | Active development version of the same framework — superset of `SI/main_code`. Adds the randomized-disclosure baseline, an all-defect stress test, and runs seeding a subset of nodes as deterministic non-LLM agents. |
| [create_figures/](create_figures/) | Figure generation for the main paper and the 2025 revision, plus intervention-matrix visualization. |
| [environment_built_new/](environment_built_new/) | Earlier, more general "Orbit" scaffolding: declarative agent families defined in JSON, arbitrary action types, and *dynamic* network rewiring (`change_network`). Exploratory; not used for the published results. |
| [sepehr/](sepehr/) | Ad-hoc analysis notebooks and intermediate result CSVs. |

Core modules, in both `SI/main_code/` and `environment_simplified/`:

| File | Role |
| --- | --- |
| `rl_train_networks.py` | Trains the A2C manager. Each episode draws a fresh Erdős–Rényi graph; steps 0–14 use fixed prompts to build interaction history, and the manager acts and learns from step 15 onward. Checkpoints every 5 episodes. |
| `rl_test_networks.py` | Main evaluation harness. Set `intervention_type` to one of `RL`, `last_action`, `agent_ratio`, `network_ratio`, `randomized` and run. Records per-step rewards, cooperation ratios, pair-action counts, per-agent intervention assignments, and the adjacency matrix. |
| `micro_validation.py` | Single-agent behavioral validation across prompt variants. |
| `agent.py` | `LLMAgent` (prompt → `C`/`D` via a provider) and `A2C_manager` (actor–critic, training step, checkpoint I/O). |
| `utils/model.py` | `ActorCritic` — separate actor and critic heads, one hidden layer of 256 units each. |
| `utils/provider.py` | LLM backends: Groq, Ollama, and local HuggingFace with 4-bit NF4 quantization. Retries on malformed output up to 20 times. |
| `utils/utils.py` | Action encoding, cooperation-ratio binning, PD payoff functions. |
| `settings/prompts.json` | All prompt templates, including the four microvalidation variants. |
| `save_model/` | Trained manager checkpoints at episodes 5, 10, 15, 20, 25. |

## Setup

```bash
git clone <this-repo>
cd LLM-RL/SI
pip install -r requirements.txt
```

Reference environment: Python 3.11, PyTorch 2.3.1, NetworkX 3.3, LangChain 0.2.5. Training and evaluation for the paper were run on Northeastern University's high-performance cluster.

**Provide your own API credentials.** `agent.py` reads the model and key from `settings/env_settings.json`:

```json
{
    "provider": "groq",
    "model": "llama3-70b-8192",
    "temperature": 0.8,
    "api_key": "YOUR_GROQ_API_KEY",
    "memory_length": 0
}
```

Set `"provider": "ollama"` with `"api_key": null` to run a local model instead, or `"provider": "hf"` with a HuggingFace token for a local quantized checkpoint. Keep this file out of version control — add `settings/env_settings.json` to `.gitignore` before committing.

## Reproducing the experiments

```bash
cd SI/main_code

# Evaluate a governance policy over the networked PD.
# Edit `intervention_type` and the parameter block near the bottom of the file:
#     num_agent = 20, steps = 20, rounds = 10, edge_prob = 0.25
python rl_test_networks.py

# Retrain the manager from scratch (optional — checkpoints are provided).
python rl_train_networks.py

# Validate agent behavior at the individual level.
python micro_validation.py
```

Evaluation loads `save_model/manager_networks_20nodes_25` by default. Results are written under `outputs/<intervention_type>/` as `.npy` arrays (cooperation ratios, last-action matrices, intervention assignments, adjacency matrices) plus a running `rewards_*.txt` log.

Figures are produced by [SI/create_figures/visualizer.ipynb](SI/create_figures/visualizer.ipynb) and [SI/create_figures/plot_evolution_pairactions.py](SI/create_figures/plot_evolution_pairactions.py); the revised versions live in [create_figures/](create_figures/).

Each evaluation round issues one LLM call per agent per edge per timestep. At `n = 20`, `p = 0.25`, 20 steps, that is on the order of 20,000 calls per condition per 10 rounds — budget accordingly.

## Scope and limitations

Stated plainly, following the paper:

- **Sample size.** 50 network instances at 20 timesteps each, bounded by compute. Confidence intervals suggest more samples are unlikely to overturn the ordering of methods, but this is a proof of concept, not a saturated estimate.
- **One game, one topology family.** Validation is restricted to the Prisoner's Dilemma on random graphs. Scale-free, small-world, and hierarchical structures, and other strategic games (public goods, coordination, resource allocation), are untested.
- **LLM agents are proxies, not humans.** They exhibit coherent strategic behavior, but transferring conclusions to human decision-makers requires controlled human-subject comparison under identical information conditions.
- **Coarse action space.** Three predefined disclosure levels. Finer granularity, continuous metrics, dynamically redefined neighborhoods, and hybrid discrete–continuous schemes are all plausible generalizations at higher training cost.
- **Full observability assumed.** The manager sees the true system state. Partial observability, measurement noise, and adversarial agents are open problems.

## Deployment shape

The manager's policy is trained **offline, at design time**, against a system model — historical data, physics simulation, or synthetic agents like the LLM agents used here. Once trained it deploys as a *feedforward* software layer: it reads system state, computes information routing through the trained network, and filters what each agent sees, with no parameter updates during operation. Periodic retraining on logged operational data handles distributional drift.

This applies wherever interaction topology is constrained but information flow remains designable — collaborative robotics on assembly lines, autonomous vehicle platoons under V2V bandwidth limits, distributed energy management with consumer-facing demand forecasts. Practitioners instantiating the framework must specify information granularity, state representation, update frequency, and a validation approach. Ethical considerations are not optional here: agents should be informed that adaptive information governance is in effect, and disclosure policies should be audited for systematic disadvantaging of any participant.

## Citation

```bibtex
@article{chen2026adaptive,
  title   = {Adaptive Information Modulation: Designing Governance Mechanisms
             for Multi-Agent Artificial Intelligence Systems},
  author  = {Chen, Qiliang and Ilami, Sepehr and Lore, Nunzio and Heydari, Babak},
  journal = {Journal of Mechanical Design},
  volume  = {148},
  number  = {4},
  pages   = {041708},
  year    = {2026},
  doi     = {10.1115/1.4070755}
}
```

## Related work from this group

- Lore, Ilami, Heydari. *Transferring Theory of Mind to Small Language Models via LoRA fine-tuning* — [arXiv:2408.05241](https://arxiv.org/abs/2408.05241). Cheap transfer of strategic reasoning from large to small models, one route past the compute cost noted above.

## License

Code released under the [MIT License](LICENSE). The article is published open access under CC-BY 4.0 by ASME.

Authors are with the Department of Mechanical and Industrial Engineering and the Institute for Experiential AI / Network Science Institute, Northeastern University, Boston, MA. Correspondence: `b.heydari@northeastern.edu`.
