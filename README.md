# VLM Road Crash — CrashGraph

Evidence-grounded causal crash-video understanding for NeurIPS-oriented research.

**Active plan:** [CRASHGRAPH_NEURIPS_PLAN.md](CRASHGRAPH_NEURIPS_PLAN.md)  
**Prior summarization work:** [PROJECT_COMPLETION_SUMMARY.md](PROJECT_COMPLETION_SUMMARY.md)  
**Superseded as main task (diagnostics only):** [EVIDENCE_CONDITIONED_FORECASTING_PROPOSAL.md](EVIDENCE_CONDITIONED_FORECASTING_PROPOSAL.md)

## Task in one sentence

Given a **complete** crash video, produce a structured report where every claim is temporally grounded, participant-linked, epistemically labeled, and withdrawn when its supporting frames are removed.

---

## Mathematical formulation

> GitHub renders math with `$...$` (inline) and `$$...$$` (display). All formulae below use that syntax.

### 1. Video and phase partition

A crash clip is a sequence of frames:

$$
V = \{x_1, x_2, \ldots, x_T\}
$$

Using annotated crash start/end times, we partition:

$$
V = V_{\mathrm{pre}} \cup V_{\mathrm{impact}} \cup V_{\mathrm{post}}
$$

where $V_{\mathrm{pre}}$ is the pre-collision segment, $V_{\mathrm{impact}}$ the collision window, and $V_{\mathrm{post}}$ the aftermath. This split is required for the **Hindsight Attribution Gap** (Section 8).

### 2. Structured crash report

The model outputs a structured report:

$$
R = (P, E, G, U, S)
$$

| Symbol | Meaning |
|--------|---------|
| $P$ | set of participants (vehicles / VRUs) |
| $E$ | set of atomic events with time intervals |
| $G$ | causal event graph over $E$ |
| $U$ | uncertain / unanswerable claims |
| $S$ | final natural-language summary |

### 3. Participants

$$
P = \{p_1, p_2, \ldots, p_n\}
$$

Each participant is a tuple:

$$
p_i = (\mathrm{type}_i,\; \mathrm{colour}_i,\; \mathrm{camera\_role}_i,\; \mathrm{impact\_zone}_i)
$$

**Example:** $p_2 = (\text{car},\; \text{black},\; \text{oncoming},\; \text{front-left})$.

### 4. Atomic events

$$
E = \{e_1, e_2, \ldots, e_m\}
$$

Each event is:

$$
e_j = (\mathrm{subject}_j,\; \mathrm{action}_j,\; \mathrm{object}_j,\; [t_j^{s}, t_j^{e}],\; \mathrm{evidence}_j)
$$

where $[t_j^{s}, t_j^{e}]$ is the supporting time interval (in seconds or frame indices).

**Example:**

$$
e_1 = (\mathrm{vehicle}_2,\; \mathrm{enters},\; \mathrm{main\_road},\; [1.1, 2.3])
$$

$$
e_2 = (\mathrm{vehicle}_1,\; \mathrm{continues},\; \mathrm{straight},\; [0.0, 2.7])
$$

$$
e_3 = (\{\mathrm{vehicle}_1, \mathrm{vehicle}_2\},\; \mathrm{collide},\; \varnothing,\; [2.7, 3.1])
$$

### 5. Causal event graph

$$
G = (E, A)
$$

$A$ is the set of directed relations between events. For a pair $(e_i, e_j)$:

$$
A_{ij} \in \{\mathrm{before},\; \mathrm{overlaps},\; \mathrm{enables},\; \mathrm{contributes\_to},\; \mathrm{causes},\; \mathrm{contradicts}\}
$$

A predicted edge score (method component) is:

$$
a_{ij} = \sigma\!\Big(\mathrm{MLP}\big[h_i,\; h_j,\; \Delta t_{ij},\; O_i,\; O_j\big]\Big)
$$

where $h_i, h_j$ are event embeddings, $\Delta t_{ij}$ is the temporal offset, $O_i, O_j$ are participant sets, and $\sigma$ is the sigmoid. **No causal edge is forced when evidence is insufficient.**

### 6. Temporal Evidence Contract (TEC) — core novelty

Every generated claim must satisfy a contract:

$$
c_k = \big(s_k,\; e_k,\; O_k,\; r_k,\; q_k\big)
$$

| Symbol | Definition |
|--------|------------|
| $s_k$ | natural-language claim text |
| $e_k = [t_k^{s}, t_k^{e}]$ | supporting video interval (or $\varnothing$ if none) |
| $O_k$ | involved participant / object IDs |
| $r_k$ | epistemic status |
| $q_k \in [0,1]$ | confidence |

Epistemic status:

$$
r_k \in \{\mathrm{observed},\; \mathrm{derived},\; \mathrm{hypothesised},\; \mathrm{undetermined}\}
$$

**Interpretation:**

- $\mathrm{observed}$: directly visible in $e_k$
- $\mathrm{derived}$: logically follows from observed events (e.g. trajectory conflict)
- $\mathrm{hypothesised}$: plausible but not visually confirmed
- $\mathrm{undetermined}$: evidence absent (e.g. traffic light not in view)

A report is accepted only if each claim either (i) passes its evidence contract or (ii) is explicitly labeled $\mathrm{undetermined}$ / $\mathrm{hypothesised}$ with low $q_k$.

### 7. Evidence support score and revision

For claim $c_k$ with cited interval $E_k = V[t_k^{s}:t_k^{e}]$:

$$
v_k = F(c_k,\; E_k,\; O_k)
$$

where $F$ is the contract verifier (alignment of claim text with frames, participant consistency, contradiction checks). Decision rule:

$$
\begin{cases}
\text{keep / accept} & \text{if } v_k \ge \gamma \\
\text{revise / abstain / mark undetermined} & \text{if } v_k < \gamma
\end{cases}
$$

with threshold $\gamma \in (0,1)$.

### 8. Evidence interventions (strongest evaluation)

Let $E_k = V[t_k^{s}:t_k^{e}]$ be the frames cited by claim $c_k$, and $q(c_k \mid V)$ the model confidence for $c_k$ given video $V$.

#### Intervention A — remove supporting evidence

$$
V_{-k} = V \setminus E_k
$$

A faithful model should satisfy:

$$
q(c_k \mid V) > q(c_k \mid V_{-k})
$$

#### Intervention B — keep only supporting evidence

$$
V_{+k} = E_k
$$

The claim should remain reasonably recoverable:

$$
q(c_k \mid V_{+k}) \approx q(c_k \mid V)
$$

#### Intervention C — remove irrelevant evidence

Let $E_{\mathrm{irr}}$ be frames unrelated to $c_k$:

$$
V_{\mathrm{irr}} = V \setminus E_{\mathrm{irr}}
$$

Confidence should stay stable:

$$
q(c_k \mid V) \approx q(c_k \mid V_{\mathrm{irr}})
$$

### 9. Hindsight Attribution Gap (HAG)

For a pre-crash causal claim $c$ annotated as visible in $V_{\mathrm{pre}}$:

$$
\mathrm{HAG}(c) = q(c \mid V_{\mathrm{full}}) - q(c \mid V_{\mathrm{pre}})
$$

A **large positive** $\mathrm{HAG}(c)$ means the model asserts a pre-crash cause mainly after seeing the outcome (post-impact leakage). Compute HAG **only** for claims labeled as establishable before impact.

### 10. Primary evaluation metrics

Assume $K$ evaluated claims.

#### Claim Evidence Precision / Recall / F1

$$
\mathrm{CEP} = \frac{\#\{\text{generated claims supported by annotated evidence}\}}{\#\{\text{generated claims}\}}
$$

$$
\mathrm{CER} = \frac{\#\{\text{GT evidence-grounded claims recovered with evidence}\}}{\#\{\text{GT evidence-grounded claims}\}}
$$

$$
\mathrm{ECF1} = \frac{2 \cdot \mathrm{CEP} \cdot \mathrm{CER}}{\mathrm{CEP} + \mathrm{CER}}
$$

(also written **EC-F1** in tables)

#### Evidence Removal Sensitivity (higher is better for supported claims)

$$
\mathrm{ERS} = \frac{1}{K} \sum_{k=1}^{K} \Big[ q(c_k \mid V) - q(c_k \mid V_{-k}) \Big]
$$

#### Irrelevant Removal Stability (higher is better)

$$
\mathrm{IRS} = 1 - \frac{1}{K} \sum_{k=1}^{K} \big| q(c_k \mid V) - q(c_k \mid V_{\mathrm{irr}}) \big|
$$

#### Temporal Evidence IoU

$$
\mathrm{TEIoU} = \frac{\big| E_{\mathrm{pred}} \cap E_{\mathrm{gt}} \big|}{\big| E_{\mathrm{pred}} \cup E_{\mathrm{gt}} \big|}
$$

(also written **TE-IoU**)

#### Causal Graph F1

$$
\mathrm{CGF1} = F_1(A_{\mathrm{pred}},\; A_{\mathrm{gt}})
$$

(also written **CG-F1**)

Report separately for $\mathrm{before}$, $\mathrm{enables}$, $\mathrm{contributes\_to}$, $\mathrm{causes}$, $\mathrm{contradicts}$.

#### Unsupported Causal Claim Rate (lower is better)

$$
\mathrm{UCCR} = \frac{\#\{\text{unsupported causal claims}\}}{\#\{\text{generated causal claims}\}}
$$

#### Report Consistency Score

$$
\mathrm{RCS} = 1 - \frac{N_{\mathrm{contradictions}}}{N_{\mathrm{claims}}}
$$

#### Epistemic Status Macro-F1

Macro-$F_1$ over $\{\mathrm{observed},\; \mathrm{derived},\; \mathrm{hypothesised},\; \mathrm{undetermined}\}$.

### 11. Optional learning objectives (ablation only; not the headline method)

If a learned verifier / preference stage is added:

**Event localisation**

$$
\mathcal{L}_{\mathrm{event}} = \mathcal{L}_{\mathrm{boundary}} + \lambda_{\mathrm{cls}} \mathcal{L}_{\mathrm{event\_type}}
$$

**Causal edge (binary cross-entropy)**

$$
\mathcal{L}_{\mathrm{causal}} = -\sum_{i,j} \Big[ y_{ij}\log a_{ij} + (1-y_{ij})\log(1-a_{ij}) \Big]
$$

**Report generation**

$$
\mathcal{L}_{\mathrm{gen}} = -\sum_{\ell} \log P(w_{\ell} \mid w_{<\ell},\; V,\; G)
$$

**Evidence alignment** (claim–frame relevance $s_{k,t}$)

$$
\mathcal{L}_{\mathrm{evid}} = -\sum_{k,t} \Big[ y_{k,t}\log s_{k,t} + (1-y_{k,t})\log(1-s_{k,t}) \Big]
$$

**Intervention hinge** (enforce confidence drop under $V_{-k}$)

$$
\mathcal{L}_{\mathrm{int}} = \sum_{k} \max\!\Big(0,\; m - q(c_k \mid V) + q(c_k \mid V_{-k})\Big)
$$

**Consistency** over incompatible claim pairs $\mathcal{C}$

$$
\mathcal{L}_{\mathrm{con}} = \sum_{(i,j)\in\mathcal{C}} P(c_i \mid V)\, P(c_j \mid V)
$$

**Joint objective**

$$
\mathcal{L} = \mathcal{L}_{\mathrm{event}} + \lambda_1\mathcal{L}_{\mathrm{causal}} + \lambda_2\mathcal{L}_{\mathrm{gen}} + \lambda_3\mathcal{L}_{\mathrm{evid}} + \lambda_4\mathcal{L}_{\mathrm{int}} + \lambda_5\mathcal{L}_{\mathrm{con}}
$$

> **Paper priority:** training-free **CrashGraph-Verify** (Generate–Verify–Revise). The losses above are optional secondary ablations — not classic caption LoRA.

### 12. Video / track notation (method internals)

Video encoder:

$$
z_{1:T} = E_v(V)
$$

Participant track for agent $i$:

$$
\tau_i = \big\{ b_{i,t},\; c_i,\; \Delta x_{i,t},\; \Delta y_{i,t} \big\}_{t=1}^{T}
$$

where $b_{i,t}$ is the box, $c_i$ appearance/class, and $(\Delta x_{i,t}, \Delta y_{i,t})$ motion.

Event proposal:

$$
e_j = (t_j^{s},\; t_j^{e},\; h_j)
$$

---

## Dataset

| Item | Path / note |
|------|-------------|
| Videos | `video1500/` — 1500 MP4s (~5 s @ 10 fps) |
| Ground truth | `Car_Crash_Text_Dataset_ground_truth.xlsx` |
| Processed annotations | `data/processed/annotations.json` |
| Video lineage | CCD crash subset ([Bao et al., ACM MM 2020](https://arxiv.org/abs/2008.00334)) |

## What we will do next

See the top of [CRASHGRAPH_NEURIPS_PLAN.md](CRASHGRAPH_NEURIPS_PLAN.md): TECs, Crash-EC annotation, CrashGraph-Verify, multi-VLM benchmark, intervention metrics.

## License / remote

Repository: https://github.com/harshalDharpure/vlm-road-crash
