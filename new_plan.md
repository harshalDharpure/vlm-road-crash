From Video to Verifiable Crash Reports: NeurIPS Research Plan

Verdict (what will get accepted)

Do not submit another “we caption crashes / we ask VQA / we fine-tune a VLM” paper. Your completed LLaVA-NeXT work already shows the trap: LoRA improves BLEU/ROUGE but hurts NLI faithfulness (~66% → ~34%). That finding becomes the motivating failure mode, not the contribution.

Submit a general video-language problem instantiated on crashes:



How do we evaluate and constrain video-language models so that each causal claim in an explanation is temporally grounded, participant-linked, epistemically labeled, and withdrawn when its cited evidence is removed?

Traffic crashes are the high-stakes domain. The transferable object is the Temporal Evidence Contract (TEC) + Evidence Intervention Protocol.

Recommended title: From Video to Verifiable Crash Reports: Evidence-Grounded Causal Reasoning for Traffic Accidents

Alternative: CrashGraph: Temporal Evidence Contracts for Explainable Traffic-Accident Video Understanding



What you already have (assets to reuse)





Videos: [video1500/](video1500/) — 1500 MP4s, ~5s @ 10fps, IDs 000001–001500



GT Excel: [Car_Crash_Text_Dataset_ground_truth.xlsx](Car_Crash_Text_Dataset_ground_truth.xlsx) — 11 fields (severity, vehicles, impact, crash start/end, explanation, ambiguity, camera, weather)



Processed: [data/processed/annotations.json](data/processed/annotations.json) — 1498 matched; splits 1048/224/226



Completed pipeline: LLaVA-NeXT zero-shot + LoRA + ablations + NLI metrics ([PROJECT_COMPLETION_SUMMARY.md](PROJECT_COMPLETION_SUMMARY.md))



Seed for epistemic labels: ~103 non-empty Ambiguity notes (“unclear traffic signals”, color uncertainty, etc.)



Drop: anticipation / TTA / early-warning from [EVIDENCE_CONDITIONED_FORECASTING_PROPOSAL.md](EVIDENCE_CONDITIONED_FORECASTING_PROPOSAL.md) — keep only source-control and temporal-shuffle as diagnostics



Literature map: papers → limitations → how we solve them

This section is the Related Work backbone of the paper. For each work: dataset, method, limitation, how CrashGraph / TECs fix it. Then: which papers we follow as design references for the plan above.

A. Accident / safety video benchmarks (direct competitors)

1. VRU-Accident — Kim et al., ICCV Workshops 2025





Paper / link: arXiv:2507.09815, GitHub



Dataset: 1,000 real dashcam VRU–vehicle accident videos; 6,000 MCQ (6 categories: weather, environment, road config, accident type, cause, prevention); 1,000 dense captions; 24K candidate options



Method: Zero-shot / API eval of 17 MLLMs on MCQ accuracy + dense caption metrics; focuses on VRU safety-critical reasoning



Limitation: Judges whether the final option / caption is correct; does not check which frames support each causal claim, does not remove cited evidence, no epistemic abstention, no causal event graph



How we solve it: Replace answer-only scoring with Temporal Evidence Contracts + Evidence Intervention (ERS/IRS) + undetermined claims when signals/causes are not visible

2. CrashSight — Gan et al., CVPR Workshops 2026 (DriveX)





Paper / link: arXiv:2604.08457, project



Dataset: 250 roadside/infrastructure crash videos; 13K MCQ under 2-tier taxonomy (grounding + forensic reasoning); phase-aware dense captions (pre / impact / post)



Method: Benchmark 8 VLMs; phase-preserving annotation pipeline (VLM draft → expert refine → LLM VQA → verify); reports hallucination/robustness probes; domain FT gains (~+13–16%)



Limitation: Still primarily MCQ correctness; phase captions help structure but do not bind each free-form claim to an interval; no remove-supporting-frames test; no participant–event causal graph for reports



How we solve it: Keep their phase idea (pre/impact/post) but upgrade to claim-level TECs; add intervention audits and causal graph G=(E,A); use full crash report task, not only MCQ

3. AUTOPILOT-VQA — CVPR 2026 competition track





Paper / link: arXiv:2607.08745



Dataset: 600+ dashcam clips (collision / near-miss / no-incident); 6,000+ QA across environment, entities, fault, impact, avoidability



Method: Fixed categorical VQA; community leaderboard (~0.66 top accuracy)



Limitation: Closed answer set; no free-form verifiable explanation; no evidence intervals; no epistemic status



How we solve it: Open structured report R=(P,E,G,U,S) with contract verification; fault/cause claims must be observed|derived|hypothesised|undetermined

4. AccidentBench — Gu et al., 2025 (OpenReview / arXiv:2509.26636)





Paper / link: arXiv:2509.26636, site



Dataset: ~2,000 videos; ~19,000 QA; land (accidents) + air + water; short/medium/long; easy/medium/hard; temporal / spatial / intent



Method: Large multimodal reasoning benchmark; even Gemini-2.5 / GPT-5 ~18% on hardest long videos



Limitation: Exposes that models fail, but still scores answer accuracy; does not verify evidence use or hindsight



How we solve it: Borrow cross-domain generality idea (their air/water) as our small TEC transfer (falls / workplace / surveillance); keep evaluation on evidence faithfulness, not only accuracy

5. SeeUnsafe — Zhang et al., Accident Analysis & Prevention 2025





Paper / link: GitHub ai4ce/SeeUnsafe



Dataset: Primarily Toyota Woven Traffic Safety (WTS) demos



Method: Frame augmentation + tracking/segmentation/visual prompts + MLLM for identification, reasoning, grounding, severity



Limitation: Object/box grounding and severity aggregation; not claim-level causal contracts; limited as a NeurIPS-style benchmark contribution



How we solve it: Use tracking/prompts as modules inside CrashGraph-Verify, but score claims, not boxes alone

B. Dataset lineage for YOUR videos

6. CCD / UString — Bao, Yu, Kong, ACM MM 2020





Paper / link: arXiv:2008.00334, CCD GitHub



Dataset: Car Crash Dataset (CCD) — 1,500 YouTube dashcam crash clips (50 frames @ 10 fps) + 3,000 BDD normal clips; weather, ego-involve, timing, accident-reason text



Method: Accident anticipation with spatio-temporal relational learning + uncertainty



Limitation: Task is early warning (TTA/AP), not verifiable post-hoc explanation; reason text is coarse; crash/normal source bias (YouTube vs BDD)



How we use it: Your [video1500/](video1500/) + Excel explanations are the CCD crash subset + richer text GT. Mentor correctly dropped anticipation. Keep CCD citation for video origin; keep source-control only as a small diagnostic, not the headline

C. Evidence / causal / motion papers (NeurIPS-level references we FOLLOW and DIFFERENTIATE)

7. VER — “When Thinking Drifts: Evidential Grounding for Robust Video Reasoning” (NeurIPS 2025)





Paper / link: NeurIPS 2025 PDF



Dataset / setup: 10 general video understanding benchmarks



Method: Diagnoses “visual thinking drift” in CoT; Visual Evidence Reward (VER) RL to ground reasoning traces



Limitation: General CoT reward; not structured forensic reports; no crash causal graphs; no explicit remove-cited-interval interventions



How we follow: Adopt the NeurIPS framing that fluent reasoning ≠ grounded reasoning



How we differ: TECs + physical frame interventions + epistemic labels + crash causal graphs (training-free GVR first, not RL CoT)

8. CausalVTG — NeurIPS 2025





Paper / link: OpenReview, PDF



Dataset: Standard VTG benchmarks (moment localization)



Method: Front-door adjustment + counterfactual contrastive learning; reject ungrounded queries



Limitation: Single-query temporal grounding, not multi-claim crash reports



How we follow: Counterfactual / remove-evidence spirit → our Interventions A/B/C



How we differ: Multi-claim contracts, causal edges between events, HAG for hindsight

9. TRACE — “Temporal Grounding Video LLM via Causal Event Modeling” (arXiv:2410.05643)





Paper / link: arXiv:2410.05643



Dataset: VTG datasets



Method: Interleaved causal events = timestamps + saliency + captions



Limitation: Event tokens for localization/captioning; no epistemic status; no intervention eval for crash forensics



How we follow: Structured atomic events with intervals as first-class objects



How we differ: Add participants, relation types (causes, contradicts, …), undetermined claims, intervention metrics

10. MotionBench — CVPR 2025





Paper / link: arXiv:2501.02955, site



Dataset: 8,052 motion MCQs; fine-grained motion categories



Method: Shows VLMs are weak at fine-grained motion; TE Fusion architecture



Limitation: Motion MCQ, not causal crash explanation



How we follow: Motivation that fine-grained temporal evidence is hard → justifies explicit TEC intervals on short 5s crash clips

D. What is NOT novel alone (do not claim these as contributions)

VLM crash captioning, accident VQA, dense description, tracking, scene graphs, severity prediction, asking “why did the crash happen?”, coloured boxes alone.

E. Shared limitation → our NeurIPS problem

Across VRU-Accident, CrashSight, AUTOPILOT-VQA, AccidentBench, and caption papers:

They score whether the answer looks right. They do not force every causal claim to be temporally grounded, participant-linked, epistemically honest, and sensitive to removing its cited frames.

Your own LLaVA LoRA result (BLEU↑, NLI faithfulness↓) is local evidence of this gap.

F. Papers we FOLLOW to build THIS plan (reference recipe)

Use this citation order in the paper; this is literally how the plan was composed:





CrashSight — follow phase-aware pre/impact/post structure + expert-refined annotation pipeline



VRU-Accident / AUTOPILOT-VQA / AccidentBench — follow multi-VLM benchmarking culture; cite as “answer-correctness only” baselines we surpass



TRACE — follow atomic event + timestamp representation



CausalVTG — follow counterfactual / evidence-absence testing → design Interventions A/B/C



VER (NeurIPS 2025) — follow framing “thinking drifts from evidence”; replace RL CoT with contracts + interventions for crash reports



MotionBench — follow motivation for fine-grained temporal eval



SeeUnsafe — follow tracking / visual-prompt modules, not the task definition



CCD / Bao MM’20 — cite as video source; explicitly reject anticipation as main task (mentor)



Your completed Crash-1500 summarization pipeline — internal motivating failure (text metrics ≠ faithfulness)

flowchart LR
  CrashSight --> PhaseTEC[Phase_plus_TEC]
  TRACE --> AtomicEvents
  CausalVTG --> Interventions
  VER --> FaithfulnessFraming
  VRU_CrashSight_Auto_AccBench --> GapAnswerOnly
  GapAnswerOnly --> OurProblem
  PhaseTEC --> OurProblem
  AtomicEvents --> OurProblem
  Interventions --> OurProblem
  FaithfulnessFraming --> OurProblem
  CCD --> OurData[Crash1500_videos]
  OurData --> OurProblem
  OurProblem --> CrashGraphVerify

G. One-line “how we get NeurIPS acceptance”

Follow CrashSight/TRACE/CausalVTG/VER for structure and evaluation rigor; solve the shared answer-only limitation with TECs + interventions + epistemic crash reports on Crash-1500; do not sell another VQA/caption/finetune paper.



Central problem formulation

Given full crash video V=x_1,\ldots,x_T, produce structured report:


R=(P,E,G,U,S)






P: participants (type, colour, camera role, impact zone)  



E: atomic events with intervals  



G=(E,A): causal event graph with relations in {before, overlaps, enables, contributes-to, causes, contradicts}  



U: uncertain / unanswerable claims  



S: natural-language summary

Every atomic claim must obey a Temporal Evidence Contract:


c_k=(s_k, e_k=[t_k^s,t_k^e], O_k, r_k, q_k)


with epistemic status r_k\in\text{observed},\text{derived},\text{hypothesised},\text{undetermined}.

A report is accepted only if claims satisfy their contracts (or are explicitly marked undetermined).

flowchart TD
  Video[FullCrashVideo] --> Phase[Pre_Impact_Post]
  Phase --> Parts[ParticipantTracks]
  Parts --> Events[AtomicEventProposals]
  Events --> Graph[CausalEventGraph]
  Graph --> Gen[VLM_ClaimGeneration]
  Gen --> TEC[EvidenceContractPack]
  TEC --> Ver[ContractVerifier]
  Ver -->|unsupported| Rev[ReviseOrAbstain]
  Ver -->|supported| Report[VerifiableCrashReport]
  Rev --> Gen
  TEC --> Int[EvidenceInterventions]
  Int --> Metrics[ERS_IRS_HAG_ECF1]



Strong novelty stack (NeurIPS contribution order)





Temporal Evidence Contracts — claim = text + interval + participants + epistemic + confidence



Evidence Intervention Evaluation (strongest experiment)





A: remove supporting frames → confidence should drop (ERS ↑)  



B: keep only supporting frames → claim recoverable  



C: remove irrelevant frames → confidence stable (IRS ↑)



Hindsight Attribution Gap (HAG) — q(c|V_\text{full})-q(c|V_\text{pre}) for claims annotated as visible pre-impact



CrashGraph-Verify — training-free Generate–Verify–Revise (primary method; no LoRA caption FT)



Crash-EC benchmark extension of your Crash-1500 with deep causal/evidence annotations

This is stronger than box grounding (“where is the car?”): it asks which frames prove lane entry, trajectory conflict, and what cannot be established.



Method: CrashGraph-Verify (no classic finetuning)

Because you are bored of LoRA caption FT—and your own results show it harms faithfulness—the headline method is training-free:

Pipeline





Phase segmentation using existing crash_start / crash_end → V_\text{pre}\cup V_\text{impact}\cup V_\text{post}



Participant proposal — off-the-shelf detector/tracker (e.g. YOLO + ByteTrack) + colour/role prompts; Molmo2 for pointing/tracking where useful



Atomic event proposal — VLM structured JSON events with intervals; optional boundary refine via CLIP/frame–text similarity



Causal edge scoring — LLM/VLM over event pairs with temporal constraints (no edge if evidence insufficient)



Claim generation — VLM emits TEC-formatted claims conditioned on frames + tracks + graph + structured GT fields



Contract verifier F(c_k,E_k,O_k) — check interval–claim alignment (frame relevance), participant consistency, contradiction detection, epistemic calibration



Revise / abstain if support score <\gamma



Intervention audit — automatically build V_{-k}, V_{+k}, V_\text{irr} and re-query claim confidence

Optional (secondary, still not “caption LoRA”)

If reviewers demand a learned component: verifier-guided preference / intervention alignment on a small claim set (DPO/GRPO on contract satisfaction + ERS), never full-report BLEU maximization. Position as ablation, not the story.

What not to do





Do not re-run LLaVA LoRA captioning as the main method  



Do not make anticipation the core task  



Do not use BLEU/ROUGE/CIDEr as primary understanding metrics



Dataset annotation plan (quality > quantity)

Reuse all 1500 broad fields. Deep-annotate a high-quality core:







Layer



Videos



Content





Broad structured (existing)



1500



severity, vehicles, impact, crash window, explanation, ambiguity, camera, weather





Atomic events + intervals



600–800



4–8 events/video





Causal graphs



400–500



3–6 edges/video





Claim–evidence contracts



400–500



5–10 claims + 2–4 uncertain





Triple-annotated gold test



150–200



inter-annotator agreement





Interventions



auto from contracts



A/B/C masks

Target scale for ~500 deep videos: ~2.5–4K events, ~1.5–3K edges, ~3–5K contracts — enough for NeurIPS if IAA is reported.

New tables: participants, atomic events, causal relations, claim–evidence, uncertainty.

Bootstrap: LLM-assisted draft from existing Explanation + crash window → human expert refine (CrashSight-style), with forced undetermined labels when signals/colours are invisible.

Transfer experiment (NeurIPS generality): 100–200 clips from one non-crash domain (falls / workplace / surveillance anomalies) with TECs only—no full second dataset.



Models for experiments (diverse families, zero-shot first)

Run eight systems + method variants. Prefer native video / grounding models; avoid another single-model LoRA paper.

Open VLMs (local)





Qwen2.5-VL-7B (or Qwen3-VL-8B if available) — strong general video + temporal encoding



InternVL3 / InternVL3.5-8B — strong multi-image/video reasoning



LLaVA-Video-7B — established video-instruction baseline



VideoLLaMA3-7B — video-native; good VideoMME/LVBench open baseline



Molmo2-8B — grounding/pointing/tracking specialist (key for TEC intervals + participants)



One stronger open (~32B or MoE) — e.g. Qwen3-VL-32B or InternVL3.5 larger, if GPU allows

Closed APIs (for upper bound)





GPT-4o / latest OpenAI video-capable model



Gemini 2.5 Pro / latest Gemini video model

(Lock exact commercial versions immediately before runs.)

Method configurations (Table 6 ablations)





Raw VLM report  



VLM + uniform frames  



VLM + participant tracks  



VLM + scene graph  



VLM + event graph  



VLM + event graph + evidence verifier  



Full CrashGraph-Verify (+ optional intervention-aligned variant)

Your old LLaVA-NeXT summarization numbers appear only as a faithfulness-vs-n-gram appendix motivating TECs.



Primary metrics (new) vs secondary (old)

Primary





Claim Evidence Precision / Recall / EC-F1  



Evidence Removal Sensitivity (ERS)  



Irrelevant Removal Stability (IRS)  



Temporal Evidence IoU  



Causal Graph F1 (per relation)  



Unsupported Causal Claim Rate (UCCR ↓)  



Epistemic Status Macro-F1  



Report Consistency Score (RCS)  



Hindsight Attribution Gap (HAG)

Secondary only





Event tIoU, severity/vehicle/impact F1  



BERTScore / BLEURT / CIDEr / ROUGE-L  



Human factual correctness

Hypotheses H1–H5 from your brainstorm stay as the experimental story.



Main paper tables





Structured crash understanding



Causal reasoning (relation-wise F1)



Evidence contracts (CEP/CER/EC-F1/UCCR)



Intervention tests (ERS/IRS)



Epistemic reasoning



Ablation of CrashGraph-Verify modules



Why NeurIPS (framing paragraph)

Existing accident benchmarks evaluate answer correctness or textual similarity, but do not verify whether individual causal claims are supported by the temporal evidence they cite. We introduce Temporal Evidence Contracts, a Crash-1500 causal/evidence extension, evidence-intervention evaluation (including hindsight attribution), and CrashGraph-Verify, a training-free generate–verify–revise system that revises or abstains on unsupported claims. A small cross-domain transfer study shows TECs are not traffic-specific.

Mentor one-liner:
“Anticipation is a different task. We use the full crash video for verifiable explanation: participants + atomic events + causal graph + every claim tied to frames; then we remove those frames and test whether the model withdraws the claim.”



Execution roadmap (implementation after plan approval)





Freeze task + schema (TEC JSON, relation set, epistemic labels)



Annotation protocol + pilot 50 videos → IAA → scale to 400–500 deep



Unified multi-VLM inference harness (replace LLaVA-only [src/models/unified_vlm.py](src/models/unified_vlm.py))



Implement CrashGraph-Verify + automatic interventions



Metrics suite (EC-F1, ERS, IRS, HAG, CG-F1)



Full benchmark tables + ablations + human study on 150–200 gold



Optional small transfer domain + optional verifier preference alignment



Paper writing with contribution order above



Risk controls





Annotation cost: start with 50-video pilot; do not annotate all 1500 deeply  



Short 5s clips: emphasize fine-grained within-clip contracts, not long-horizon narrative  



Confidence calibration across models: use self-consistency / logprob / verbal Likert mapped to [0,1] consistently  



Do not overclaim “causality discovery”; claim evidence-constrained causal graph prediction under human annotations  



Cite VER/CausalVTG/TRACE early to pre-empt “already done” reviews

