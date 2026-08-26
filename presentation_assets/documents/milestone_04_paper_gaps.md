# Milestone 04 — Dual-SwinOrd Model Architecture: Paper Gap Register

This is the authoritative, living audit trail of every architectural value or mechanism that the Dual-SwinOrd paper does **not** explicitly specify. Per the project's strict paper-fidelity policy, none of these gaps is silently resolved with a "common practice" default anywhere in `src/models/`. Every gapped field is instead a **required (no-default) configuration field** or an **abstract interface with no concrete production implementation**, so attempting to build a real model fails fast, naming the missing value, until a human resolves the gap here first.

## Source availability (read before using this register)

- **Source #2 ("engineering report extracted from the paper") does not exist in this repository.** `APTOS_REPORT/` contains only `Report1_EDA_Analysis.docx` and `Report2_Data_Preparation_Recommendations.docx`, both Data Preparation-milestone artifacts. If such a report exists elsewhere, it should be added under `docs/literature/` and this register re-audited against it.
- **The paper's full text/PDF could not be fetched** (`mdpi.com` returned `403 Forbidden` for both the canonical URL and the DOI redirect). Every claim below is based on **search-engine-indexed excerpts** only: the abstract, Figure 1's caption, and a handful of Section 3 sentences. A gap marked "unresolved" here may in fact be resolved in the paper's full text (equations, appendix, hyperparameter table) — this register should be re-checked if/when the full PDF becomes available.

## What is explicitly supported (kept, not gapped)

| Fact | Evidence |
|---|---|
| Backbone is a hierarchical Swin Transformer with 4 stages | Abstract; Figure 1 caption ("Stage 1 to Stage 4") |
| PLKA dilation rates are exactly 1 ("standard"), 2, 3 | Figure 1 caption: "parallel convolutional branches with different dilation rates (standard, r = 2, r = 3)" |
| SPM's gating signal uses a sigmoid activation | Figure 1 caption: "a gating signal through sigmoid activation" |
| Dual-Head macro order: (1) Multi-modal Input/SPM interacting with (2) Backbone → (3) PLKA → (4) Decoupled Dual-Head | Figure 1 caption's numbered component list |
| Classification Head: `K=5`-way linear/softmax output | Section 3 (quoted): "maps the feature vector to `K = 5` logits" |
| Ordinal Head predicts, per threshold `k`, whether severity `> k` | Figure 1 caption (quoted): "predicting whether severity > k" |
| Final inference uses `argmax` over classification output, refined by ordinal constraints | Figure 1 caption — explicitly an **inference-time** rule, out of scope for the model's `forward()` |
| `timm` is an acceptable implementation vehicle for the Swin backbone | Not paper-derived; `timm>=1.0` was already a pinned dependency in `pyproject.toml` before this milestone began, so using it introduces no new dependency |
| **[Milestone 05]** Classification Head loss: Cross-Entropy with Label Smoothing, `L_cls = -sum_{i=0}^{K-1} y_i log(p_i^cls)` | Section 3, Eq. 7 (quoted, found via a deeper search than the one available when this register was first written) |
| **[Milestone 05]** Ordinal Head strategy is named "Deep Progressive Enhancement"; it decomposes the task into `K-1` binary sub-tasks with independent per-threshold targets `v_k`, trained with binary cross-entropy: `L_ord = -sum_{k=1}^{K-1} [v_k log(p_k^ord) + (1-v_k) log(1-p_k^ord)]` | Section 3, Eq. 8 (quoted): "a Deep Progressive Enhancement strategy to enforce rank consistency... decompose the task into K-1 binary sub-tasks" |
| **[Milestone 05]** Total loss is a fixed convex combination `L_total = λ L_cls + (1-λ) L_ord`, with `λ = 0.5` used in the paper's own experiments | Section 3, Eq. 9 (quoted): "In our experiments, we set λ = 0.5" |
| **[Milestone 05]** Optimizer = AdamW; learning rate = 1e-4; weight decay = 1e-4; epoch budget = 50; LR schedule family = cosine annealing | Section 4 (quoted): "optimized using the AdamW algorithm... maximum epochs = 50, learning rate = 1×10⁻⁴, and weight decay = 1×10⁻⁴... A cosine annealing scheduler was employed" |
| **[Milestone 05]** Training-time augmentation: horizontal flip, vertical flip, random rotation, color jitter | Section 4 (quoted): "horizontal and vertical flipping, random rotation, and color jittering" — matches `configs/data.yaml`'s `evidence: "paper"`-tagged transforms already shipped; no change required there |

### Resolved gaps (Milestone 05 evidence)

| ID | Resolved by | Date | Note |
|---|---|---|---|
| PG-13 | Eq. 8 (quoted above): independent per-threshold binary sub-tasks, each a sigmoid/BCE target `v_k`, matches the "independent per-threshold" parameterization option this register originally listed as one of several candidates. `src/losses/ordinal_loss.py` implements this exactly; a concrete `OrdinalHead` subclass may now be added to `src/models/heads/` in a future architecture patch — not added by Milestone 05, which only consumes the existing abstract interface's `[B, K-1]` logit contract. | Milestone 05 planning session | Per the resolution protocol (row 4 below), this citation resolves the *loss* parameterization; a concrete production `OrdinalHead` subclass is a separate, not-yet-made architecture-package change. |
| PG-14 | "DPE" = "Deep Progressive Enhancement" (Eq. 8's surrounding sentence, quoted above) | Milestone 05 planning session | Acronym fully resolved; no further ambiguity. |

## Open Paper Gaps

| ID | Component | Gap | Evidence available | Implementation status |
|---|---|---|---|---|
| PG-01 | Backbone | Swin variant, patch size, and window size unspecified | Only "hierarchical Swin Transformer... four stages" | `BackboneConfig.variant`/`pretrained` — required, no default. `src/models/backbones/swin.py` |
| PG-02 | Backbone | Input resolution not paper-confirmed | Not mentioned in excerpts | `BackboneConfig.image_size` — required, no default |
| PG-03 | Semantic Prior / TextAdapter | PubMedCLIP library and checkpoint identity unspecified | Only the name "PubMedCLIP" appears | `src/models/semantic_prior/text_adapter.py::TextAdapter` is a pure ABC; **no concrete subclass ships in this milestone**; **no new dependency added** |
| PG-04 | Semantic Prior / TextAdapter | Clinical text prompt set unspecified | Only one example prompt quoted ("Microaneurysms") | Prompts are a runtime argument to `TextAdapter.encode()`, never hardcoded anywhere in `src/models/` |
| PG-05 | SPM | Which backbone stage(s) receive the gating signal is unspecified; Figure 1's component ordering (SPM/Backbone interaction listed *before* PLKA/Dual-Head) if anything suggests injection happens during/at the backbone rather than strictly "after" it — the opposite of an earlier draft's "last stage only" guess, which is explicitly retracted here | Figure 1 caption: signal "injected into the Swin Transformer backbone" (no stage detail) | `SPMConfig.inject_at_stage` — a single required stage index, no default. Whether *multiple* stages should be injected simultaneously is itself unresolved and not supported by the current interface (see PG-05b) |
| PG-05b | SPM | Single-stage vs. multi-stage injection is itself unresolved | Same as PG-05 | Current interface deliberately supports exactly one injection stage (the more conservative reading); extending to multiple stages is deferred until this sub-question is resolved |
| PG-05c | SPM | The implementation modulates the backbone's already-extracted stage *output*, not the backbone's internal transformer blocks (no hook for the latter is exposed by `timm`, and the paper gives no detail to justify a deeper, custom integration) | N/A — implementation-boundary note, not a paper quote | Documented limitation of `src/models/dual_swinord.py`'s wiring |
| PG-06 | SPM | The "fusion matrix" algebra and the gate/feature combination rule (multiply / add / FiLM-style scale-shift) are unspecified — only the sigmoid nonlinearity is paper-fixed | "fusion matrix... gating signal through sigmoid activation" | `SemanticPriorModulation.fuse()` and `.apply_gate()` are abstract; **no concrete subclass ships in this milestone** |
| PG-07 | PLKA | Per-branch activation function and convolution kernel size unspecified | Not mentioned | `PLKAConfig.activation`, `PLKAConfig.kernel_size` — required, no default |
| PG-08 | PLKA | Per-branch normalization layer unspecified | Not mentioned | `PLKAConfig.normalization` — required, no default |
| PG-09 | PLKA | The "attention-based fusion mechanism" architecture is named but never defined | "followed by an attention-based fusion mechanism" | `PLKAFusion` is an ABC; **no concrete subclass ships in this milestone** |
| PG-10 | PLKA | Which backbone stage feeds PLKA is unspecified (only the *sequential* Backbone → PLKA order is explicit) | Figure 1 component order | `PLKAConfig.input_stage` — a single required stage index, no default |
| PG-11 | Shared Neck | Pooling strategy reducing PLKA's spatial output to a vector is unspecified | "shared fully connected layer" implies but never names a reduction step | `NeckPooling` is an ABC; **no concrete subclass ships in this milestone**. `NeckConfig.pooling` names the registry key, currently unresolvable |
| PG-12a | Shared Neck | Hidden dimension of the shared FC layer unspecified | Not mentioned | `NeckConfig.hidden_dim` — required, no default |
| PG-12b | Shared Neck | Whether any activation/dropout follows the shared FC layer *at all* is unspecified (not just their values) | Not mentioned | `NeckConfig.activation`/`dropout` are required, no-default fields; both fully disable via `"identity"`/`0.0`, so their presence in the config schema is a configurability affordance, not a claim that the paper uses them |
| PG-13 | Ordinal Head | The parameterization achieving "> k" outputs is unspecified: independent per-threshold linear layers, a CORAL-style shared-weight/per-threshold-bias formulation, and a CONDOR-style conditional formulation are all consistent with the quoted semantics and are architecturally distinct | "predicting whether severity > k" (semantics only) | `OrdinalHead` is an ABC; **no concrete subclass ships in this milestone** |
| PG-14 | Ordinal Head | The abstract names this head "DPE" ("Ordinal Regression Head (DPE)") with **zero expansion or definition** anywhere in the retrieved excerpts | Named once, in the abstract | `OrdinalHead`'s docstring explicitly states it does **not** claim to implement "DPE" — only the quoted ">k" semantic |
| PG-15 | Sources | No "engineering report" document exists in the repository for the model architecture | N/A | Source-priority policy honored using Source #1 (paper excerpts) and Source #3 (data pipeline) only |
| PG-16 | Sources | Full paper text/PDF inaccessible; only indexed excerpts available | N/A | Every gap above should be re-audited if the full text becomes available |
| PG-17 | **[Milestone 05]** CARM Loss | The abstract names the Ordinal Head's loss "Cost-sensitive Adaptive Risk Minimization (CARM)" and states its purpose is "to prevent bias toward majority classes", but Eq. 8 (the only concrete ordinal-loss equation found) is a plain, unweighted per-threshold binary cross-entropy with no visible cost matrix or adaptive weighting term | Abstract + Section 3 (Table 1 discussion) name/purpose CARM; Eq. 8 shows no weighting term | `src/losses/carm_loss.py::CARMLoss` implements Eq. 8 exactly as its default behavior; exposes an optional, off-by-default `pos_weight` constructor argument labeled a **future-extension, not paper-confirmed**, so the discrepancy is neither silently ignored nor silently "fixed" with an invented mechanism |
| PG-18 | **[Milestone 05]** LR Scheduler | Cosine annealing is paper-confirmed, but `T_max` and `eta_min` are not given a value anywhere in the retrieved excerpts | Section 4 names the scheduler family only | `TrainingConfig.scheduler.eta_min` — required, no default. `T_max` defaults to `training.epochs` (a derivation from the paper-confirmed epoch budget, not an invented value) but remains overridable |
| PG-19 | **[Milestone 05]** Optimizer | Layer-wise learning-rate decay is never mentioned | Not mentioned in any excerpt | `TrainingConfig.optimizer.layerwise_lr_decay: float \| None` — disabled (`None`) by default; a purely optional, clearly-labeled engineering knob, never applied unless explicitly set |

## Resolution protocol

When a gap is resolved:

1. Cite the exact paper section, equation, table, or appendix entry that closes it — never "resolved by common practice."
2. Update this table's row: add a "Resolved by" column entry and the date.
3. Only then may a concrete implementation be added to `src/models/` (a new registered subclass, or a filled-in default) — implementation must not precede the citation.
4. Test-only implementations under `tests/` (see `tests/model_doubles.py` and `tests/fixtures/non_paper_test_config.yaml`) never count as resolving a gap; they exist solely to exercise the architecture's interfaces in CI and are explicitly labeled non-paper-faithful.
