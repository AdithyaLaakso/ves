# Paper Section: Why the Pipeline Shift Is Methodologically Defensible

## Main Version

An earlier assumption in this project was that training should rely primarily, or even exclusively, on synthetic data. That position has an understandable methodological appeal. Synthetic generation offers a high degree of control over corruption, contrast, geometry, and class balance, and it can be especially valuable when domain-relevant labeled data are scarce. In that setting, synthetic training is not merely convenient; it may be the only practical way to begin experimentation. For an early proof-of-concept, therefore, a synthetic-first strategy was reasonable.

The difficulty is that synthetic control is not identical to domain validity. A synthetic pipeline necessarily encodes prior assumptions about what the target domain should look like. Those assumptions include not only noise models and degradation processes, but also assumptions about letter morphology, stroke structure, background texture, and the kinds of variation that matter most. If those assumptions are incomplete or inaccurate, the model may learn the simulator's regularities rather than the structure of the real target domain. This risk becomes more serious once the project shifts from demonstrating bare executability to asking whether the learned behavior is likely to transfer to historically relevant material.

That is the point at which the revised pipeline changes course. Rather than relying solely on synthetic exemplars, it now uses a manifest-based corpus of real ancient Greek uncial images and applies controlled degradation during training. This change preserves the useful part of the synthetic approach, namely the ability to introduce explicit corruption and reconstruction pressure, while relocating supervision onto a more domain-aligned corpus. In other words, synthetic corruption remains in the workflow, but it is no longer asked to carry the full burden of morphological realism.

This shift should not be understood as a rejection of synthetic methods in principle. It is better understood as a refinement of their role. Synthetic generation remains useful for augmentation, controlled perturbation, and stress testing. What has changed is the judgment that, once a sufficiently relevant uncial corpus is available, purely synthetic supervision becomes harder to defend as the primary training basis. At that stage, real exemplars provide a stronger foundation for claims about whether the model is learning patterns that are historically and visually pertinent to the target problem.

The same logic applies to the decision to introduce mild sampler-level reweighting. The revised pipeline does not replace the underlying corpus with an artificial uniform distribution. It preserves the real class distribution and uses only inverse-square-root weighting during sampling so that the most common letters do not dominate the gradient signal too strongly. This is a conservative intervention. It changes training exposure, not the composition of the corpus itself, and it is materially milder than full inverse-frequency balancing. The methodological purpose is not to deny the true imbalance of the dataset, but to reduce the risk that rarer but still important forms become nearly irrelevant during optimization.

Taken together, these revisions make the pipeline easier rather than harder to defend. The training source is now more historically aligned, the validation regime is more defensible because it separates samples by source document, and the remaining use of synthetic corruption is explicit and limited. The revised workflow therefore represents not a relaxation of rigor, but an attempt to reduce dependence on simulator assumptions while retaining controlled augmentation where it remains genuinely useful.

## Shorter Version

Earlier stages of the project relied more heavily on synthetic generation, which was a reasonable choice when more domain-relevant labeled material was not yet integrated. Synthetic data offer useful control over noise, degradation, and class balance, but they also embed strong assumptions about morphology and background structure. Once a more relevant corpus of ancient Greek uncial images became available, it became more methodologically defensible to train on real exemplars and use synthetic degradation as augmentation rather than as the sole source of supervision. The same reasoning supports the introduction of mild sampler-level weighting: the underlying corpus is left intact, while training exposure is adjusted only enough to reduce domination by the most common classes. In this sense, the revised pipeline should be understood not as a rejection of synthetic methods, but as a narrower and more defensible use of them.

## Compact Bridge Paragraph

The move away from a purely synthetic training basis should therefore be understood as a methodological refinement rather than as a departure from rigor. Synthetic data remain useful for controlled degradation and augmentation, but once a more domain-relevant uncial corpus became available, it was more defensible to place that corpus at the center of supervision and use synthetic corruption only in a supporting role.

## Polished Transfer Paragraph

The principal argument for the revised pipeline is not that it is intrinsically more efficient than the earlier synthetic-first workflow, but that it may have a better chance of transferring to the actual target problem. Letter detection on virtually unrolled parchment slices depends not only on corruption and noise, but on the underlying morphology of the letterforms themselves. A fully synthetic pipeline necessarily encodes assumptions about what those forms should look like, and a model trained under that regime may learn the simulator's regularities rather than patterns that are genuinely relevant to ancient material. By centering supervision on real ancient Greek uncial exemplars while retaining controlled degradation as augmentation, the revised workflow reduces dependence on simulator assumptions about morphology without abandoning the useful role of synthetic corruption. It does not eliminate the domain gap between curated training images and real scroll data, but it narrows one important part of that gap and is therefore a more defensible basis for asking whether the learned behavior has any realistic chance of transfer.

## Notes on Tone

This framing is useful if the paper needs to remain collegial toward readers who were trained to prefer synthetic-first methods:

- It acknowledges why synthetic-first reasoning is attractive.
- It does not caricature that view as naive.
- It argues that the project crossed a threshold where the stronger methodological choice changed.
- It presents the revision as a narrowing of assumptions, not an abandonment of discipline.
