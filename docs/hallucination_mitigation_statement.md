# Hallucination Mitigation Statement

## Purpose

This note explains how hallucination risk is mitigated in the current VES pilot workflow and why the present results are treated as evidence of real signal rather than fabricated output.

## Main Paper Version

In this project, hallucination risk does not take the form of unconstrained linguistic invention, but rather of visually plausible reconstruction that may not correspond to genuine underlying signal. The model is therefore treated as a constrained image-prediction system rather than as an autonomous reader of text. Several steps are used to reduce that risk. Training is performed on supervised image-label records rather than open-ended text generation. Labels are normalized to a canonical Greek-letter inventory before training so that the model is not rewarded for noise in the metadata. Most importantly, train and validation data are separated at the `document_id` level rather than by random patch alone, reducing leakage between related samples from the same source document. Even with these safeguards, outputs are interpreted conservatively: visually plausible results are not treated as self-validating, but as provisional evidence that must remain stable on held-out documents, survive direct inspection, and later be tested under the contest-facing `64 x 64` tiling workflow rather than only under the earlier `96 x 96` pilot configuration.

## Shorter Paper Version

Hallucination risk in this project is understood not as free-form text invention, but as false visual reconstruction, train/validation leakage, or overinterpretation of weak image signal. We mitigate that risk by training on supervised image-label data, normalizing labels before training, and separating training and validation samples by `document_id` rather than by random patch alone. Accordingly, the model is not claimed to "read" text autonomously. Its outputs are treated as provisional and meaningful only insofar as they remain reproducible on held-out documents, visually consistent under inspection, and robust under the submission-relevant tiling regime.

## Summary Statement

The current system does not rely on unconstrained text generation. It is trained on supervised image-label records and evaluated with document-level train/validation separation. For that reason, the main hallucination risk is not free-form linguistic invention, but overfitting, leakage, or visually plausible reconstruction artifacts that do not generalize beyond the training data. We mitigate that risk by constraining the task, separating related samples by document, filtering labels to a canonical set, and treating outputs as provisional unless they remain stable on held-out documents and under direct visual inspection.

## Mitigations Used

1. The training setup uses supervised image records from a manifest rather than open-ended generative text output.
2. Labels are canonicalized and filtered to an allowed label set before training, which reduces metadata noise and label inconsistency.
3. The train/validation split is performed by `document_id` rather than by random patch alone, reducing leakage between closely related samples from the same source document.
4. Sampling is document-aware, using inverse-square-root weighting at the document/class level to reduce domination by more frequent classes.
5. Results are interpreted conservatively. We do not treat a single visually plausible output as sufficient evidence; we treat outputs as meaningful only if they persist on held-out data and remain visually consistent under inspection.

## Why We Believe the Results Are Real

Our confidence is limited and specific. We are not claiming that the model independently "read" text in a human interpretive sense. We are claiming only that the present pipeline learns stable local image patterns that appear to generalize beyond the exact training examples when evaluated on held-out documents. The strongest basis for confidence is therefore not the plausibility of any one output image, but the combination of:

- document-level separation between train and validation data
- reproducible behavior on held-out samples
- label normalization and filtering before training
- cautious interpretation of outputs as provisional visual evidence rather than definitive readings

## Residual-Risk Sentence

The following sentence can be added if you want the paper to acknowledge the remaining weakness explicitly:

Visually plausible output can still be wrong, especially when the underlying signal is faint or the evaluation regime is too close to the training distribution; for that reason, the present results should be interpreted as provisional evidence of document-generalizing visual signal rather than as proof of textual recovery.

## Conservative Submission Language

The following paragraph is suitable for adaptation in a contest submission:

To mitigate hallucination risk, we do not rely on unconstrained generative text output. Our system is trained on supervised image-label pairs with document-level train/validation separation to reduce leakage across related samples. We treat model outputs as provisional unless they are reproducible on held-out data and remain visually consistent under independent inspection. We are therefore not claiming that the model "read" text by itself; rather, we claim that it learned stable image patterns that generalize beyond the training documents.

## Confidence Statement

The following paragraph is suitable for adaptation in a contest submission:

We are confident that the current results are real only to the extent that they persist on held-out documents, are not explained by train/validation leakage, and are visible as repeatable local image patterns rather than isolated one-off predictions. Our confidence is therefore limited and local: we interpret the current results as evidence of executable, document-generalizing signal, not as proof of full-text recovery.
