# Window Size Transition Note for the Paper

## Purpose

This note provides paper-ready language explaining why the project is introducing a `64 x 64` configuration now while still retaining a provisional `96 x 96` path during the current validation phase.

## Core Rationale

The project originally used larger windows and later reduced them during model refinement. The current pilot configuration uses `96 x 96` inputs, which are smaller than the earlier `128 x 128` setup but still larger than the contest guidance discouraging machine-learning image generation from windows above `0.5 x 0.5 mm` (`64 x 64` pixels at `8 um` resolution). For that reason, the project now needs a contest-facing `64 x 64` variant.

At the same time, it would be methodologically unwise to discard the `96 x 96` path immediately. The current pilot evidence, smoke-test behavior, and local validation reporting are all tied to that configuration. Retaining it temporarily provides a continuity baseline against which the new `64 x 64` variant can be compared.

## Short Explanation for the Paper

The following paragraph is suitable for adaptation in the methods or pilot-planning section:

During the present refinement stage, the model configuration was reduced from earlier larger windows to a `96 x 96` input setting. That reduction improved practicality while preserving continuity with the current pilot validation results. However, because the contest guidance discourages model-generated images from windows larger than `0.5 x 0.5 mm` (`64 x 64` pixels at `8 um`), we are now introducing a parallel `64 x 64` configuration. The purpose of this change is not to abandon the existing `96 x 96` pilot immediately, but to begin testing the submission-relevant operating point before further optimization effort accumulates around a larger window size.

## Why `96 x 96` Is Still Retained for Now

The following paragraph is suitable for adaptation in the discussion or limitations section:

For the moment, one foot remains in the `96 x 96` configuration because that setting anchors the current smoke-test evidence, runtime expectations, and local validation results. Keeping it temporarily allows direct comparison between the established pilot configuration and the newer `64 x 64` contest-facing variant. In practical terms, the `96 x 96` model is being treated as a continuity and calibration baseline, while `64 x 64` becomes the more important target for submission-relevant evaluation.

That continuity argument is also technical, not just rhetorical. The currently available checkpoints and validation evidence belong to the `96 x 96` profile. Because the `64 x 64` variant changes architectural dimensions tied to patching and output geometry, those earlier checkpoints do not transfer directly as plug-compatible inference weights. The `96 x 96` path therefore remains useful during the transition as the last fully runnable reference configuration while `64 x 64` is brought into active training and evaluation.

## More Formal Alternative

If a more formal style is preferable, the same point can be stated as follows:

The transition to `64 x 64` inputs is being made now in response to contest constraints rather than after finalizing all further testing at `96 x 96`. This decision reflects a methodological concern: strong performance at a larger, non-compliant window size would not by itself establish the viability of the submission-relevant configuration. Accordingly, the `96 x 96` setting is retained only as a comparative baseline while the `64 x 64` variant is brought into active evaluation.

## Suggested Next Experiment Sentence

The following sentence can be used if you want to state the immediate plan explicitly:

The next testing phase will therefore compare the established `96 x 96` pilot configuration with a newly introduced `64 x 64` variant, focusing on smoke-test stability, runtime, and held-out validation behavior.
