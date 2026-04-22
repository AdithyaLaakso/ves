# Window Size Compliance Note

## Purpose

This note records the current model window size against the contest guidance discouraging machine-learning image generation with windows larger than `0.5 x 0.5 mm`, stated as `64 x 64` pixels for `8 um` scans.

## Configuration History

Earlier revisions of the model configuration used:

- `image_size = 128`
- `input_size = 128`

The current pilot configuration sets:

- `image_size = 96`
- `input_size = 96`

This shows a verified reduction from `128 x 128` to `96 x 96` during model refinement. What can be supported from the repository is that the window size was reduced. What cannot presently be supported from the repository is a documented plan to halve the window all the way to `64 x 64`.

Under an `8 um` scan resolution:

- `128 x 128` corresponds to `1.024 x 1.024 mm`
- `96 x 96` corresponds to `0.768 x 0.768 mm`

This means the present pilot model window is approximately:

- `0.768 x 0.768 mm`

## Compliance Interpretation

If the contest submission uses this exact `96 x 96` model window to generate submission images, it exceeds the stated `0.5 x 0.5 mm` guidance. On that basis, the current pilot configuration is not safely compliant with the quoted rule and could expose the submission to rejection or a request to resubmit with a smaller window.

## Recommended Action

The safest interpretation of the rule is to treat `64 x 64` pixels at `8 um` resolution as the contest-facing ceiling and rerun the submission pipeline under that constraint. In practice, that means:

1. Set the effective model input window to `64 x 64`.
2. Retrain or rerun any model whose generated images would otherwise depend on the larger `96 x 96` receptive field.
3. State the final submission window size explicitly in the technical description.

## Suggested Submission Language

The following paragraph is suitable for adaptation in a contest submission:

We reviewed the contest guidance discouraging machine-learning image generation from windows larger than `0.5 x 0.5 mm` (`64 x 64` pixels at `8 um`). Earlier versions of our configuration used `128 x 128` windows, and the current pilot configuration uses `96 x 96`, corresponding to approximately `0.768 x 0.768 mm`. This is smaller than the original setup but still above the quoted guideline. For a contest-facing submission, the safer course is to reduce the effective input window to `64 x 64` and regenerate results under that constraint.
