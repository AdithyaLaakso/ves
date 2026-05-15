# Third-Party Notices

This repository contains project code alongside data preparation workflows and
archival research materials. The root `LICENSE` file covers project-authored
source code, documentation, and configuration unless a file or directory says
otherwise. It does not relicense third-party datasets, generated data, sample
images, model weights, or external assets.

## ALPUB_v2

The revised training path expects the ALPUB_v2 dataset to be available locally
under `ALPUB_v2/`, and `data_gen/build_alpub_manifest.py` can generate
`data/alpub_v2_manifest.json` from that local dataset. ALPUB_v2 is published
separately and is listed on Kaggle under the Apache 2.0 license.

## Perseus/Ancient Greek Text Materials

Some archived Greek text materials in this repository have filenames that
identify Perseus editions, such as files under `archive/Greek_Full/`,
`archive/Greek_Raw_Cleaned/`, and related `archive/prob_field/` paths. These
materials are retained as archival or experimental inputs and remain subject to
their own source terms. They are not relicensed by the root `LICENSE` file.

## Generated Outputs

Generated manifests, generated training data, sample images, and model weights
may depend on external datasets or local experiment inputs. Treat those outputs
as having the license obligations of their source materials unless a specific
output states otherwise.
