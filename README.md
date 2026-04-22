Ves is an attempt to do ink detection on virtually unwrapped segments of ancient Greek scrolls. The revised branch uses a custom attention-based hybrid segmentation and classification architecture trained on a manifest-based ALPUB_v2 uncial dataset, with on-the-fly degradation applied during loading for reconstruction training.

Earlier versions of the project relied more heavily on synthetic generation workflows. The current direction shifts toward historically better-matched uncial source material while retaining some legacy synthetic-generation code for comparison and experimentation.

Here are some samples of what it is capable of:

![sample1](samples/sample1.png)
![sample2](samples/sample2.png)

To Install Dependencies:
1. ```pip install -r requirements.txt```

To Prepare Data For The Revised ALPUB_v2 Pipeline:
1. Make sure the extracted `ALPUB_v2/images/...` tree exists in the repo root.
2. Make sure you are in the root of this repo.
3. Run `python3 data_gen/build_alpub_manifest.py`
4. This writes `data/alpub_v2_manifest.json`, which is the dataset entry point used by the current training path.

What changed in the revised pipeline:
1. The current branch does not pre-generate noisy/clean reconstruction pairs on disk.
2. Training now reads clean ALPUB_v2 images from the manifest and applies degradation on the fly during loading.
3. The clean source image is used as the reconstruction target.

Legacy synthetic-data workflow:
1. The older synthetic generator still exists in `data_gen/papyrus/gen_data.py`.
2. That path is not the primary workflow for the revised manifest-based branch.

To train:
1. Make sure you have prepared the manifest data (see above). You can check that the repo contains `ALPUB_v2/` and `data/alpub_v2_manifest.json`.
2. The repo root should look something like this:
   ```
    % ls
    archive  data  data_gen  model  README.md  requirements.txt  samples
   ```
3. `cd model/`
4. Set the desired parameters in `settings.py` or via environment variables if needed.
5. Run the setup script:
   - standard: `./setup.zsh`
   - with TensorBoard: `VES_RUN_TENSORBOARD=1 ./setup.zsh`
   - with visualization after training: `VES_RUN_VISUALIZE=1 ./setup.zsh`
6. The script now auto-detects whether CUDA is available:
   - on CUDA systems, it defaults to a larger GPU-oriented validation run
   - on non-CUDA systems, it falls back to the smaller CPU smoke-test path
7. The script prints the effective run mode at startup so you can confirm whether it is using CPU/GPU, smoke-test/fuller validation settings, and whether visualization is enabled.
8. If TensorBoard is enabled, you may view the loss chart in your browser at `localhost:6006/`
9. When the model is finished, it will display results if visualization was enabled.

The model will save the weights to the specified location (default `new.pth` inside the `model/` directory).

The script will automatically start logging to the specified log directory (default logs/).

To view the logs of a previous run:
1. `cd model/`
2. `python -m tensorboard.main --logdir ./logs --port=6006`
3. Navigate to `localhost:6006` in your browser.

To run inference visualizer on a saved model using the preset data:
1. `cd model/`
2. `python visualize_model.py path/to/weights.pth`
3. Arrow keys navigate between samples, `q` exits


To run inference visualizer on a saved model using your own sample:
1. `cd model/usage/`
2. `python run_model.py path_to_your_input_image.png`
3. If you want a specific checkpoint, pass it explicitly or update `settings.display_from` in `model/settings.py`.

To tile a large real scroll image into fixed-size windows:
1. `python model/usage/build_scroll_tile_manifest.py path/to/scroll_image.tif --tile-size 64`
2. This writes tiles and a manifest under `data/scroll_tiles/`.

To run tiled inference on a large image:
1. `cd model/`
2. `VES_SIZE_PROFILE=64 python usage/infer_scroll_tiles.py ../data/scroll_tiles/<scroll_name>_64/manifest.json --weights path/to/64x64_weights.pth --save-preview`
3. A `96x96` checkpoint cannot be used directly with the `64x64` model profile.

Note: the model has only been tested on Linux. If there is interest in windows/OSX support, add an issue or contact on discord.

If you have any questions, feel free to leave an issue or reach out on discord.
