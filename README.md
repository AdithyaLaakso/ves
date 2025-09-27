Ves is an attempt to do ink detection on virtually unwrapped segments of ancient greek scrolls. It uses a custom attention-based hybird segmentation and classifcation arcitecture trained on synthetic data.

This methodology has seen significant success on modern greek sample data. The team is currently working to rework the data generation process to more closly mimic the predicted appearance of letters within the scrolls.

Here are some samples of what it is capable of:

![sample1](sample1.png)
![sample2](sample2.png)

To Install Dependencies:
1. ```pip install -r requirements.txt```

To Generate Data:
1. Set desired data parameters in data_gen/papyrus/gen_data.py (optional)
2. Make sure you are in the root of this repo (important)
3. run ```python data_gen/papyrus/gen_data.py``` (this step can take quite a while)

To train:
1. ```cd model/```
2. Set the desired parameters in settings.py (optional)
3. Run the setup script ```./setup.zsh```
4. While it is running, you may view the loss chart in your browser at localhost:6006/
5. When the model is finished, it will display results

The model will save the .pth to the specified location (default new.pth).

The script will automatically start logging to the specified log directory (default logs/).

To view the logs of a previous run:
1. python -m tensorboard.main --logdir models/logs/example/
2. Navigate to localhost:6006 in your browser.

To run inference visualizer on a saved model:
1. Python letter_visualization_model/visualize_model.py example.pth
2. Arrow keys navigate between samples, 'q' exits

Note: the model has only been tested on Linux. If there is interest in windows/OSX support, add an issue or contact on discord.

If you have any questions, feel free to leave an issue or reach out to on discord.
