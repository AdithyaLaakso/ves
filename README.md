An attempt to do ink detection on virtually unwrapped segments of ancient greek scrolls. It uses a custom attention-based hybird segmentation and classifcation arcitecture trained on synthetic data.

This methodology has seen significant success on modern greek sample data. The team is currently working to rework the data generation process to more closly mimic the predicted appearance of letters within the scrolls.

Here are some samples of what it is capable of:

![sample1](sample1.png)
![sample2](sample2.png)

To train:
1. Open greek_letters/src/main.py and set the desginered parameters for data generation.
2. Set the desired paramaters in letter_visualization_model/settings.py
3. Run letter_visualization_model/setup.zsh

The model will save the .pth to the specified location (default new.pth).

The script will automatically start logging to the specified log directory (default logs/).

To view the logs:
1. python -m tensorboard.main --logdir letter_visualization_model/logs/
2. Navigate to localhost:6006 in your browser.

To run inferance on a saved pth:
1. Python letter_visualization_model/visualize_model.py example.pth
2. Arrow keys navigate between samples, 'q' exits

If you have any questions, feel free to leave an issue or reach out to me on discord.
