import torch
hyperparams_list = [
    {"batch_size": 1, "learning_rate": 5e-4, "num_epochs": 40, "train_percent": 0.8, "optimizer_class": torch.optim.Adam},
    {"batch_size": 1, "learning_rate": 5e-4, "num_epochs": 40, "train_percent": 0.8, "optimizer_class": torch.optim.SGD},
    {"batch_size": 1, "learning_rate": 5e-4, "num_epochs": 40, "train_percent": 0.8, "optimizer_class": torch.optim.RMSprop},
    {"batch_size": 1, "learning_rate": 5e-4, "num_epochs": 40, "train_percent": 0.8, "optimizer_class": torch.optim.Adagrad},
    {"batch_size": 1, "learning_rate": 5e-4, "num_epochs": 40, "train_percent": 0.8, "optimizer_class": torch.optim.AdamW},
]
