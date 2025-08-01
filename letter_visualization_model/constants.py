import torch
MAX_SIZE = 10000000000
hyperparams_list = [
    {"batch_size": 128, "learning_rate": 5e-4, "num_epochs": 10, "train_percent": 0.8, "optimizer_class": torch.optim.Adam},
    {"batch_size": 128, "learning_rate": 5e-4, "num_epochs": 10, "train_percent": 0.8, "optimizer_class": torch.optim.SGD},
    {"batch_size": 128, "learning_rate": 5e-4, "num_epochs": 10, "train_percent": 0.8, "optimizer_class": torch.optim.RMSprop},
    {"batch_size": 128, "learning_rate": 5e-4, "num_epochs": 10, "train_percent": 0.8, "optimizer_class": torch.optim.Adagrad},
    {"batch_size": 128, "learning_rate": 5e-4, "num_epochs": 10, "train_percent": 0.8, "optimizer_class": torch.optim.AdamW},
]

greek_letters = {
    "ALPHA": 0,
    "BETA": 1,
    "GAMMA": 2,
    "DELTA": 3,
    "EPSILON": 4,
    "ZETA": 5,
    "HETA": 6,
    "THETA": 7,
    "IOTA": 8,
    "KAPA": 9,
    "LAMDA": 10,
    "MI": 11,
    "NI": 12,
    "XI": 13,
    "KSI": 13,
    "OMIKRON": 14,
    "PII": 15,
    "RO": 16,
    "SIGMA": 17,
    "TAU": 18,
    "YPSILON": 19,
    "FI": 20,
    "CHI": 21,
    "PSI": 22,
    "OMEGA": 23,
}
