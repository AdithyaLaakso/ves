greek_letters = {
    "ALPHA": 0,
    "BETA": 1,
    "GAMMA": 2,
    "DELTA": 3,
    "EPSILON": 4,
    "ZETA": 5,
    "ETA": 6,
    "THETA": 7,
    "IOTA": 8,
    "KAPPA": 9,
    "LAMBDA": 10,
    "MU": 11,
    "NU": 12,
    "XI": 13,
    "OMICRON": 14,
    "PI": 15,
    "RHO": 16,
    "LUNATE_SIGMA": 17,
    "TAU": 18,
    "UPSILON": 19,
    "PHI": 20,
    "CHI": 21,
    "PSI": 22,
    "OMEGA": 23,
}


greek_letter_aliases = {
    "HETA": "ETA",
    "KAPA": "KAPPA",
    "LAMDA": "LAMBDA",
    "MI": "MU",
    "NI": "NU",
    "KSI": "XI",
    "OMIKRON": "OMICRON",
    "PII": "PI",
    "RO": "RHO",
    "SIGMA": "LUNATE_SIGMA",
    "YPSILON": "UPSILON",
    "FI": "PHI",
}


def canonicalize_label(label: str) -> str:
    upper = label.upper()
    return greek_letter_aliases.get(upper, upper)
