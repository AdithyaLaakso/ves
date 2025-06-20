import numpy as np
from typing import Dict, List, Optional, Union

class greek_char_prob_field:
    # Greek alphabet mapping
    GREEK_LETTERS = [
        # Lowercase
        'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ',
        'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
        # Uppercase  
        'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'Μ',
        'Ν', 'Ξ', 'Ο', 'Π', 'Ρ', 'Σ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω'
    ]
    def __init__(self, probabilities: Optional[Union[List[float], np.ndarray, Dict[str, float]]] = None, start_char = None):
        if start_char is not None and start_char in self.GREEK_LETTERS:
            self.probs = np.zeros(48)
            idx = self.GREEK_LETTERS.index(start_char)
            self.probs[idx] = 1
        elif start_char is not None and start_char not in self.GREEK_LETTERS:
            raise ValueError(f"Attempted to init a greek prob field with not greek char: {start_char}")
        elif probabilities is None and start_char is None:
            # Uniform distribution
            self.probs = np.ones(48) / 48.0
        elif isinstance(probabilities, dict):
            self.probs = np.zeros(48)
            for char, prob in probabilities.items():
                if char in self.GREEK_LETTERS:
                    idx = self.GREEK_LETTERS.index(char)
                    self.probs[idx] = prob
                else:
                    raise ValueError(f"Unknown Greek letter: {char}")
        else:
            self.probs = np.array(probabilities, dtype=float)
        if len(self.probs) != 48:
            raise ValueError(f"Expected 48 probabilities, got {len(self.probs)}")
        # Normalize to ensure sum equals 1
        if not np.isclose(self.probs.sum(), 1.0):
            print(f"Warning: Probabilities sum to {self.probs.sum():.6f}, normalizing...")
            self.probs = self.probs / self.probs.sum()
    def get_probability(self, char: str) -> float:
        """Get probability for a specific Greek letter."""
        if char not in self.GREEK_LETTERS:
            raise ValueError(f"Unknown Greek letter: {char}")
        idx = self.GREEK_LETTERS.index(char)
        return self.probs[idx]
    def set_probability(self, char: str, prob: float):
        """Set probability for a specific Greek letter and renormalize."""
        if char not in self.GREEK_LETTERS:
            raise ValueError(f"Unknown Greek letter: {char}")
        idx = self.GREEK_LETTERS.index(char)
        self.probs[idx] = prob
        self.probs = self.probs / self.probs.sum()  # Renormalize
    def get_top_k(self, k: int = 5) -> List[tuple]:
        sorted_indices = np.argsort(self.probs)[::-1]
        return [(self.GREEK_LETTERS[i], self.probs[i]) for i in sorted_indices[:k]]
    def most_likely_char(self) -> str:
        """Return the most likely character."""
    def entropy(self) -> float:
        """Calculate the entropy of the probability distribution."""
        # Avoid log(0) by adding small epsilon
        eps = 1e-15
        return -np.sum(self.probs * np.log2(self.probs + eps))
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary mapping characters to probabilities."""
        return {char: prob for char, prob in zip(self.GREEK_LETTERS, self.probs)}
    def sample(self, n_samples: int = 1) -> Union[str, List[str]]:
        samples = np.random.choice(self.GREEK_LETTERS, size=n_samples, p=self.probs)
        return samples[0] if n_samples == 1 else samples.tolist()
    def __str__(self) -> str:
        """String representation showing top 5 most likely characters."""
        top_5 = self.get_top_k(5)
        top_str = ", ".join([f"{char}: {prob:.3f}" for char, prob in top_5])
        return f"GreekCharProbability(top 5: {top_str})"
    def add_noise(self,
                  uniform_noise: float = 0.0,
                  gaussian_noise: float = 0.0,
                  dropout_rate: float = 0.0,
                  swap_noise: float = 0.0,
                  random_seed: Optional[int] = None) -> 'GreekCharProbability':
        if random_seed is not None:
            np.random.seed(random_seed)
        noisy_probs = self.probs.copy()
        # Apply dropout (randomly zero out some probabilities)
        if dropout_rate > 0:
            dropout_mask = np.random.random(48) > dropout_rate
            noisy_probs *= dropout_mask
        # Add uniform random noise
        if uniform_noise > 0:
            uniform_additions = np.random.uniform(0, uniform_noise, 48)
            noisy_probs += uniform_additions
        # Add Gaussian noise
        if gaussian_noise > 0:
            gaussian_additions = np.random.normal(0, gaussian_noise, 48)
            noisy_probs += gaussian_additions
        # Apply probability swapping
        if swap_noise > 0:
            n_swaps = int(swap_noise * 48 * np.random.random())
            for _ in range(n_swaps):
                i, j = np.random.choice(48, 2, replace=False)
                noisy_probs[i], noisy_probs[j] = noisy_probs[j], noisy_probs[i]
        # Ensure no negative probabilities
        noisy_probs = np.maximum(noisy_probs, 0)
        # Handle case where all probabilities become zero
        if noisy_probs.sum() == 0:
            noisy_probs = np.ones(48) / 48.0  # Fallback to uniform
        print("end of noise generation")
        return greek_char_prob_field(noisy_probs)
    def __str__(self) -> str:
        top_5 = self.get_top_k(5)
        top_str = ", ".join([f"{char}: {prob:.3f}" for char, prob in top_5])
        return f"GreekCharProbability(top 5: {top_str})"
    def __repr__(self) -> str:
        return f"GreekCharProbability(entropy={self.entropy():.3f})"

def prob_field_tests():
    # Create uniform distribution
    uniform_dist = greek_char_prob_field()
    print("Uniform distribution:")
    print(uniform_dist)
    print(f"Entropy: {uniform_dist.entropy():.3f}")
    print()
    
    # Create distribution from dictionary
    char_probs = {
        'α': 0.3, 'ε': 0.2, 'ο': 0.15, 'ι': 0.1, 'η': 0.08,
        'τ': 0.05, 'ν': 0.04, 'σ': 0.03, 'ρ': 0.03, 'κ': 0.02
    }
    custom_dist = greek_char_prob_field(char_probs)
    alpha_char = greek_char_prob_field(start_char='α')
    print("Custom distribution:")
    print(custom_dist)
    print("Defined with alpha")
    print(alpha_char)
    print(f"Most likely: {custom_dist.most_likely_char()}")
    print(f"P(α) = {custom_dist.get_probability('α'):.3f}")
    print(f"Sample: {custom_dist.sample(5)}")
    
    # Demonstrate noise application
    print("Applying different types of noise:")
    
    # Light uniform noise
    noisy1 = custom_dist.add_noise(uniform_noise=0.05, random_seed=42)
    print(f"With uniform noise: {noisy1}")
    
    # Gaussian noise
    noisy2 = custom_dist.add_noise(gaussian_noise=0.02, random_seed=42)
    print(f"With Gaussian noise: {noisy2}")
    
    # Dropout
    noisy3 = custom_dist.add_noise(dropout_rate=0.3, random_seed=42)
    print(f"With dropout: {noisy3}")
    
    # Combined noise
    noisy4 = custom_dist.add_noise(
        uniform_noise=0.03, 
        gaussian_noise=0.01, 
        dropout_rate=0.1, 
        swap_noise=0.2, 
        random_seed=42
    )
    print(f"With combined noise: {noisy4}")
    print(f"Original entropy: {custom_dist.entropy():.3f}")
    print(f"Noisy entropy: {noisy4.entropy():.3f}")
