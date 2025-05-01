import numpy as np
import math
import random
from collections import defaultdict
from scipy.stats import norm
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import os
import pickle

from DPSU import DPSU
from util import estimate_valid_k_grams, prune_invalid, clean_text

class DPNE:
    def __init__(self, user_data, epsilon, delta, p, T, eta=0.01, Deltas=[], sigmas=[], rhos=[]):
        """
        Differentially Private N-gram Extraction (DPNE) main class.
        
        Args:
            user_data (list[str]): List of user documents (raw text).
            epsilon (float): Differential privacy parameter.
            delta (float): Differential privacy parameter.
            p (float): Probability threshold for candidate selection.
            T (int): Maximum length of k-grams to extract.
            eta (float): Confidence parameter.
            Deltas (list[float]): Thresholds for sampling (optional).
            sigmas (list[float]): Gaussian noise std-devs (optional).
            rhos (list[float]): Acceptance thresholds (optional).
        """
        self.user_data = [clean_text(text) for text in user_data]
        self.epsilon = epsilon
        self.delta = delta
        self.p = p
        self.T = T
        self.eta = eta
        self.Deltas = Deltas or self.calculate_deltas()
        self.sigmas = sigmas or self.calculate_sigmas()
        self.rhos = rhos or self.calculate_rhos()

    def calculate_deltas(self):
        """Set Δ (sampling limit) as the median unigram count across users."""
        kgram_counts = [len(user.split()) for user in self.user_data]
        median = np.median(kgram_counts)
        return [median] * self.T

    def calculate_sigmas(self):
        """Compute σ satisfying (ε, δ)-DP using the paper's Equation 1."""
        def equation(sigma):
            term1 = norm.cdf(-self.epsilon * sigma + 1 / (2 * sigma))
            term2 = np.exp(self.epsilon) * norm.cdf(-self.epsilon * sigma - 1 / (2 * sigma))
            return term1 - term2 - self.delta / 2
        
        result = root_scalar(equation, bracket=[1e-5, 1e5], method='brentq')
        sigma_star = result.root * math.sqrt(self.T)
        return [sigma_star] * self.T

    def calculate_rhos(self):
        """Compute ρ₁ (threshold) using Theorem 2.1 from the paper."""
        sigma = self.sigmas[0]
        max_rho = -np.inf
        for t in range(1, int(self.Deltas[0]) + 1):
            quantile = norm.ppf((1 - self.delta / 2) ** (1 / t))
            rho = 1 / math.sqrt(t) + sigma * quantile
            max_rho = max(max_rho, rho)
        return [max_rho]

    def generate_kgrams_from_text(self, text, k):
        """Tokenizes text into k-grams."""
        tokens = text.split()
        return [" ".join(tokens[i:i+k]) for i in range(len(tokens) - k + 1)]

    def run(self, filename='k_grams'):
        """
        Main algorithm loop to compute private k-grams.
        
        Args:
            filename (str): Prefix for saving plots and output files.
        
        Returns:
            list[set]: Sets of discovered k-grams from k=1 to k=T.
        """
        data = [self.generate_kgrams_from_text(user, 1) for user in self.user_data]
        S = []  # Extracted k-gram sets
        V = []  # Estimated valid k-grams

        print(f"Δ₁: {self.Deltas[0]}, σ₁: {self.sigmas[0]}, ρ₁: {self.rhos[0]}")
        
        # Step 1: DPSU on unigrams
        dpsu = DPSU(self.Deltas[0], self.rhos[0], self.sigmas[0])
        S1 = dpsu.calculate_S1(data)

        if not S1:
            print("S₁ is empty. Terminating.")
            return []

        S.append(S1)
        V.append(S1)

        # Step 2: Iteratively compute Sk for k=2 to T
        for k in range(2, self.T + 1):
            print(f"\n=== Iteration k={k} ===")
            prev_S = S[k-2]

            if not prev_S:
                print(f"S{k-1} is empty. Skipping k={k}.")
                S.append(set())
                continue

            Vk = estimate_valid_k_grams(S[0], prev_S, self.p)
            if Vk == 0:
                print(f"V{k} is 0. Skipping k={k}.")
                S.append(set())
                continue
            
            rho_k = self.sigmas[k-1] * norm.ppf(1 - self.eta * min(1, len(prev_S)/(Vk+1e-7)))
            self.rhos.append(rho_k)

            # Histogram collection
            Hk = defaultdict(float)
            for user_text in self.user_data:
                W = self.generate_kgrams_from_text(user_text, k)
                W = prune_invalid(W, S[0], prev_S)
                if not W:
                    continue
                sampled = np.random.choice(list(W), min(len(W), int(self.Deltas[k-1])), replace=False)
                for w in sampled:
                    Hk[w] += 1 / math.sqrt(len(W))

            # DPSU filtering
            Sk = set()
            for key, count in Hk.items():
                noise = np.random.normal(0, self.sigmas[k-1])
                if count + noise > rho_k:
                    Sk.add(key)

            # Spurious k-grams (false positives)
            supp_Hk = {key for key, val in Hk.items() if val > 0}
            diff = max(1, Vk - len(supp_Hk))
            Bk = np.random.binomial(diff, norm.cdf(-self.rhos[k-1]/self.sigmas[k-1]))

            SPk = set()
            attempts = 0
            while len(SPk) < Bk and attempts < 10000:
                x = random.choice(list(S[0]))
                w = random.choice(list(prev_S))
                y = " ".join(w.rsplit(" ", 1)[:-1])
                z = w.split()[-1]
                new_gram = f"{x} {w}"
                xy = x if not y else f"{x} {y}"
                if xy in prev_S and z in S[0] and new_gram not in SPk.union(supp_Hk):
                    SPk.add(new_gram)
                    attempts = 0
                else:
                    attempts += 1

            if attempts >= 10000:
                print(f"Spurious generation capped after {attempts} failed attempts.")
            
            Sk.update(SPk)
            S.append(Sk)
            print(f"Generated S{k} with {len(Sk)} elements")

        # Final plot and save
        os.makedirs("plots", exist_ok=True)
        os.makedirs("n_grams", exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.T+1), [len(S[i]) for i in range(self.T)], marker='o')
        plt.title("Length of k-grams produced")
        plt.xlabel("k")
        plt.ylabel("Number of k-grams")
        plt.grid(True)
        plt.xticks(range(1, self.T+1))
        plt.savefig(f"plots/{filename}.png")
        print(f"Plot saved to plots/{filename}.png")

        with open(f"n_grams/{filename}.pkl", 'wb') as f:
            pickle.dump(S, f)
        print(f"N-grams saved to n_grams/{filename}.pkl")

        return S
