import numpy as np
import math
import random
from DPSU import DPSU
from collections import defaultdict
from util import estimate_valid_k_grams, prune_invalid, clean_text
from scipy.stats import norm
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# Indexing issues : the parameters of k-gram are stored in k-1 index of the lists

class DPNE:
    def __init__(self, user_data, epsilon, delta, p, T, eta = 0.01, Deltas = [], sigmas = [], rhos = []):
        """
        user_data: list of n users data given in the form of a single large string
        T: maximum length of k-grams to be extracted
        epsilon: privacy parameter
        delta: privacy parameter
        p: Probability parameter
        eta: privacy parameter
        """
        self.user_data = [clean_text(text) for text in user_data] # remove non-alphanumeric characters
        self.epsilon = epsilon
        self.delta = delta
        self.p = p
        self.T = T
        self.eta = eta
        self.Deltas = []
        self.rhos = []
        self.sigmas = []
        
        if len(Deltas):
            self.Deltas = Deltas
        else:
            self.calculate_deltas()
        if len(sigmas):
            self.sigmas = sigmas
        else:
            self.calculate_sigmas()
        if len(rhos):
            self.rhos = rhos
        else:
            self.calculate_rhos()


    def calculate_deltas(self):
        """
        Calculate deltas for each k-gram.
        We will set the same delta for all k-grams. 
        The value is the median of k-gram counts of the users.
        """
        # Step 1: Calculate the median of k-gram counts for each user (We are considering 1-grams)
        kgram_counts = []
        for i in range(len(self.user_data)):
            kgram_counts.append(len(self.user_data[i].split(" ")))
        median = np.median(kgram_counts)
        # Step 2: Set delta to the median value
        self.Deltas = [median] * self.T

    def calculate_sigmas(self):
        """
        We will set the same sigma for all k-grams.
        The value is obtained by solving the equation 1.
        """
        def equation(sigma):
            term1 = norm.cdf(-self.epsilon * sigma + 1/(2*sigma))
            term2 = np.exp(self.epsilon) * norm.cdf(-self.epsilon * sigma - 1/(2*sigma))
            return term1 - term2 - self.delta/2
    
        # Initial guess and bounds (σ* > 0)
        result = root_scalar(equation, bracket=[1e-5, 100000], method='brentq')
        self.sigmas = [result.root * math.sqrt(self.T)] * self.T

    def calculate_rhos(self):
        """
        Precompute rho1 and rho2 before running the algorithm.
        rho1 uses a privacy guarantee formula, rho2 (and later) controls spurious k-grams.
        """
        # Compute rho1 using Theorem 2.1 from the paper
        max_rho = -np.inf
        for t in range(1, int(self.Deltas[0]) + 1):
            quantile = norm.ppf((1 - self.delta / 2) ** (1 / t))
            current_rho = 1 / math.sqrt(t) + self.sigmas[0] * quantile
            max_rho = max(max_rho, current_rho)
        self.rhos.append(max_rho)  # rho1
        # Calculating rho2, rho3, ..., rhoT is done in the run() method while executing the for loop

    def generate_kgrams_from_text(self, text, k):
        """
        text: string (a full sentence or document)
        k: length of k-gram
        """
        tokens = text.split()
        s_k = []
        for i in range(len(tokens) - k + 1):
            s_k.append(" ".join(tokens[i:i+k]))
        return s_k


    def run(self, filename = 'k_grams'):
        """
        self.user_data: list of n users data where each user i has some subset Wi^k of k-grams.(List of sets)
        """
        data = [] # stores the 1-grams of users
        S = [] # stores i-grams for each i
        V = [] # stores the valid k-grams for each i
        for i in range(len(self.user_data)):
            data.append(self.generate_kgrams_from_text(self.user_data[i], 1))
        
        # Step 1 : Run DPSU to get S1
        print("Delta_0 : ", self.Deltas[0])
        print("Sigma_0 : ", self.sigmas[0])
        print("Rho_0 : ", self.rhos[0])
        print("Length of Deltas : ", len(self.Deltas))
        print("Length of sigmas : ", len(self.sigmas))
        print("Length of rhos : ", len(self.rhos))

        dpsu = DPSU(self.Deltas[0], self.rhos[0], self.sigmas[0])
        S1 = dpsu.calculate_S1(data)
        
        # Check if S1 is empty
        if not S1:
            print("S1 is empty. The algorithm cannot proceed.")
            return []
        
        quantile_2 = norm.ppf(1 - self.eta/len(S1))
        rho2 = self.sigmas[1] * quantile_2
        self.rhos.append(rho2)  # rho2

        dpsu2 = DPSU(self.Deltas[1], self.rhos[1], self.sigmas[1])
        S2 = dpsu2.calculate_S2(data, S1)
        
        print("S1 length: ", len(S1))
        print("S2 length: ", len(S2))
        S.append(S1)
        S.append(S2)
        V1 = S1
        V2 = S2
        V.append(V1)
        V.append(V2)

        # Step 2 : Iteratatively calculate S2, S3, ..., ST
        for k in range(3, self.T+1):
            # Check if the previous set is empty
            if not S[k-2]:
                print(f"S{k-2} is empty. Skipping iteration {k}.")
                S.append(set())  # Add empty set for this k-gram level
                continue
                
            Vk = estimate_valid_k_grams(S1, S[k-2], self.p)
            print(f"V{k} : ", Vk)
            
            # If Vk is 0, we can't generate meaningful k-grams for this level
            if Vk == 0:
                print(f"V{k} is 0. Skipping iteration {k}.")
                S.append(set())  # Add empty set for this k-gram level
                continue
                
            # estimating the value of rho_k
            rho_k = self.sigmas[k-1] * norm.ppf(1 - self.eta*min(1, len(S[k-2])/(Vk+1e-7)))
            self.rhos.append(rho_k)

            # building a histogram with gaussian update policy
            Hk = defaultdict(float)

            for i in range(len(self.user_data)):
                W = self.generate_kgrams_from_text(self.user_data[i], k)
                W = prune_invalid(W, S1, S[k-2])

                if len(W) >= self.Deltas[k-1]:
                    # uniformly sample delta items from W
                    W = np.random.choice(list(W), self.Deltas[k-1], replace=False)
                elif not W:  # If W is empty, skip this user
                    continue

                # build the histogram
                for w in W:
                    Hk[w] += 1/math.sqrt(len(W))
            
            # step 3 : adding noise to the histogram and calculate Sk
            Sk = set()
            for key, value in Hk.items():
                noise = np.random.normal(0, self.sigmas[k-1])
                if value + noise > rho_k:
                    Sk.add(key)

            # step 4 : adding spurious k-grams to Sk
            supp_Hk = {key for key, value in Hk.items() if value > 0}
            print(f"Length of supp_H{k} : ", len(supp_Hk))
            
            # Handle the case where Vk might be 0 or very small
            if Vk - len(supp_Hk) <= 0:
                print(f"No need to add spurious k-grams for iteration {k}.")
                S.append(Sk)
                print(f"Completed {k}th iteration")
                print(f"len(S{k}) : ", len(Sk))
                continue
                
            Bk = np.random.binomial(max(1, Vk - len(supp_Hk)), norm.cdf(-self.rhos[k-1]/self.sigmas[k-1]))
            SPk = set()
            print(f"B{k} : ", Bk)
            
            # No need to add spurious k-grams if Bk is 0
            if Bk == 0:
                Sk = Sk.union(SPk)
                S.append(Sk)
                print(f"Completed {k}th iteration")
                print(f"len(S{k}) : ", len(Sk))
                continue
                
            attempt_count = 0
            max_attempts = 10000
            while len(SPk) < Bk:
                attempt_count += 1
                if attempt_count > max_attempts:
                    print(f"Couldn't sample enough SP{k}. Reached {len(SPk)} out of {Bk} required.")
                    break
                    
                # Check if S[0] or S[k-2] is empty to avoid IndexError
                if not S[0] or not S[k-2]:
                    print(f"Cannot create spurious k-grams for iteration {k} - required sets are empty.")
                    break
                    
                x = random.choice(list(S[0]))
                w = random.choice(list(S[k-2]))

                # w = yz
                y = " ".join(w.rsplit(" ", 1)[:-1])
                z = w.split(" ")[-1]
                
                new_gram = f"{x} {w}"

                xy = x if y == "" else f"{x} {y}"
                if (xy in S[k-2]) and (z in S1) and (new_gram not in SPk.union(supp_Hk)):
                    SPk.add(new_gram)
                    attempt_count = 0  # Reset counter when we successfully add an element
            
            Sk = Sk.union(SPk)
            S.append(Sk)
            print(f"Completed {k}th iteration")
            print(f"len(S{k}) : ", len(Sk))
        
        # The rest of the function remains unchanged
        import os
        plots_dir = "plots"
        ngrams_dir = "n_grams"
        
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(ngrams_dir, exist_ok=True)
        
        # plot the length of k grams produced
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.T+1), [len(S[i]) for i in range(self.T)], marker='o')
        plt.title('Length of k-grams produced')
        plt.xlabel('k')
        plt.ylabel('Length of k-grams')
        plt.xticks(range(1, self.T+1))
        plt.grid()
        plot_path = os.path.join(plots_dir, f'{filename}.png')
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
        
        # Save the n-grams to a file
        import pickle
        ngram_path = os.path.join(ngrams_dir, f'{filename}.pkl')
        with open(ngram_path, 'wb') as f:
            pickle.dump(S, f)
        print(f"N-grams saved to: {ngram_path}")
        
        return S

            