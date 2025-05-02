import numpy as np
import math
from collections import defaultdict
import random
from util import prune_invalid_2grams, clean_text

class DPSU:
    def __init__(self, delta1, rho1, sigma1):
        """
        Parameters:
        delta1: Maximum contribution for 1-grams
        rho1: Threshold for 1-grams
        sigma1: Noise for 1-grams

        delta2: Maximum contribution for 2-grams
        rho2: Threshold for 2-grams
        sigma2: Noise for 2-grams
        """
        self.delta1 = delta1
        self.rho1 = rho1
        self.sigma1 = sigma1

    import random

    def _build_dp_set(self, data_list, delta, rho, sigma):
        H = defaultdict(float)

        for item in data_list:
            item = list(item)
            if len(item) > int(delta):
                item = random.sample(item, int(delta))

            for x in item:
                H[x] += 1 / math.sqrt(len(item))

        S = set()
        for key, value in H.items():
            noise = np.random.normal(0, sigma)
            if value + noise > rho:
                S.add(key)
        return S


    def _get_2grams(self, word_set):
        """Create 2-grams from a set of words"""
        words = list(word_set)
        return set("{} {}".format(words[i], words[i+1]) for i in range(len(words)-1))

    def calculate_S1(self, user_data):
        """
        user_data: list of sets (one per user) containing 1-grams
        Returns: (S1, S2) - DP 1-grams and DP 2-grams
        """
        # Process 1-grams
        S1 = self._build_dp_set(user_data, self.delta1, self.rho1, self.sigma1)
        return S1
    
    def calculate_S2(self, user_data, S1):
        # Extract and process 2-grams
        user_2grams = [self._get_2grams(user_set) for user_set in user_data]
        valid_2grams = prune_invalid_2grams(user_2grams, S1)
        # print(f"User 2-grams: {user_2grams}")
        S2 = self._build_dp_set(valid_2grams, self.delta1, self.rho1, self.sigma1)
        return S2
