import numpy as np
import math
from collections import defaultdict
import random

class DPSU:
    def __init__(self, delta1, rho1, sigma1):
        """
        Parameters:
        delta1: Maximum contribution for 1-grams
        rho1: Threshold for 1-grams
        sigma1: Noise for 1-grams
        """
        self.delta1 = delta1
        self.rho1 = rho1
        self.sigma1 = sigma1

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

    def calculate_S1(self, user_data):
        """
        user_data: list of sets (one per user) containing 1-grams
        Returns: (S1, S2) - DP 1-grams and DP 2-grams
        """
        # Process 1-grams
        S1 = self._build_dp_set(user_data, self.delta1, self.rho1, self.sigma1)
        return S1
