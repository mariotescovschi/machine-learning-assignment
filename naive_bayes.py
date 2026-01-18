import numpy as np
from collections import defaultdict


class NaiveBayesRanking:
    def __init__(self):
        self.product_counts = defaultdict(int)
        self.cooccurrence = defaultdict(lambda: defaultdict(int))
        self.total_carts = 0
        self.product_prices = {}

    def fit(self, carts, prices):
        self.product_prices = prices
        self.total_carts = len(carts)

        for cart in carts:
            for product in cart:
                self.product_counts[product] += 1
                for other in cart:
                    if other != product:
                        self.cooccurrence[product][other] += 1

    def predict_proba(self, cart, candidate):
        if self.product_counts[candidate] == 0:
            return 0.0

        prior = self.product_counts[candidate] / self.total_carts

        likelihood = 1.0
        for product in cart:
            co_count = self.cooccurrence[candidate][product]
            cand_count = self.product_counts[candidate]
            p = (co_count + 1) / (cand_count + len(self.product_counts))
            likelihood *= p

        return prior * likelihood

    def rank(self, cart, candidates, use_price=True):
        scores = []
        for candidate in candidates:
            if candidate in cart:
                continue
            prob = self.predict_proba(cart, candidate)
            price = self.product_prices.get(candidate, 1.0)
            score = prob * price if use_price else prob
            scores.append((candidate, score, prob))

        return sorted(scores, key=lambda x: x[1], reverse=True)


class PopularityBaseline:
    def __init__(self):
        self.popularity = {}
        self.prices = {}

    def fit(self, carts, prices):
        self.prices = prices
        counts = defaultdict(int)
        total = len(carts)
        for cart in carts:
            for p in cart:
                counts[p] += 1
        self.popularity = {p: c / total for p, c in counts.items()}

    def rank(self, cart, candidates, use_price=True):
        scores = []
        for c in candidates:
            if c in cart:
                continue
            pop = self.popularity.get(c, 0)
            price = self.prices.get(c, 1.0)
            score = pop * price if use_price else pop
            scores.append((c, score, pop))
        return sorted(scores, key=lambda x: x[1], reverse=True)
