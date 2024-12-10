import copy
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.spatial.distance import cdist, pdist
import warnings
from random import choice


def get_random_point(feas_func, rng, com_step=0.1/9):
    while True:
        # Composition random generator
        xyz = rng.random(3)
        if sum(xyz) > 1.0:
            continue
        if min(xyz) < com_step:
            continue
        if sum(xyz) != 1.0 and (1 - sum(xyz)) < com_step:
            continue
        if not feas_func(xyz.reshape(1, -1)):
            continue
        comp = xyz.tolist() + [1-sum(xyz.tolist())]

        # Morphologies with pre-stretch, OneHotEncoder random generator
        morph_ohe = np.ones(4)
        morph_ohe[:3] = 0
        rng.shuffle(morph_ohe)

        # Pre-stretch random generator
        if morph_ohe[0] == 1 or morph_ohe[1] == 1:
            pre_str = 0
        else:
            pre_str = choice([100, 200, 300])

        # Thickness random generator
        thick = choice([800,1200,1600])

        point = comp + morph_ohe.tolist() + [pre_str, thick]
        return point


def generate_random_compositions(materials, n_comps=30, random_state=None, return_df=True, feas_func=None):

    rng = np.random.default_rng(random_state)
    comps = [get_random_point(feas_func, rng) for _ in range(len(materials))]
    if return_df:
        return pd.DataFrame(comps, columns=materials)
    else:
        return comps[:n_comps]


class UniformCompositionGenerator:
    '''
    Generating spaced-out compositions within design space while maximizing
    prediction variance using Monte Carlo method.
    '''
    def __init__(self, materials=None, n_comps=30, n_iters=500000, existing_comps: pd.DataFrame = None,
                 random_state=None, perf_func=None, **kwargs):
        self.materials = materials
        self.n_comps = n_comps
        self.existing_comps = existing_comps
        self.n_iters = n_iters
        self.random_state = random_state
        self.perf_func = perf_func
        self.kwargs = kwargs
        self._rng = np.random.default_rng(random_state)

    def _get_rand_comp(self, with_pred=False):
        ''' return a random composition '''

        # If the cache does not exist, or the cache is empty, rebuild cache
        comp_cache = getattr(self, '_comp_cache', [])
        if len(comp_cache) == 0:
            self._build_comp_cache()

        # pop one composition, and with its prediction if existed, from cache
        comp = self._comp_cache.pop()
        pred = self._pred_cache.pop() if self.perf_func else None
        if with_pred:
            return comp, pred
        else:
            return comp

    def _build_comp_cache(self):
        ''' build cache containing a list of random composition '''
        seed = self._rng.integers(0, np.iinfo(np.int64).max)
        comps = generate_random_compositions(self.materials, n_comps=self.n_comps * 10,
                                             random_state=seed, return_df=False, **self.kwargs)
        self._comp_cache = [c for c in comps]
        if self.perf_func:
            self._pred_cache = [c for c in self.perf_func(self._comp_cache)]

    def optimize(self):
        global comps, idx
        # Initialize comps and calculate its score
        best_comps = [self._get_rand_comp() for _ in range(self.n_comps)]
        best_score = self.score_comps(best_comps, perf_func=self.perf_func)

        if self.perf_func:
            # Keep a list of prediction so we don't need to re-predict all of them
            # each time we changed one composition
            best_predictions = self.perf_func(best_comps)

        for i in tqdm(range(self.n_iters), leave=False):
            if self.perf_func:
                # Randomly select a composition, compositions with lower
                # predicted performance are more likely to be selected.
                sorted_idxs = np.argsort(best_predictions)
                p = np.e ** (-1.0 * np.arange(len(sorted_idxs)))
                idx = self._rng.choice(sorted_idxs, p=p / p.sum())[0]
            else:
                # Randomly select a composition
                idx = self._rng.integers(0, len(best_comps))

            # Replace the composition with a new random one
            comps = copy.deepcopy(best_comps)
            new_comp, new_pred = self._get_rand_comp(with_pred=True)
            comps[idx] = new_comp

            # Update predictions
            if self.perf_func:
                predictions = copy.deepcopy(best_predictions)
                predictions[idx] = new_pred
            else:
                predictions = None

            # Update score
            score = self.score_comps(comps, predictions=predictions)

            if score > best_score:
                best_comps = comps
                best_score = score
                if self.perf_func:
                    best_predictions = predictions

        return best_comps

    def score_comps(self, comps, perf_func=None, predictions=None):
        if predictions is None:
            if perf_func is None:
                predictions = [1.0]
            else:
                predictions = perf_func(comps)
        perf_score = np.mean(predictions)

        min_intra_distance = pdist(comps).min()

        if self.existing_comps is not None:
            min_inter_distance = cdist(self.existing_comps, comps).min()
        else:
            min_inter_distance = np.inf

        distance = min(min_intra_distance, min_inter_distance)

        # score of a list of comps = (min paired-distance) * mean(predicted performance)
        score = distance * perf_score

        return score