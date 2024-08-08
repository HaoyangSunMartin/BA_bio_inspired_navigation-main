import numpy as np
from scipy.ndimage.filters import gaussian_filter

from system.helper import compute_angle

# Image processing approach to identify peaks of grid cell spiking and match them accordingly.
# Used for comparison of different decoders


def compute_vec_in_dir(s, t, n, vec_exp):

    dx1 = t[0] - s[0]
    dx2 = (n - abs(dx1)) * (-np.sign(dx1))

    dx = min([dx1, dx2], key=lambda x: abs(x-vec_exp[0]))

    dy1 = t[1] - s[1]
    dy2 = (n - abs(dy1)) * (-np.sign(dy1))

    dy = min([dy1, dy2], key=lambda x: abs(x - vec_exp[1]))

    return [dx, dy]


def find_spikes(w):

    n = int(np.sqrt(len(w)))
    sheet = np.reshape(w, (n, n))
    sheet_blurred = gaussian_filter(sheet, sigma=0.5)

    # plot3DSheet(sheet)

    spikes = []
    a = np.array(sheet_blurred)

    for i in range(4):
        # find index
        index = np.unravel_index(a.argmax(), a.shape)
        # print(index)
        spikes.append((index[1], index[0]))

        # clean spike
        xmin = max(0, min(index[0]-5, n))
        xmax = max(0, min(index[0]+5, n))
        ymin = max(0, min(index[1]-5, n))
        ymax = max(0, min(index[1]+5, n))
        a[xmin:xmax, ymin:ymax] = 0.
        # plotSheet(a)

    return spikes


def match_spikes(s_max, t_max, vec_exp):
    matches = {}
    vectors = {}

    d_exp = np.linalg.norm(vec_exp)
    for s_spike in s_max:
        for t_spike in t_max:
            vec_calc = compute_vec_in_dir(s_spike, t_spike, 40, vec_exp)
            d = np.linalg.norm(vec_calc)

            if d_exp < 7:
                if s_spike in matches.keys():
                    vec_prior = vectors[s_spike]
                    d_prior = np.linalg.norm(vec_prior)
                    if d < d_prior:
                        matches[s_spike] = t_spike
                        vectors[s_spike] = vec_calc
                else:
                    matches[s_spike] = t_spike
                    vectors[s_spike] = vec_calc
            else:
                if 0.7 * d_exp - 2 < d < 1.3 * d_exp + 2:
                    angle = abs(compute_angle(vec_exp, vec_calc))
                    if angle < np.pi/6:
                        if s_spike not in matches.keys():
                            matches[s_spike] = t_spike
                            vectors[s_spike] = vec_calc
                        # else:
                            # print("Apparently detected two valid matches...", s_spike, t_spike)

    # print("Matches", matches)
    # print("Vectors", vectors)

    return [matches, vectors]


class SpikeDetector:
    def __init__(self):
        self.matches_dict = {}
        self.vector_dict = {}

    def compute_direction_signal(self, gc_modules):
        vec_avg_array = []
        vectors_array = []
        matches_array = []
        gm_array = []
        for i, gc in enumerate(gc_modules):
            # print("Looking at Model i with gm", i, gc.gm)
            gm_array.append(gc.gm)
            s_max = find_spikes(gc.s)
            t_max = find_spikes(gc.t)
            # plotSheetsWithMaxima(gc.s, gc.t, s_max, t_max)

            if i == 0:
                vec_exp = np.array([0, 0])
            else:
                vec_prior = vec_avg_array[i - 1]
                factor = gm_array[i] / gm_array[i - 1]
                vec_exp = vec_prior * factor
            [matches, vectors] = match_spikes(s_max, t_max, vec_exp)

            matches_array.append(matches)
            vectors_array.append(vectors)

            vec_array = list(vectors.values())

            if len(vec_array) == 0:
                vec_avg_array.append(vec_exp)
                # print("Had problems to match in module...", i, gc.gm, vec_exp)
            else:
                vec_avg = np.sum(vec_array, axis=0) / len(vec_array)
                vec_avg_array.append(vec_avg)
                # print("Vec avg", vec_avg)

        vec_avg_array_scaled = []
        for i, vec_avg in enumerate(vec_avg_array):
            vec_avg_array_scaled.append(vec_avg / gm_array[i])

        vec_avg_overall = np.average(vec_avg_array_scaled, axis=0, weights=gm_array)
        # print("Overall Vec avg", vec_avg_overall)

        # self.matches_dict[nr_steps] = matches_array
        # self.vector_dict[nr_steps] = vectors_array

        # plotCurrentAndTargetMatched(gc_network.gc_modules, matches_array, vectors_array)

        return vec_avg_overall

