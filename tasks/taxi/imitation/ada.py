"""ADA code kindly provided by by Luis C. Cobo and adapted as noted."""

import math
import numpy
import random

class MutualInformation:
    def __init__(self):
        self.instances = []
        self.names = []
        self.types = []
        self.n_features = 0
        self.n_labels = 64

    def add_instance(self, str):
        if str == '':
            return
        instance = str.split(',')
        if self.n_features == 0:
            self.n_features = len(instance)
        self.instances.append(instance)

    def add_instance_vector(self, v):
        self.instances.append(v)

    def copy_names(self, obj):
        self.names = obj.names[:]

    def copy_types(self, obj):
        self.types = obj.types[:]

    def set_names(self, str):
        names = str.split(',')
        if self.n_features == 0:
            self.n_features = len(names)
        self.names = names

    def set_types(self, str):
        types = str.split(',')
        if self.n_features == 0:
            self.n_features = len(types)
        self.types = types

    def is_discrete(self, idx):
        if self.types[idx] == 'd':
            return True
        else:
            return False

    def get_bucketized_values(self, idx):
        values = []

        if self.is_discrete(idx):
            values = [instance[idx] for instance in self.instances]
        else:
            vmax = float(self.instances[0][idx])
            vmin = float(self.instances[0][idx])

            for instance in self.instances:
                v = float(instance[idx])
                if v > vmax:
                    vmax = v
                elif v < vmin:
                    vmin = v

            vrange = vmax - vmin
            values_idx = [min(self.n_labels - 1,
                              int(self.n_labels *
                                  (float(instance[idx]) - vmin) / vrange))
                          for instance in self.instances]

            step = float(vrange) / self.n_labels
            labels = [vmin + step * (i + 0.5) for i in range(self.n_labels)]
            values = [labels[v] for v in values_idx]

        return values

    def get_all_bucket_labels(self, idx):
        labels = []

        if self.is_discrete(idx):
            labels = list(set([instance[idx] for instance in self.instances]))
            labels.sort()
        else:
            vmax = float(self.instances[0][idx])
            vmin = float(self.instances[0][idx])

            for instance in self.instances:
                v = float(instance[idx])
                if v > vmax:
                    vmax = v
                elif v < vmin:
                    vmin = v

            vrange = vmax - vmin
            step = float(vrange) / self.n_labels
            labels = [vmin + step * (i + 0.5) for i in range(self.n_labels)]

        return labels

    def marginal_prob1(self, idx, n=1):
        values = self.get_bucketized_values(idx)
        if n:
            values = values[0:int(len(self.instances)/n)]

        outcomes = {}
        for value in values:
            if value not in outcomes:
                outcomes[value] = 1
            else:
                outcomes[value] += 1

        probabilities = {k: float(v)/int(len(self.instances)/n) for k,v in outcomes.items()}
        return probabilities

    def marginal_prob2(self, idx1, idx2, n=1):
        values1 = self.get_bucketized_values(idx1)
        values2 = self.get_bucketized_values(idx2)

        if n:
            values1 = values1[0:int(len(self.instances)/n)]
            values2 = values2[0:int(len(self.instances)/n)]

        outcomes = {}
        for sample in zip(values1, values2):
            if sample not in outcomes:
                outcomes[sample] = 1
            else:
                outcomes[sample] += 1

        probabilities = {k: float(v)/int(len(self.instances)/n) for k,v in outcomes.items()}
        return probabilities

    def get_mutual_info(self, idx1, idx2, n=1):
        p1 = self.marginal_prob1(idx1, n)
        p2 = self.marginal_prob1(idx2, n)
        joint_p = self.marginal_prob2(idx1, idx2, n)

        keys1 = p1.keys()
        keys2 = p2.keys()
        return sum([joint_p[(x, y)] * math.log(joint_p[(x, y)] / (p1[x] * p2[y]), 2)
                    if (x,y) in joint_p else 0
                    for x in keys1 for y in keys2])

    def get_all_mutual_info(self, n=1):
        class_idx = len(self.instances[0]) - 1
        return [self.get_mutual_info(x, class_idx, n) for x in range(len(self.instances[0]) - 1)]


def good_abs(v):
    min_mi = 0.05
    v = v[:]
    v = [x for x in v if x > min_mi]
    v.sort()
    v_diff = [v[i+1] - v[i] for i in range(len(v)-1)]
    if v_diff:  # Avoid error from empty list
        v_bigger_step_idx = v_diff.index(max(v_diff))
        v_lower_range = sum(v_diff[:v_bigger_step_idx])
        v_upper_range = sum(v_diff[v_bigger_step_idx+1:])
        return max(v) > max(v_lower_range, v_upper_range)
    else:
        return False


def chosen_feats(v):
    min_mi = 0.05
    v1 = v[:]
    v1 = [x for x in v1 if x > min_mi]
    v1_sorted = sorted(v1)
    v1_diff = [v1_sorted[i+1] - v1_sorted[i] for i in range(len(v1)-1)]
    if v1_diff:  # Avoid error from empty list
        v1_bigger_step_idx = v1_diff.index(max(v1_diff))
        v1_selected_base = sum(v1_diff[:v1_bigger_step_idx+1])
        v_chosen = {i for i in range(len(v)) if v[i] >= v1_selected_base}
        return v_chosen
    else:
        return set()


def divide_by_threshold(e, feat_idx, threshold, m):
    l = MutualInformation()
    l.copy_names(m)
    l.copy_types(m)

    r = MutualInformation()
    r.copy_names(m)
    r.copy_types(m)

    for i in range(len(e.instances)):
        if e.types[feat_idx] == 'd':
            if e.instances[i][feat_idx] == threshold:
                r.add_instance_vector(e.instances[i])
            else:
                l.add_instance_vector(e.instances[i])
        else:
            if float(e.instances[i][feat_idx]) > threshold:
                r.add_instance_vector(e.instances[i])
            else:
                l.add_instance_vector(e.instances[i])

    return l, r


def norm(v):
    return numpy.sqrt(numpy.dot(v,v))


def normalize(v):
    return v/norm(v)


def get_score(v1, v2):
    v1 = normalize(v1)
    v2 = normalize(v2)
    return sum([ (v1[i] - v2[i])**2 for i in range(len(v1)) ])


def decompose(demos):
    m = MutualInformation()
    m.set_names('row,column,passenger,destination,action')
    m.set_types('d,d,d,d,d')

    for state,action in demos.items():
        if 'pickup' not in state and 'dropoff' not in state:
            m.add_instance(",".join(map(str, state + (action,))))

    full = m.get_all_mutual_info(1)
    candidates = []
    rounds = 10

    for i in [2, 3, 3.3, 4, 8, 16, 32, 64]:
        total = [0] * (len(m.instances[0]) - 1)
        for j in range(rounds):
            random.shuffle(m.instances)
            d = m.get_all_mutual_info(i)
            if 0 in d:  # Avoid error from divide-by-zero
                rounds -= 1
            else:
                percent_diff = [abs(x-y)/y for x,y in zip(full,d)]
                total = [total[n] + percent_diff[n] for n in range(len(percent_diff))]

        if rounds == 0:  # Avoid error from divide-by-zero
            break

        total = [float(x)/rounds for x in total]
        if max(total) < 0.75:  # Increased from 0.35 to improve success rate
            candidates.append(i)
        else:
            break

    if len(candidates) == 0:
        return False  # Too much deviation

    mss = float(len(m.instances)) / max(candidates)
    S = [(m, 'O')]
    T = []

    while S:
        e, desc = S.pop()
        thresholds = {}
        l = r = None

        for feat_idx in range(len(m.instances[0])-1):
            for threshold in e.get_all_bucket_labels(feat_idx):
                l_prev = l
                r_prev = r

                l, r = divide_by_threshold(e, feat_idx, threshold, m)
                if l_prev and l.instances == l_prev.instances:
                    continue

                if min(len(l.instances), len(r.instances)) < mss:
                    continue

                l_mi = l.get_all_mutual_info()
                r_mi = r.get_all_mutual_info()

                if not good_abs(l_mi) and not good_abs(r_mi):
                    continue

                if chosen_feats(l_mi) == chosen_feats(r_mi):
                    continue

                score = get_score(l_mi, r_mi)
                thresholds[score] = (feat_idx, threshold)

        if len(thresholds) == 0:
            tot = e.get_all_mutual_info()
            feats = chosen_feats(tot)
            T.append((e, desc, feats))

            if desc == 'OL':
                if feats == {0, 1, 2}:
                    continue
                else:
                    return False  # Suboptimal features - should use [row, column, passenger] during pickup
            elif desc == 'OR':
                if feats == {0, 1, 3} or feats == {0, 1, 2, 3}:
                    continue
                else:
                    return False  # Suboptimal features - should use [row, column, passenger?, destination] during dropoff
            else:
                return False  # Suboptimal subtasks - should divide into two for pickup and dropoff

        max_score = max(thresholds.keys())
        feat_idx, threshold = thresholds[max_score]

        if (e.names[feat_idx], threshold) != ('passenger', '4'):
            return False  # Suboptimal threshold - should use passenger < 4

        l, r = divide_by_threshold(e, feat_idx, threshold, m)
        S.append((r, desc + 'R'))
        S.append((l, desc + 'L'))

    return True  # Found the optimal decomposition for Taxi
