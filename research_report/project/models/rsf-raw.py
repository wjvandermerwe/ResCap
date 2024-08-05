import numpy as np

class SurvivalTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, survival_curve=None, is_leaf=False):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.survival_curve = survival_curve
        self.is_leaf = is_leaf


class RandomSurvivalForest:
    def __init__(self, n_estimators=100, max_features='sqrt', max_depth=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.oob_indices = []

    def fit(self, X, T, E):
        n_samples, n_features = X.shape
        self.oob_indices = np.zeros((self.n_estimators, n_samples), dtype=bool)

        for i in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            oob_indices = np.setdiff1d(np.arange(n_samples), indices)
            self.oob_indices[i, oob_indices] = True

            if self.max_features == 'sqrt':
                features = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
            else:
                features = np.random.choice(n_features, self.max_features, replace=False)

            X_sample = X[indices][:, features]
            T_sample = T[indices]
            E_sample = E[indices]

            tree = self._build_tree(X_sample, T_sample, E_sample, features)
            self.trees.append((tree, features))

    def _build_tree(self, X, T, E, features, depth=0):
        if depth == self.max_depth or len(X) < self.min_samples_split:
            survival_curve = self._compute_survival_curve(T, E)
            return SurvivalTreeNode(survival_curve=survival_curve, is_leaf=True)

        best_feature, best_threshold = self._best_split(X, T, E, features)
        if best_feature is None:
            survival_curve = self._compute_survival_curve(T, E)
            return SurvivalTreeNode(survival_curve=survival_curve, is_leaf=True)

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left = self._build_tree(X[left_indices], T[left_indices], E[left_indices], features, depth + 1)
        right = self._build_tree(X[right_indices], T[right_indices], E[right_indices], features, depth + 1)

        return SurvivalTreeNode(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def _best_split(self, X, T, E, features):
        best_feature, best_threshold = None, None
        best_impurity = -np.inf

        for feature in features:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                if len(T[left_indices]) == 0 or len(T[right_indices]) == 0:
                    continue

                impurity = self._log_rank_statistic(T[left_indices], E[left_indices], T[right_indices],
                                                    E[right_indices])
                if impurity > best_impurity:
                    best_impurity = impurity
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _log_rank_statistic(self, T_left, E_left, T_right, E_right):
        event_times = np.unique(np.concatenate([T_left, T_right]))
        observed_left = np.array([np.sum((T_left == t) & (E_left == 1)) for t in event_times])
        observed_right = np.array([np.sum((T_right == t) & (E_right == 1)) for t in event_times])
        at_risk_left = np.array([np.sum(T_left >= t) for t in event_times])
        at_risk_right = np.array([np.sum(T_right >= t) for t in event_times])

        O = np.concatenate([observed_left, observed_right])
        E = np.concatenate([at_risk_left * np.sum(O) / np.sum(at_risk_left + at_risk_right),
                            at_risk_right * np.sum(O) / np.sum(at_risk_left + at_risk_right)])
        V = E * (1 - E / np.sum(O))

        return np.sum((O - E) ** 2 / V)

    def _compute_survival_curve(self, T, E):
        event_times = np.sort(np.unique(T[E == 1]))
        n = len(T)
        survival_curve = np.ones_like(event_times, dtype=float)
        for i, t in enumerate(event_times):
            survival_curve[i] = (n - np.sum(T <= t)) / n
        return survival_curve

    def predict(self, X):
        survival_curves = np.zeros((X.shape[0], len(self.trees[0][0].survival_curve)))
        for tree, features in self.trees:
            survival_curves += self._predict_tree(tree, X[:, features])
        return survival_curves / self.n_estimators

    def _predict_tree(self, tree, X):
        if tree.is_leaf:
            return np.tile(tree.survival_curve, (X.shape[0], 1))

        feature = tree.feature
        threshold = tree.threshold
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold

        predictions = np.zeros((X.shape[0], len(tree.survival_curve)))
        predictions[left_indices] = self._predict_tree(tree.left, X[left_indices])
        predictions[right_indices] = self._predict_tree(tree.right, X[right_indices])
        return predictions

    def oob_score(self, X, T, E):
        oob_predictions = np.zeros_like(T, dtype=float)
        oob_counts = np.zeros_like(T, dtype=int)

        for i, (tree, features) in enumerate(self.trees):
            oob_indices = self.oob_indices[i]
            if np.sum(oob_indices) == 0:
                continue

            oob_predictions[oob_indices] += self._predict_tree(tree, X[oob_indices][:, features])[:, -1]
            oob_counts[oob_indices] += 1

        oob_predictions /= oob_counts
        oob_predictions[oob_counts == 0] = 0  # Set predictions to zero where there were no OOB samples
        concordance = self._concordance_index(T, E, oob_predictions)
        return concordance

    def _concordance_index(self, T, E, predictions):
        concordant = 0
        permissible = 0
        n = len(T)

        for i in range(n):
            for j in range(i + 1, n):
                if T[i] == T[j]:
                    continue
                if (E[i] == 1 and T[i] < T[j]) or (E[j] == 1 and T[j] < T[i]):
                    permissible += 1
                    if (E[i] == 1 and T[i] < T[j] and predictions[i] < predictions[j]) or \
                            (E[j] == 1 and T[j] < T[i] and predictions[j] < predictions[i]):
                        concordant += 1

        return concordant / permissible if permissible > 0 else 0

    def variable_importance(self, X, T, E):
        baseline_score = self.oob_score(X, T, E)
        importances = np.zeros(X.shape[1])

        for feature in range(X.shape[1]):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, feature])

            permuted_score = self.oob_score(X_permuted, T, E)
            importances[feature] = baseline_score - permuted_score

        return importances

# Example data
X = np.array([[0.5, 1.2], [1.3, 0.7], [2.1, 1.9], [1.0, 0.8], [1.8, 1.3]])
T = np.array([5, 10, 15, 20, 25])  # Survival times
E = np.array([1, 0, 1, 1, 0])  # Event occurred or censored

# Fit model
rsf = RandomSurvivalForest(n_estimators=10, max_features='sqrt', max_depth=3)
rsf.fit(X, T, E)

# Predict
predictions = rsf.predict(X)
print("Predictions:", predictions)

# OOB score
oob_score = rsf.oob_score(X, T, E)
print("OOB Score (Concordance Index):", oob_score)