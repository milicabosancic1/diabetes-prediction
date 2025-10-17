import numpy as np


def _minkowski(a, b, p=2):
    """
    Razdaljina po Minkowskom između SVAKOG reda u 'a' (n_samples, n_features)
    i vektora 'b' (n_features,). Podržava p=1 (L1), p=2 (L2), p=np.inf (Chebyshev),
    i opšti slučaj p>0.
    """
    if p == 1:
        return np.sum(np.abs(a - b), axis=1)
    if p == 2:
        return np.sqrt(np.sum((a - b) ** 2, axis=1))
    if p == np.inf:
        return np.max(np.abs(a - b), axis=1)
    if p <= 0:
        raise ValueError("p must be > 0 or np.inf")
    return np.sum(np.abs(a - b) ** p, axis=1) ** (1.0 / p)


class KNN:
    def __init__(self, n_neighbors=7, p=2, weights="uniform"):
        # k: broj najbližih komšija
        self.k = int(n_neighbors)
        # p za Minkowski (1=L1, 2=L2, inf=Chebyshev)
        self.p = p
        # dodeljivanje tezina: 'uniform' ili 'distance'
        self.weights = weights
        self.X = None
        self.y = None
        self.classes_ = None
        self.n_features_ = None

    def fit(self, X, y, X_val=None, y_val=None):
        # Pamti ceo train skup (lazy learner)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features).")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")
        if self.k < 1:
            raise ValueError("n_neighbors (k) must be >= 1.")
        self.X = X
        self.y = y
        self.classes_ = np.unique(y)  # redosled klasa zaključujemo iz podataka
        self.n_features_ = X.shape[1]
        return self

    def _check_X(self, X):
        # Prilagođava ulaz i radi brzu validaciju dimenzija
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}.")
        return X

    def _neighbor_weights(self, dists):
        # Vraća težine po komšiji u zavisnosti od dodeljenih tezina
        if self.weights == "uniform":
            return np.ones_like(dists, dtype=float)
        if self.weights == "distance":
            # Ako postoji identičan primer (dist=0), ravnomerno podeli težinu među njima
            zero = (dists == 0)
            if np.any(zero):
                w = np.zeros_like(dists, dtype=float)
                w[zero] = 1.0 / zero.sum()
                return w
            # Inače inverzno sa malim eps da izbegnemo /0
            return 1.0 / (dists + np.finfo(float).eps)
        raise ValueError("weights must be 'uniform' or 'distance'.")

    def predict_proba(self, X):
        # Vraća verovatnoće po klasama (redovi = uzorci, kolone = klase po self.classes_)
        if self.X is None:
            raise RuntimeError("Call fit before predict_proba.")
        X = self._check_X(X)
        n_train = len(self.X)
        k = min(self.k, n_train)  # ako je train manji od k, koristi koliko ima
        out = []
        for x in X:
            # Distanca do svih train primera
            d = _minkowski(self.X, x, p=self.p)
            # O(O(n)) izbor k najbližih bez kompletnog sortiranja
            idx = np.argpartition(d, k-1)[:k]
            # Stabilno presortiranje tih k po stvarnoj udaljenosti
            idx = idx[np.argsort(d[idx], kind="stable")]
            yy = self.y[idx]
            dd = d[idx]
            w = self._neighbor_weights(dd)
            probs = np.zeros(len(self.classes_), dtype=float)
            denom = w.sum()
            if denom == 0:
                # ravnomerna podela
                probs[:] = 1.0 / len(self.classes_)
            else:
                for i, c in enumerate(self.classes_):
                    probs[i] = w[(yy == c)].sum() / denom
            out.append(probs)
        return np.vstack(out)

    def predict(self, X, threshold=0.5):
        # Ako je binarna klasifikacija i dat prag, koristi kolonu klase 1
        proba = self.predict_proba(X)
        if proba.shape[1] == 2 and threshold is not None:
            return np.where(proba[:, 1] >= float(threshold), self.classes_[1], self.classes_[0])
        # Inače uzima klasu sa najvećom verovatnoćom
        return self.classes_[np.argmax(proba, axis=1)]
