import numpy as np


class GaussianNB:
    def __init__(self, var_smoothing: float = 1e-9):
        # var_smoothing: mala konstanta da “umiri” varijanse (sprecava deljenje nulom)
        self.var_smoothing = float(var_smoothing)
        self.class_prior_ = None
        self.classes_ = None
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y, X_val=None, y_val=None):
        # ucenje Gaussovog NB: procena sredina, varijansi i priora po klasi
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")

        # eksplicitno zapamtimo redosled klasa i y kao indekse klasa
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        nC = len(self.classes_)
        counts = np.bincount(y_enc, minlength=nC).astype(float)
        self.class_prior_ = counts / counts.sum()  # P(y=c)

        means = []
        vars_ = []

        # globalna varijansa za stabilnost (za var_smoothing)
        global_var = np.var(X, axis=0)
        eps = self.var_smoothing * max(global_var.max(), 1e-12)

        for c in range(nC):
            Xc = X[y_enc == c]
            if len(Xc) == 0:
                # Ako nema primera za klasu
                means.append(np.zeros(X.shape[1]))
                vars_.append(np.ones(X.shape[1]))
            else:
                means.append(Xc.mean(axis=0))
                v = Xc.var(axis=0) + eps  # blago uvećanje varijanse
                v[v <= 0] = eps           # sigurnosna mreža protiv nule/negativne
                vars_.append(v)

        self.mean_ = np.vstack(means)  # shape: (n_classes, n_features)
        self.var_ = np.vstack(vars_)
        return self

    def _jll(self, X):
        # Joint log-likelihood: log P(X|y=c) + log P(y=c) za svaku klasu
        X = np.asarray(X, dtype=float)
        nC = self.mean_.shape[0]
        out = np.empty((X.shape[0], nC), dtype=float)
        for c in range(nC):
            mean = self.mean_[c]
            var = self.var_[c]
            # Zbir po dimenzijama: log N(x|μ,σ²) = -0.5[log(2πσ²) + ((x-μ)²/σ²)]
            logp = -0.5 * (np.sum(np.log(2.0 * np.pi * var)) + np.sum(((X - mean) ** 2) / var, axis=1))
            logp += np.log(self.class_prior_[c] + 1e-15)  # dodaj prior klase
            out[:, c] = logp
        return out

    def predict_proba(self, X):
        # Softmax nad JLL (stabilizovan oduzimanjem max-a po redu)
        X = np.asarray(X, dtype=float)
        jll = self._jll(X)
        a = jll - jll.max(axis=1, keepdims=True)
        e = np.exp(a)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X, threshold=0.5):
        # ako je binarno i dat prag – koristi P(klasa=1) >= threshold
        proba = self.predict_proba(X)
        if proba.shape[1] == 2 and threshold is not None:
            return np.where(proba[:, 1] >= float(threshold), self.classes_[1], self.classes_[0])
        # Inače uzmi klasu sa najvećom verovatnoćom
        return self.classes_[np.argmax(proba, axis=1)]
