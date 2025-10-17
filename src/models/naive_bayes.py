import numpy as np


class GaussianNB:
    """
    Gaussian Naive Bayes (GNB) klasifikator.

      - mean_[c, j] = srednja vrednost featura j u klasi c
      - var_[c, j]  = varijansa featura j u klasi c (uvećana za mali eps radi stabilnosti)
      - class_prior_[c] = P(y=c) = relativna učestalost klase c

    Oblici:
      - X: (n_samples, n_features)
      - y: (n_samples,)
      - classes_: (n_classes,)
      - mean_, var_: (n_classes, n_features)
      - class_prior_: (n_classes,)
    """

    def __init__(self, var_smoothing: float = 1e-9):
        # var_smoothing: mala konstanta za numeričku stabilnost (dodaje se varijansama).

        self.var_smoothing = float(var_smoothing)
        self.class_prior_ = None   # P(y=c) po klasama
        self.classes_ = None       # unikatne vrednosti iz y (npr. [0, 1])
        self.mean_ = None          # srednje vrednosti po klasi i feature-u, (n_classes, n_features)
        self.var_ = None           # varijanse po klasi i feature-u, (n_classes, n_features)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Uči parametre GNB-a iz podataka.

        X: (n_samples, n_features)
        y: (n_samples,)
        """
        # Konverzija na numpy i osnovne validacije dimenzija
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2D. (n_samples, n_features)")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")

        # Eksplicitno zadržavanje redosleda klasa i enkodovanje y u [0..nC-1]
        self.classes_, y_enc = np.unique(y, return_inverse=True)  # classes_: sortirane klase
        nC = len(self.classes_)

        # Prior verovatnoće: P(y=c) = count(c)/N
        counts = np.bincount(y_enc, minlength=nC).astype(float)
        self.class_prior_ = counts / counts.sum()  # oblik: (n_classes,)

        means = []  # akumulira mean po klasi → (n_features,)
        vars_ = []  # akumulira var  po klasi → (n_features,)

        # Određivanje eps na osnovu globalnog opsega varijansi (po feature-ima).
        # max(global_var) obezbeđuje da eps bude "razmeran" skali podataka.
        global_var = np.var(X, axis=0)                 # (n_features,)
        eps = self.var_smoothing * max(global_var.max(), 1e-12)

        # Procena parametara po klasama
        for c in range(nC):
            Xc = X[y_enc == c]  # svi uzorci koji pripadaju klasi c → oblik: (n_c, n_features)
            if len(Xc) == 0:
                # neutralni parametri da model ne pukne.
                means.append(np.zeros(X.shape[1]))
                vars_.append(np.ones(X.shape[1]))
            else:
                # Sredina po koloni (feature-u)
                means.append(Xc.mean(axis=0))
                # Varijansa po koloni; dodaj eps radi stabilnosti (sprečava deljenje nulom)
                v = Xc.var(axis=0) + eps
                # Ako je usled numerike nešto ≤ 0, podigni na eps
                v[v <= 0] = eps
                vars_.append(v)

        # Uobličavanje u matrice dimenzije (n_classes, n_features)
        self.mean_ = np.vstack(means)
        self.var_ = np.vstack(vars_)
        return self

    def _jll(self, X):
        """
        Joint Log-Likelihood (JLL):
          za svaki uzorak i svaku klasu računa:
            log P(X | y=c) + log P(y=c)

        Pošto pretpostavljamo nezavisnost feature-a:
          log P(X|y=c) = Σ_j log N(x_j | mu_{c,j}, var_{c,j})

        Vraća: matrica oblika (n_samples, n_classes) sa log-skorovima.
        """
        X = np.asarray(X, dtype=float)
        nC = self.mean_.shape[0]
        out = np.empty((X.shape[0], nC), dtype=float)

        for c in range(nC):
            mean = self.mean_[c]  # (n_features,)
            var = self.var_[c]   # (n_features,)

            # Prvi deo: zbir log(2πσ²) preko feature-a je konstanta po uzorku (skalarnо)
            const_term = np.sum(np.log(2.0 * np.pi * var))  # skalar

            # Drugi deo: suma po feature-ima ((x-μ)² / σ²) po uzorku → vektor (n_samples,)
            quad_term = np.sum(((X - mean) ** 2) / var, axis=1)

            # Ukupan log P(X|y=c) = -0.5 * (const_term + quad_term)
            log_likelihood = -0.5 * (const_term + quad_term)

            # Dodaj log priora log P(y=c)
            out[:, c] = log_likelihood + np.log(self.class_prior_[c] + 1e-15)

        return out

    def predict_proba(self, X):
        """
        Računa P(y=c | X) za sve klase (softmax nad JLL).

        Koraci:
          1) jll = log P(X|y=c) + log P(y=c)
          2) stabilizacija: oduzmi po uzorku max(jll) da izbegneš overflow u exp
          3) softmax: exp(a) / sum(exp(a))
        Vraća: (n_samples, n_classes)
        """
        X = np.asarray(X, dtype=float)
        jll = self._jll(X)

        # Stabilizacija softmax-a: oduzmi po redu maksimum (log-sum-exp trik)
        a = jll - jll.max(axis=1, keepdims=True)

        # Prevedi nazad iz log-sveta u “obične” skorove
        e = np.exp(a)

        # Normalizuj da se po klasama sabira na 1
        return e / e.sum(axis=1, keepdims=True)