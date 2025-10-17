import numpy as np


def _sigmoid(z):
    """
    Ulaz:  z  -> skalar, vektor ili matrica realnih brojeva
    Izlaz: σ  -> vrednosti u (0, 1), tumačimo kao verovatnoće
    """
    return 1.0 / (1.0 + np.exp(-z))


def _logloss(p, y, l2, W, eps=1e-12):
    """
    BINARNI LOG-LOSS + L2 REGULARIZACIJA
    Ulaz:
      p   : verovatnoće P(y=1) za svaki uzorak
      y   : prave binarne etikete (0 ili 1
      l2  : jačina L2 regularizacije (λ). Ako je 0.0, nema regularizacije.
      W   : vektor težina modela
      eps : mali broj da izbegnemo log(0)

    Formula:
      loss = - mean( y*log(p) + (1-y)*log(1-p) )  +  (λ/2) * ||W||^2
    """
    p = np.clip(p, eps, 1.0 - eps)            # sigurna zona da log() ne "pukne"
    data_term = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    reg_term = 0.5 * l2 * np.sum(W * W)       # L2 penal NEMA bias (samo W)
    return data_term + reg_term


class LogisticRegression:
    """
    Parametri:
      lr         : korak učenja.  Veći -> brže, ali može da osciluje; manji -> stabilnije, ali sporije.
      l2         : jačina L2 regularizacije (λ). 0.0 znači bez L2.
      max_epochs : maksimalan broj epoha (prolaza kroz ceo skup).
      tol        : kriterijum konvergencije na PROMENU loss-a (na treningu).
                   Ako je poboljšanje < tol, prekidamo rano (jednostavan stop).
    """

    def __init__(self, lr=0.05, l2=0.0, max_epochs=3000, tol=1e-6):
        self.lr = lr
        self.l2 = l2
        self.max_epochs = max_epochs
        self.tol = tol

        # Parametri koje učimo:
        self.W = None  # težine (d,)
        self.b = 0.0   # bias (skalar)

    def fit(self, X, y):
        """
        UČENJE PARAMETARA (W, b) PUTEM GRADIJENTNOG SPUSTA
        Ulaz:
          X : (n, d) matrica osobina — n uzoraka, d osobina
          y : (n,)   vektor binarnih etiketa {0,1}
        """

        # 1) Uverimo se da su ulazi u pravom formatu i dobijemo dimenzije
        X = np.asarray(X, dtype=float)         # (n, d)
        y = np.asarray(y, dtype=float)         # (n,)
        n, d = X.shape

        # 2) Inicijalizacija parametara (nule su OK za logističku regresiju)
        self.W = np.zeros(d)                   # (d,)
        self.b = 0.0                           # skalar

        # 3) Početni loss (korisno za merenje poboljšanja)
        p = _sigmoid(X @ self.W + self.b)      # verovatnoće na startu
        prev_loss = _logloss(p, y, self.l2, self.W)

        for epoch in range(self.max_epochs):
            # FORWARD: izračunam trenutne verovatnoće ---
            # z = XW + b -> (n,)

            p = _sigmoid(X @ self.W + self.b)

            # GRADIJENTI
            # Izvod log-loss-a po W (sa L2):
            #   gW = (1/n) * X^T * (p - y) + λ * W

            gW = (X.T @ (p - y)) / n + self.l2 * self.W  # (d,)

            # Izvod log-loss-a po b (BEZ L2 na b):
            #   gb = mean(p - y)
            gb = (p - y).mean()                           # skalar

            # AŽURIRANJAM PARAMETARE (KORAK NIZBRDO)
            # Pomerim se u smeru SUPROTNOM od gradijenta, razmerno lr.

            self.W -= self.lr * gW
            self.b -= self.lr * gb

            # PRATIM NAPRETKA NA TRENINGU (JEDNOSTAVAN STOP) ---
            # Izračunaj novi loss; ako je poboljšanje premalo, prekini.

            cur_loss = _logloss(_sigmoid(X @ self.W + self.b), y, self.l2, self.W)
            if prev_loss - cur_loss < self.tol:
                # Poboljšanje loss-a je manje od zadatog praga tol -> smatramo da smo konvergirali.
                break
            prev_loss = cur_loss

        return self

    def predict_proba(self, X):

        X = np.asarray(X, dtype=float)         # (m, d)
        p1 = _sigmoid(X @ self.W + self.b)     # (m,)
        return np.column_stack([1.0 - p1, p1])  # (m, 2)
