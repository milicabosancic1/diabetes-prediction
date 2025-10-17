import numpy as np
def _sigmoid(z): return 1.0/(1.0+np.exp(-z))  # standardna sigmoid funkcija


class LogisticRegression:
    def __init__(self, lr=0.05, l2=0.0, max_epochs=3000, tol=1e-6, patience=100, random_state=42):
        # Hiperparametri: korak učenja, L2 regularizacija i rani prekid (patience/tol)
        self.lr = lr
        self.l2 = l2
        self.max_epochs = max_epochs
        self.tol = tol
        self.patience = patience
        # Parametri modela
        self.W = None
        self.b = 0.0
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y, X_val=None, y_val=None):
        # Jednostavan GD sa L2 regularizacijom i ranim zaustavljanjem po val-loss
        X = np.asarray(X, dtype=float)
        y = y.astype(float)
        n, d = X.shape

        # Inicijalizacija na nule
        self.W = np.zeros(d)
        self.b = 0.0

        best = np.inf  # najbolji validacioni loss do sada
        bad = 0        # broj uzastopnih epizoda bez poboljšanja

        for epoch in range(self.max_epochs):
            # Forward
            p = _sigmoid(X.dot(self.W)+self.b)

            # gradijenti (log-loss + L2 za W, ne za b)
            gW = (X.T.dot(p-y))/n + self.l2*self.W
            gb = np.mean(p-y)

            # korak gradijentnog spustanja
            self.W -= self.lr*gW
            self.b -= self.lr*gb

            # rano zaustavljanje – prati validacioni gubitak ako je dat val skup
            if X_val is not None:
                pv = _sigmoid(np.asarray(X_val).dot(self.W)+self.b)
                # Numericki stabilno logovanje (clip) + L2 penal
                lv = -(y_val*np.log(np.clip(pv, 1e-12, 1-1e-12)) + (1-y_val)*np.log(np.clip(1-pv, 1e-12, 1-1e-12))).mean() + 0.5*self.l2*np.sum(self.W**2)

                if lv < best - self.tol:
                    # Poboljšanje – zapamti “checkpoint”
                    best = lv
                    bad = 0
                    BW = self.W.copy()
                    Bb = self.b
                else:
                    # Nema poboljšanja – brojim “loše” epohe
                    bad += 1
                    if bad >= self.patience:
                        # Vraćam se na najbolju verziju i prekidam treniranje
                        self.W = BW
                        self.b = Bb
                        break

        return self

    def predict_proba(self, X):
        # Vraća [P(y=0), P(y=1)] po uzorku – pogodnije za spoljnu evaluaciju
        z = np.asarray(X).dot(self.W)+self.b
        p = _sigmoid(z)
        return np.vstack([1-p, p]).T

    def predict(self, X, threshold=0.5):
        # Binarna odluka po pragu nad P(y=1)
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
