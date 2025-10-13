import numpy as np
def _sigmoid(z): return 1.0/(1.0+np.exp(-z))


class LogisticRegression:
    def __init__(self, lr=0.05, l2=0.0, max_epochs=3000, tol=1e-6, patience=100, random_state=42):
        self.lr = lr
        self.l2 = l2
        self.max_epochs = max_epochs
        self.tol = tol
        self.patience = patience
        self.W = None
        self.b = 0.0
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X,dtype=float)
        y = y.astype(float)
        n, d = X.shape
        self.W = np.zeros(d)
        self.b = 0.0
        best = np.inf
        bad = 0
        for epoch in range(self.max_epochs):
            p = _sigmoid(X.dot(self.W)+self.b)
            gW = (X.T.dot(p-y))/n + self.l2*self.W
            gb = np.mean(p-y)
            self.W -= self.lr*gW
            self.b -= self.lr*gb
            if X_val is not None:
                pv = _sigmoid(np.asarray(X_val).dot(self.W)+self.b)
                lv = -(y_val*np.log(np.clip(pv,1e-12, 1-1e-12)) + (1-y_val)*np.log(np.clip(1-pv,1e-12,1-1e-12))).mean() + 0.5*self.l2*np.sum(self.W**2)
                if lv<best-self.tol:
                    best = lv
                    bad = 0
                    BW = self.W.copy()
                    Bb = self.b
                else:
                    bad += 1
                    if bad >= self.patience:
                        self.W = BW
                        self.b = Bb
                        break

        return self

    def predict_proba(self, X):
        z = np.asarray(X).dot(self.W)+self.b
        p = _sigmoid(z)
        return np.vstack([1-p,p]).T

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:,1] >= threshold).astype(int)
