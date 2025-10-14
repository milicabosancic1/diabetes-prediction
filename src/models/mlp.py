import numpy as np

def _relu(x):
    return np.maximum(0.0, x)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))

def _bce_with_logits(z, y):
    z = z.ravel(); y = y.ravel()
    return (np.maximum(0.0, z) - z*y + np.log1p(np.exp(-np.abs(z)))).mean()

class MLP:
    def __init__(self, hidden=64, lr=0.01, l2=1e-4, max_epochs=500, patience=50,
                 random_state=42, batch_size=64, dropout=0.0, grad_clip=None,
                 beta1=0.9, beta2=0.999, eps=1e-8):
        self.h = int(hidden)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.bs = int(batch_size)
        self.dropout = float(dropout)
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        self.beta1, self.beta2, self.eps = float(beta1), float(beta2), float(eps)
        self.rng = np.random.default_rng(random_state)

        self.W1 = self.b1 = self.W2 = self.b2 = None
        self.mW1 = self.vW1 = self.mb1 = self.vb1 = None
        self.mW2 = self.vW2 = self.mb2 = self.vb2 = None
        self._t = 0

    def _init(self, d):
        self.W1 = self.rng.normal(0, np.sqrt(2.0/d), size=(d, self.h))
        self.b1 = np.zeros(self.h)
        self.W2 = self.rng.normal(0, np.sqrt(2.0/self.h), size=(self.h, 1))
        self.b2 = np.zeros(1)
        zW1 = np.zeros_like(self.W1); zb1 = np.zeros_like(self.b1)
        zW2 = np.zeros_like(self.W2); zb2 = np.zeros_like(self.b2)
        self.mW1 = zW1.copy(); self.vW1 = zW1.copy()
        self.mb1 = zb1.copy(); self.vb1 = zb1.copy()
        self.mW2 = zW2.copy(); self.vW2 = zW2.copy()
        self.mb2 = zb2.copy(); self.vb2 = zb2.copy()
        self._t = 0

    def _forward(self, X, train=False):
        Z1 = X.dot(self.W1) + self.b1
        A1 = _relu(Z1)
        if train and self.dropout > 0.0:
            mask = (self.rng.random(A1.shape) >= self.dropout).astype(A1.dtype)
            A1 = A1 * mask / (1.0 - self.dropout)
        Z2 = A1.dot(self.W2) + self.b2
        return Z1, A1, Z2

    def _adam_step(self, W, gW, mW, vW):
        self._t += 1
        mW[:] = self.beta1*mW + (1-self.beta1)*gW
        vW[:] = self.beta2*vW + (1-self.beta2)*(gW*gW)
        m_hat = mW / (1 - self.beta1**self._t)
        v_hat = vW / (1 - self.beta2**self._t)
        W[:] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        self._init(d)

        best_loss = np.inf
        bad = 0
        BW1 = self.W1.copy(); Bb1 = self.b1.copy()
        BW2 = self.W2.copy(); Bb2 = self.b2.copy()

        idx_all = np.arange(n)
        for _ in range(self.max_epochs):
            self.rng.shuffle(idx_all)
            for start in range(0, n, self.bs):
                idx = idx_all[start:start+self.bs]
                Xb, yb = X[idx], y[idx]

                Z1, A1, Z2 = self._forward(Xb, train=True)
                loss = _bce_with_logits(Z2, yb) + 0.5*self.l2*(np.sum(self.W1**2)+np.sum(self.W2**2))

                p = _sigmoid(Z2).ravel()
                dZ2 = (p - yb).reshape(-1,1) / len(Xb)
                dW2 = A1.T.dot(dZ2) + self.l2*self.W2
                db2 = dZ2.sum(axis=0)

                dA1 = dZ2.dot(self.W2.T)
                dZ1 = dA1 * (Z1 > 0).astype(float)
                dW1 = Xb.T.dot(dZ1) + self.l2*self.W1
                db1 = dZ1.sum(axis=0)

                if self.grad_clip is not None:
                    for g in (dW1, db1, dW2, db2):
                        gn = np.linalg.norm(g.ravel())
                        if gn > self.grad_clip:
                            g *= (self.grad_clip / (gn + 1e-12))

                self._adam_step(self.W2, dW2, self.mW2, self.vW2)
                self._adam_step(self.b2, db2, self.mb2, self.vb2)
                self._adam_step(self.W1, dW1, self.mW1, self.vW1)
                self._adam_step(self.b1, db1, self.mb1, self.vb1)

            if X_val is not None:
                _, _, Z2v = self._forward(X_val, train=False)
                val_loss = _bce_with_logits(Z2v, y_val) + 0.5*self.l2*(np.sum(self.W1**2)+np.sum(self.W2**2))
            else:
                _, _, Z2t = self._forward(X, train=False)
                val_loss = _bce_with_logits(Z2t, y) + 0.5*self.l2*(np.sum(self.W1**2)+np.sum(self.W2**2))

            if val_loss < best_loss - 1e-7:
                best_loss = val_loss; bad = 0
                BW1 = self.W1.copy(); Bb1 = self.b1.copy()
                BW2 = self.W2.copy(); Bb2 = self.b2.copy()
            else:
                bad += 1
                if bad >= self.patience:
                    self.W1, self.b1, self.W2, self.b2 = BW1, Bb1, BW2, Bb2
                    break
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        _, _, Z2 = self._forward(X, train=False)
        p = _sigmoid(Z2).ravel()
        return np.vstack([1-p, p]).T

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= float(threshold)).astype(int)
