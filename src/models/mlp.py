import numpy as np


def _relu(x):
    # ReLU – sve negativno na nulu, ostalo ostaje
    return np.maximum(0.0, x)


def _sigmoid(x):
    # Sigmoid sa klipovanjem radi numeričke stabilnosti
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _bce_with_logits(z, y, pos_weight=None, sample_weight=None):
    # Stabilna BCE varijanta: direktno iz logita (bez prethodnog sigmoid-a)
    # z,y shape handling
    z = z.ravel()
    y = y.ravel()
    # stable BCE-with-logits
    loss = (np.maximum(0.0, z) - z*y + np.log1p(np.exp(-np.abs(z))))
    if pos_weight is not None:
        # Teže pozitivne primere (npr. kod disbalansa klasa)
        w = np.where(y == 1.0, float(pos_weight), 1.0)
        loss = loss * w
    if sample_weight is not None:
        loss = loss * sample_weight
        return loss.sum() / (np.sum(sample_weight) + 1e-12)
    return loss.mean()


class MLP:
    def __init__(self, hidden=64, lr=0.01, l2=1e-4, max_epochs=500, patience=50,
                 random_state=42, batch_size=64, dropout=0.0, grad_clip=None,
                 beta1=0.9, beta2=0.999, eps=1e-8, use_adamw=False, weight_decay=1e-4,
                 reduce_on_plateau=True, patience_lr=10, lr_factor=0.5,
                 pos_weight=None):
        # Osnovni hiperparametri i optimizacija
        self.h = int(hidden)
        self.lr0 = float(lr)       # čuvamo početni LR
        self.lr = float(lr)
        self.l2 = float(l2)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.bs = int(batch_size)
        self.dropout = float(dropout)
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        self.beta1, self.beta2, self.eps = float(beta1), float(beta2), float(eps)
        self.use_adamw = bool(use_adamw)
        self.weight_decay = float(weight_decay)
        self.reduce_on_plateau = bool(reduce_on_plateau)
        self.patience_lr = int(patience_lr)
        self.lr_factor = float(lr_factor)
        self.pos_weight = pos_weight
        self.rng = np.random.default_rng(random_state)

        # Težine, momenti (Adam), brojač koraka
        self.W1 = self.b1 = self.W2 = self.b2 = None
        self.mW1 = self.vW1 = self.mb1 = self.vb1 = None
        self.mW2 = self.vW2 = self.mb2 = self.vb2 = None
        self._t = 0

        # Trening istorija i zapamćeni prag
        self.history_ = {"train_loss": [], "val_loss": []}
        self.best_threshold_ = 0.5

    def _init(self, d):
        # inicijalizacija (ReLU-friendly)
        self.W1 = self.rng.normal(0, np.sqrt(2.0/d), size=(d, self.h))
        self.b1 = np.zeros(self.h)
        self.W2 = self.rng.normal(0, np.sqrt(2.0/self.h), size=(self.h, 1))
        self.b2 = np.zeros(1)
        # Adam momenti
        zW1 = np.zeros_like(self.W1)
        zb1 = np.zeros_like(self.b1)
        zW2 = np.zeros_like(self.W2)
        zb2 = np.zeros_like(self.b2)
        self.mW1 = zW1.copy()
        self.vW1 = zW1.copy()
        self.mb1 = zb1.copy()
        self.vb1 = zb1.copy()
        self.mW2 = zW2.copy()
        self.vW2 = zW2.copy()
        self.mb2 = zb2.copy()
        self.vb2 = zb2.copy()
        self._t = 0
        self.lr = self.lr0

    def _forward(self, X, train=False):
        # Jedan prolaz unapred: FC -> ReLU -> (dropout) -> FC
        Z1 = X.dot(self.W1) + self.b1
        A1 = _relu(Z1)
        mask = None
        if train and self.dropout > 0.0:
            # Inverted dropout (skaliranje na train-u)
            mask = (self.rng.random(A1.shape) >= self.dropout).astype(A1.dtype)
            A1 = A1 * mask / (1.0 - self.dropout)
        Z2 = A1.dot(self.W2) + self.b2
        return Z1, A1, Z2, mask

    def _adam_update_inplace(self, W, gW, mW, vW, lr):
        # Jedan Adam korak (bias correction koristi zajednički t)
        mW[:] = self.beta1*mW + (1-self.beta1)*gW
        vW[:] = self.beta2*vW + (1-self.beta2)*(gW*gW)
        m_hat = mW / (1 - self.beta1**self._t)
        v_hat = vW / (1 - self.beta2**self._t)
        W[:] -= lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def fit(self, X, y, X_val=None, y_val=None):
        # Mini-batch trening sa Adam/AdamW, L2, early stopping i LR schedulom
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape
        self._init(d)

        best_loss = np.inf
        bad = 0
        bad_lr = 0
        # Checkpoint-ovi za najbolje težine
        BW1 = self.W1.copy()
        Bb1 = self.b1.copy()
        BW2 = self.W2.copy()
        Bb2 = self.b2.copy()

        idx_all = np.arange(n)
        for epoch in range(self.max_epochs):
            self.rng.shuffle(idx_all)

            # Trening po batch-evima
            for start in range(0, n, self.bs):
                idx = idx_all[start:start+self.bs]
                Xb, yb = X[idx], y[idx]

                Z1, A1, Z2, mask = self._forward(Xb, train=True)

                # BCE with logits + opcioni L2 (ako nije AdamW)
                base_loss = _bce_with_logits(Z2, yb, pos_weight=self.pos_weight)
                reg = 0.5*self.l2*(np.sum(self.W1**2)+np.sum(self.W2**2)) if not self.use_adamw else 0.0

                # Grad izlaza
                p = _sigmoid(Z2).ravel()
                dZ2 = (p - yb).reshape(-1, 1) / len(Xb)
                dW2 = A1.T.dot(dZ2) + (self.l2*self.W2 if not self.use_adamw else 0.0)
                db2 = dZ2.sum(axis=0)

                # Grad skrivenog sloja
                dA1 = dZ2.dot(self.W2.T)
                if mask is not None:
                    dA1 = dA1 * mask / (1.0 - self.dropout)
                dZ1 = dA1 * (Z1 > 0).astype(float)
                dW1 = Xb.T.dot(dZ1) + (self.l2*self.W1 if not self.use_adamw else 0.0)
                db1 = dZ1.sum(axis=0)

                # Opcioni grad clipping
                if self.grad_clip is not None:
                    for g in (dW1, db1, dW2, db2):
                        gn = np.linalg.norm(g.ravel())
                        if gn > self.grad_clip:
                            g *= (self.grad_clip / (gn + 1e-12))

                # Jedan “time step” za Adam
                self._t += 1

                # Adam/AdamW koraci
                self._adam_update_inplace(self.W2, dW2, self.mW2, self.vW2, self.lr)
                self._adam_update_inplace(self.b2, db2, self.mb2, self.vb2, self.lr)
                self._adam_update_inplace(self.W1, dW1, self.mW1, self.vW1, self.lr)
                self._adam_update_inplace(self.b1, db1, self.mb1, self.vb1, self.lr)

                if self.use_adamw and self.weight_decay > 0.0:
                    # Decoupled weight decay (AdamW stil)
                    self.W2[:] -= self.lr * self.weight_decay * self.W2
                    self.W1[:] -= self.lr * self.weight_decay * self.W1

            # — kraj epohe: izračunaj val loss (ili train ako nema val)
            if X_val is not None:
                _, _, Z2v, _ = self._forward(X_val, train=False)
                val_loss = _bce_with_logits(Z2v, y_val, pos_weight=self.pos_weight) + (0.0 if self.use_adamw else 0.5*self.l2*(np.sum(self.W1**2)+np.sum(self.W2**2)))
            else:
                _, _, Z2t, _ = self._forward(X, train=False)
                val_loss = _bce_with_logits(Z2t, y, pos_weight=self.pos_weight) + (0.0 if self.use_adamw else 0.5*self.l2*(np.sum(self.W1**2)+np.sum(self.W2**2)))

            # Train loss za istoriju (čisto za graf)
            _, _, Z2tr, _ = self._forward(X, train=False)
            train_loss = _bce_with_logits(Z2tr, y, pos_weight=self.pos_weight) + (0.0 if self.use_adamw else 0.5*self.l2*(np.sum(self.W1**2)+np.sum(self.W2**2)))

            self.history_["train_loss"].append(float(train_loss))
            self.history_["val_loss"].append(float(val_loss))

            # Early stopping + LR scheduler (reduce on plateau)
            if val_loss < best_loss - 1e-7:
                best_loss = val_loss
                bad = 0
                bad_lr = 0
                BW1 = self.W1.copy()
                Bb1 = self.b1.copy()
                BW2 = self.W2.copy()
                Bb2 = self.b2.copy()
            else:
                bad += 1
                bad_lr += 1
                if self.reduce_on_plateau and bad_lr >= self.patience_lr:
                    self.lr = max(self.lr * self.lr_factor, 1e-6)  # smanji LR kada zapne
                    bad_lr = 0
                if bad >= self.patience:
                    # Vrati najbolje težine i prekini
                    self.W1, self.b1, self.W2, self.b2 = BW1, Bb1, BW2, Bb2
                    break

        # Posle treninga: ako imamo validaciju, nađi prag po Youden indeksu (TPr+TNr-1)
        if X_val is not None:
            p = self.predict_proba(X_val)[:, 1]
            thresholds = np.linspace(0.05, 0.95, 37)  # gruba mreža pragova
            bestJ, bestT = -1.0, 0.5
            yv = y_val.ravel().astype(int)
            for t in thresholds:
                yhat = (p >= t).astype(int)
                tp = np.sum((yhat == 1) & (yv == 1))
                tn = np.sum((yhat == 0) & (yv == 0))
                fp = np.sum((yhat == 1) & (yv == 0))
                fn = np.sum((yhat == 0) & (yv == 1))
                tpr = tp / (tp+fn+1e-12)
                tnr = tn / (tn+fp+1e-12)
                J = tpr + tnr - 1.0
                if J > bestJ:
                    bestJ, bestT = J, t
            self.best_threshold_ = float(bestT)
        else:
            self.best_threshold_ = 0.5  # default ako nema val skupa

        return self

    def predict_proba(self, X):
        # Vraća [P(y=0), P(y=1)] po uzorku (standardni format)
        X = np.asarray(X, dtype=float)
        _, _, Z2, _ = self._forward(X, train=False)
        p = _sigmoid(Z2).ravel()
        return np.vstack([1-p, p]).T

    def predict(self, X, threshold=None):
        # Ako prag nije dat, koristi onaj naučen na validaciji
        thr = self.best_threshold_ if threshold is None else float(threshold)
        return (self.predict_proba(X)[:, 1] >= thr).astype(int)
