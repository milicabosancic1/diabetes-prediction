import numpy as np


def _relu(x):
    """
    ReLU: sve negativne vrednosti “seku” se na 0, pozitivne se ne menjaju.
    """
    return np.maximum(0.0, x)


def _sigmoid(x):
    """
    Sigmoid sa klipovanjem ulaza radi numeričke stabilnosti:
      p = 1 / (1 + exp(-x))
    - np.clip(x, -40, 40) sprečava overflow/underflow za velike |x|.
    - Koristi se samo kada nam treba verovatnoća iz logita
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _bce_with_logits(z, y, pos_weight=None, sample_weight=None):
    """
    Stabilna binarna unakrsna entropija (BCE) rađena DIREKTNO nad logitima z (tj. pre sigmoid-a).

    Loss po uzorku (stabilna forma):
        loss = max(0, z) - z*y + log(1 + exp(-|z|))

    Parametri:
      z : logits (R^{n} ili (n,1))
      y : ciljne oznake u {0,1}
      pos_weight : ako je zadat, multiplicira doprinos pozitivnih uzoraka
      sample_weight : težine po uzorku

    Vraća:
      prosek gubitka (float), ili ponderisani prosek ako je sample_weight zadan.
    """

    # Poravnaj vektore
    z = z.ravel()
    y = y.ravel()

    # Stabilna BCE-with-logits formula
    loss = (np.maximum(0.0, z) - z*y + np.log1p(np.exp(-np.abs(z))))

    if pos_weight is not None:
        # Pojačaj pozitivne uzorke ako je klasa 1 retka: w=pos_weight za y=1, inače 1
        w = np.where(y == 1.0, float(pos_weight), 1.0)
        loss = loss * w

    if sample_weight is not None:
        # Ponderisani prosek (čuvamo stabilnost kada je suma težina mala)
        loss = loss * sample_weight
        return loss.sum() / (np.sum(sample_weight) + 1e-12)

    return loss.mean()


class MLP:
    """
      MLP za binarnu klasifikaciju sa 1 skrivenim slojem:
      Ulaz -> (FC -> ReLU -> Dropout) -> FC -> logit

    Trening:
      - Mini-batch Adam ili AdamW
      - L2 (ako nije AdamW) ili decoupled weight decay (ako je AdamW)
      - Early stopping po val loss
      - ReduceLROnPlateau scheduler (smanjuje LR kada “zaglavi”)
      - Opcioni grad clipping
      - Opcioni pos_weight u BCE (balansiranje klasa)
      - Posle treninga, određivanje najboljeg praga predikcije po Youden indeksu
    """

    def __init__(self, hidden=64, lr=0.01, l2=1e-4, max_epochs=500, patience=50,
                 random_state=42, batch_size=64, dropout=0.0, grad_clip=None,
                 beta1=0.9, beta2=0.999, eps=1e-8, use_adamw=False, weight_decay=1e-4,
                 reduce_on_plateau=True, patience_lr=10, lr_factor=0.5,
                 pos_weight=None):

        # Hiperparametri modela i optimizacije (vidi docstring iznad)
        self.h = int(hidden)               # broj neurona u skrivenom sloju
        self.lr0 = float(lr)               # pamtimo početni LR (za restart pri _init)
        self.lr = float(lr)                # trenutni LR (može se smanjivati scheduler-om)
        self.l2 = float(l2)                # L2 koef. (koristi se samo kada NE koristimo AdamW)
        self.max_epochs = int(max_epochs)  # maksimalan broj epoha
        self.patience = int(patience)      # strpljenje za early stopping
        self.bs = int(batch_size)          # veličina mini-batch-a
        self.dropout = float(dropout)      # verovatnoća gašenja neurona (0.0 = bez dropout-a)
        self.grad_clip = None if grad_clip is None else float(grad_clip)  # L2 kliping gradijenta
        self.beta1, self.beta2, self.eps = float(beta1), float(beta2), float(eps)  # Adam hiperparametri
        self.use_adamw = bool(use_adamw)   # True => koristi decoupled weight decay
        self.weight_decay = float(weight_decay)  # koef. za AdamW decoupled decay
        self.reduce_on_plateau = bool(reduce_on_plateau)  # LR scheduler aktivan?
        self.patience_lr = int(patience_lr)  # posle ovoliko epoha bez napretka smanji LR
        self.lr_factor = float(lr_factor)    # faktor smanjenja LR (npr. 0.5)
        self.pos_weight = pos_weight         # balans pozitivne klase u BCE
        self.rng = np.random.default_rng(random_state)  # generator za mešanje, dropout, inicijalizaciju

        # Težine, bias-i i Adam momenti (postaviće se u _init)
        self.W1 = self.b1 = self.W2 = self.b2 = None
        self.mW1 = self.vW1 = self.mb1 = self.vb1 = None
        self.mW2 = self.vW2 = self.mb2 = self.vb2 = None
        self._t = 0  # globalni “time step” za Adam bias-correction

        # Istorija treninga (za grafove) i naučen klasifikacioni prag
        self.history_ = {"train_loss": [], "val_loss": []}
        self.best_threshold_ = 0.5

    def _init(self, d):
        """
        Inicijalizacija parametara i optimizatora.
        - He inicijalizacija za ReLU slojeve: std ~ sqrt(2/fan_in)
        - Nulti bias-i
        - Nulti Adam momenti
        - Resetujemo brojač koraka i LR
        """
        # FC1: d -> h
        self.W1 = self.rng.normal(0, np.sqrt(2.0/d), size=(d, self.h))
        self.b1 = np.zeros(self.h)
        # FC2: h -> 1 (logit)
        self.W2 = self.rng.normal(0, np.sqrt(2.0/self.h), size=(self.h, 1))
        self.b2 = np.zeros(1)

        # Adam momenti (m i v) za sve parametre — inicijalno nule
        zW1 = np.zeros_like(self.W1)
        zb1 = np.zeros_like(self.b1)
        zW2 = np.zeros_like(self.W2)
        zb2 = np.zeros_like(self.b2)
        self.mW1 = zW1.copy(); self.vW1 = zW1.copy()
        self.mb1 = zb1.copy(); self.vb1 = zb1.copy()
        self.mW2 = zW2.copy(); self.vW2 = zW2.copy()
        self.mb2 = zb2.copy(); self.vb2 = zb2.copy()

        self._t = 0      # reset Adam koraka
        self.lr = self.lr0  # reset LR-a na početnu vrednost

    def _forward(self, X, train=False):
        """
        Jedan prolaz unapred: FC -> ReLU -> (opciono Dropout) -> FC -> logit
        - train=True uključuje inverted dropout (skaliran na train-u).
        - Vraćamo i masku kako bismo je koristili u backprop-u.
        """
        Z1 = X.dot(self.W1) + self.b1   # linearni deo skrivenog sloja
        A1 = _relu(Z1)                  # ReLU aktivacija

        mask = None
        if train and self.dropout > 0.0:
            # Inverted dropout: na treniranju skaliraš aktivacije tako da se očekivanje zadrži,
            # a na evaluaciji dropout se ne primenjuje (nema dodatnog skaliranja).
            mask = (self.rng.random(A1.shape) >= self.dropout).astype(A1.dtype)
            A1 = A1 * mask / (1.0 - self.dropout)

        Z2 = A1.dot(self.W2) + self.b2  # izlazni logit (pre sigmoid-a)
        return Z1, A1, Z2, mask

    def _adam_update_inplace(self, W, gW, mW, vW, lr):
        """
        Jedan Adam korak (u mestu):
          m_t = beta1 * m_{t-1} + (1-beta1) * g
          v_t = beta2 * v_{t-1} + (1-beta2) * g^2
          m_hat = m_t / (1 - beta1^t)       (bias correction)
          v_hat = v_t / (1 - beta2^t)
          W -= lr * m_hat / (sqrt(v_hat) + eps)
        """
        mW[:] = self.beta1*mW + (1-self.beta1)*gW
        vW[:] = self.beta2*vW + (1-self.beta2)*(gW*gW)
        m_hat = mW / (1 - self.beta1**self._t)
        v_hat = vW / (1 - self.beta2**self._t)
        W[:] -= lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Trening petlja:
          - mini-batch optimizacija (Adam/AdamW)
          - BCE-with-logits gubitak (uz pos_weight i/ili sample weighting ako treba)
          - L2 regularizacija u gubitku (samo kada nije AdamW)
          - decoupled weight decay (samo kada jeste AdamW)
          - early stopping + ReduceLROnPlateau
          - određivanje praga po Youden indeksu (ako postoji validacioni skup)
        """
        # Osiguraj tipove/oblike
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, d = X.shape

        # Inicijalizuj parametre i optimizator
        self._init(d)

        # “Najbolje do sada” stanje za early stopping
        best_loss = np.inf
        bad = 0          # broj uzastopnih epoha bez poboljšanja (za early stopping)
        bad_lr = 0       # broj uzastopnih epoha bez poboljšanja (za LR scheduler)

        # kopije za najbolje težine
        BW1 = self.W1.copy(); Bb1 = self.b1.copy()
        BW2 = self.W2.copy(); Bb2 = self.b2.copy()

        idx_all = np.arange(n)  # indeksi za mešanje batch-eva
        for epoch in range(self.max_epochs):
            self.rng.shuffle(idx_all)

            # --- prolaz kroz sve mini-batcheve
            for start in range(0, n, self.bs):
                idx = idx_all[start:start+self.bs]
                Xb, yb = X[idx], y[idx]

                # Forward (sa dropout-om na train-u)
                Z1, A1, Z2, mask = self._forward(Xb, train=True)

                # BCE-with-logits gubitak; L2 dodajemo u gradijent ako nije AdamW
                base_loss = _bce_with_logits(Z2, yb, pos_weight=self.pos_weight)
                reg = 0.5*self.l2*(np.sum(self.W1**2)+np.sum(self.W2**2)) if not self.use_adamw else 0.0

                # Gradijent izlaznog sloja:
                # dL/dZ2 = (sigmoid(Z2) - y) / batch_size
                p = _sigmoid(Z2).ravel()
                dZ2 = (p - yb).reshape(-1, 1) / len(Xb)
                dW2 = A1.T.dot(dZ2) + (self.l2*self.W2 if not self.use_adamw else 0.0)  # L2 samo ako nije AdamW
                db2 = dZ2.sum(axis=0)

                # Gradijent skrivenog sloja:
                # dA1 = dZ2 * W2^T; zatim ReLU derivacija: 1 za Z1>0, inače 0
                dA1 = dZ2.dot(self.W2.T)
                if mask is not None:
                    # Vraćamo isti dropout faktor i ovde (inverted dropout)
                    dA1 = dA1 * mask / (1.0 - self.dropout)
                dZ1 = dA1 * (Z1 > 0).astype(float)
                dW1 = Xb.T.dot(dZ1) + (self.l2*self.W1 if not self.use_adamw else 0.0)
                db1 = dZ1.sum(axis=0)

                # Jedan Adam “time step” (potreban za bias-correction)
                self._t += 1

                # Adam/AdamW ažuriranja parametara
                self._adam_update_inplace(self.W2, dW2, self.mW2, self.vW2, self.lr)
                self._adam_update_inplace(self.b2, db2, self.mb2, self.vb2, self.lr)
                self._adam_update_inplace(self.W1, dW1, self.mW1, self.vW1, self.lr)
                self._adam_update_inplace(self.b1, db1, self.mb1, self.vb1, self.lr)

                if self.use_adamw and self.weight_decay > 0.0:
                    # Decoupled weight decay (AdamW): direktno “skuplja” težine,
                    # bez ubacivanja u gubitak (za razliku od klasičnog L2).
                    self.W2[:] -= self.lr * self.weight_decay * self.W2
                    self.W1[:] -= self.lr * self.weight_decay * self.W1

            # --- kraj epohe: evaluacija gubitka na validaciji
            if X_val is not None:
                _, _, Z2v, _ = self._forward(X_val, train=False)
                val_loss = _bce_with_logits(Z2v, y_val, pos_weight=self.pos_weight) \
                           + (0.0 if self.use_adamw else 0.5*self.l2*(np.sum(self.W1**2)+np.sum(self.W2**2)))
            else:
                _, _, Z2t, _ = self._forward(X, train=False)
                val_loss = _bce_with_logits(Z2t, y, pos_weight=self.pos_weight) \
                           + (0.0 if self.use_adamw else 0.5*self.l2*(np.sum(self.W1**2)+np.sum(self.W2**2)))

            # Uvek računamo i train_loss (za lep graf istorije)
            _, _, Z2tr, _ = self._forward(X, train=False)
            train_loss = _bce_with_logits(Z2tr, y, pos_weight=self.pos_weight) \
                         + (0.0 if self.use_adamw else 0.5*self.l2*(np.sum(self.W1**2)+np.sum(self.W2**2)))

            # Sačuvaj u istoriju (float radi JSON-serializacije)
            self.history_["train_loss"].append(float(train_loss))
            self.history_["val_loss"].append(float(val_loss))

            # Early stopping + ReduceLROnPlateau
            if val_loss < best_loss - 1e-7:
                # Poboljšanje: reset brojača i snimi najbolje težine
                best_loss = val_loss
                bad = 0
                bad_lr = 0
                BW1 = self.W1.copy(); Bb1 = self.b1.copy()
                BW2 = self.W2.copy(); Bb2 = self.b2.copy()
            else:
                # Nema poboljšanja
                bad += 1
                bad_lr += 1

                # LR scheduler: ako je “zapelo” duže od patience_lr epoha, smanji LR
                if self.reduce_on_plateau and bad_lr >= self.patience_lr:
                    self.lr = max(self.lr * self.lr_factor, 1e-6)  # čuvaj donju granicu LR
                    bad_lr = 0

                # Early stopping: dosta stagnacije => vrati najbolje viđene težine i prekini
                if bad >= self.patience:
                    self.W1, self.b1, self.W2, self.b2 = BW1, Bb1, BW2, Bb2
                    break

        # Nakon treninga: podešavanje klasifikacionog praga po Youden indeksu ---
        if X_val is not None:
            # Računamo verovatnoće na validacionom skupu
            p = self.predict_proba(X_val)[:, 1]

            # mreža pragova
            thresholds = np.linspace(0.05, 0.95, 37)
            bestJ, bestT = -1.0, 0.5
            yv = y_val.ravel().astype(int)

            for t in thresholds:
                yhat = (p >= t).astype(int)
                tp = np.sum((yhat == 1) & (yv == 1))
                tn = np.sum((yhat == 0) & (yv == 0))
                fp = np.sum((yhat == 1) & (yv == 0))
                fn = np.sum((yhat == 0) & (yv == 1))

                # TPR (sensitivity/recall) i TNR (specificity)
                tpr = tp / (tp + fn + 1e-12)
                tnr = tn / (tn + fp + 1e-12)

                # Youden J = TPR + TNR - 1 (maksimizuje balans senzitivnosti i specifičnosti)
                J = tpr + tnr - 1.0
                if J > bestJ:
                    bestJ, bestT = J, t

            self.best_threshold_ = float(bestT)
        else:
            # Ako nema validacije, koristim default prag 0.5
            self.best_threshold_ = 0.5

        return self

    def predict_proba(self, X):
        """
        Računa verovatnoće klasa:
          vraća matricu oblika (n, 2) = [P(y=0), P(y=1)].
        - Standardni format kompatibilan sa scikit-learn interfejsima.
        """
        X = np.asarray(X, dtype=float)
        _, _, Z2, _ = self._forward(X, train=False)
        p = _sigmoid(Z2).ravel()  # P(y=1)
        return np.vstack([1-p, p]).T
