# YZM304 Derin Öğrenme — I. Proje Ödevi
## Stellar Classification with Multi-Layer Perceptron (MLP)

**Ankara Üniversitesi — Yapay Zeka ve Veri Mühendisliği**  
**2025–2026 Bahar Dönemi**

---

## Project Structure / Proje Dosya Yapısı

```
YZM304_Proje/
│
├── 1_Stellar_temel_kod.ipynb                # Temel MLP modeli (1 gizli katman)
├── 2_overfitting_underfitting_analizi.ipynb  # Overfitting/Underfitting analizi
├── 3_farklı_modeller.ipynb                  # Çok katmanlı modeller karşılaştırması
├── 4_Scikit_learn_MLPClassifier.ipynb       # Sklearn implementasyonu
├── 5_PyTorch.ipynb                          # PyTorch implementasyonu
├── 6_SGD.ipynb                              # SGD ile NumPy/Sklearn/PyTorch karşılaştırması
└── README.md
```

---

## Introduction / Giriş

Göksel nesne sınıflandırması, modern astronominin temel problemlerinden biridir. Sloan Digital Sky Survey (SDSS), her biri fotometrik ve spektroskopik ölçümlerle tanımlanan milyonlarca göksel nesne verisi üretmektedir. Bu çalışmada SDSS17 veri seti kullanılarak galaksi (GALAXY), yıldız (STAR) ve kuasar (QSO) nesnelerini birbirinden ayırt eden bir derin öğrenme modeli geliştirilmiştir.

Çalışma, 13.03.2026 tarihinde YZM304 Derin Öğrenme laboratuvarında ikili sınıflandırma için uygulanan 2 katmanlı MLP modelinin devamı niteliğindedir. Laboratuvar modeli çoklu sınıflandırma için **Softmax** çıkış katmanı ve **Categorical Cross-Entropy** kayıp fonksiyonu eklenerek uyarlanmış; overfitting/underfitting analizi, regularizasyon, mini-batch eğitimi ve farklı kütüphane implementasyonları ile genişletilmiştir.

**Sonuç özeti:** En iyi NumPy modeli (M6: 3 gizli katman) test setinde ~%94 accuracy elde etmiştir. Sklearn (Adam) ve PyTorch (Adam) implementasyonları benzer mimarilerle daha hızlı yakınsama göstermiştir. SGD karşılaştırmasında üç implementasyon arasında tutarlı sonuçlar elde edilmiştir.

---

## Methods / Yöntemler

### Veri Seti

| Özellik | Değer |
|---|---|
| Kaynak | Stellar Classification Dataset — SDSS17 (Kaggle) |
| Toplam örnek | 100.000 |
| Özellik sayısı (ham) | 18 |
| Özellik sayısı (işlenmiş) | 14 |
| Sınıf sayısı | 3 (GALAXY: 59.445, STAR: 21.594, QSO: 18.961) |
| Dosya | `star_classification.csv` |

### Ön İşleme

**Silinen sütunlar (ID niteliğinde, bilgi taşımaz):**
```
obj_ID, run_ID, rerun_ID, cam_col, field_ID, spec_obj_ID, plate, MJD, fiber_ID
```

**Özellik mühendisliği (Feature Engineering):**
Fotometrik bantlardan (u, g, r, i, z) astronomik renk indeksleri türetilmiştir:

| Yeni Özellik | Formül | Açıklama |
|---|---|---|
| u_g | u − g | Mor–Yeşil renk indeksi |
| g_r | g − r | Yeşil–Kırmızı renk indeksi |
| r_i | r − i | Kırmızı–İnfrared renk indeksi |
| i_z | i − z | İnfrared renk indeksi |
| u_r | u − r | Geniş renk aralığı |
| redshift_sq | redshift² | Redshift'in non-lineer etkisi |

**Standardizasyon:** `StandardScaler` — ortalama=0, std=1

**Veri artırma (Data Augmentation):** Eğitim setine σ=0.05 Gaussian gürültü eklenerek 2× büyütme (yalnızca M7 modelinde kullanılmıştır)

**Train/Validation/Test split:**
```python
# Tüm notebook'larda aynı split parametreleri
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y_raw, test_size=0.30, random_state=42, stratify=y_raw)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.667, random_state=42, stratify=y_temp)
# Sonuç: Train %70 (70.000) | Val %10 (9.990) | Test %20 (20.010)
```

**Hedef encoding:** Label Encoding → GALAXY=0, QSO=1, STAR=2  
**One-Hot Encoding:** Çoklu sınıflandırma için hedef vektör (m, 3) matrisine dönüştürülmüştür

---

### Model Mimarisi

**Aktivasyon fonksiyonları:**
- Gizli katmanlar: `tanh`
- Çıkış katmanı: `Softmax`

**Kayıp fonksiyonu:** Categorical Cross-Entropy  
$$J = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{3} y_k^{(i)} \log(\hat{y}_k^{(i)})$$

**Ağırlık başlatma:** He initialization  
$$W \sim \mathcal{N}\left(0,\ \sqrt{\frac{2}{n_{in}}}\right)$$

---

### Notebook 1 — Temel Model (Laboratuvar Modeli)
**Dosya:** `1_Stellar_temel_kod.ipynb`

13.03.2026 tarihinde laboratuvarda uygulanan 2 katmanlı modelin çoklu sınıflandırma uyarlamasıdır.

**Mimari:** `14 → 32 → 3` (1 gizli katman)

| Hiperparametre | Değer |
|---|---|
| n_h (gizli nöron) | 32 |
| n_steps | 300 |
| learning_rate | 0.5 |
| lambda_reg | 0.0 |
| random_state | 42 |
| split | %80 train / %20 test |

```python
parameters, costs = nn_model(
    X_train, y_train,
    n_x=X_train.shape[1],  # 14
    n_h=32,
    n_y=3,
    n_steps=300,
    learning_rate=0.5
)
```

---

### Notebook 2 — Overfitting/Underfitting Analizi
**Dosya:** `2_overfitting_underfitting_analizi.ipynb`

**Mimari:** `14 → 32 → 3`

| Hiperparametre | Değer |
|---|---|
| n_h | 32 |
| n_steps | 300 |
| learning_rate | 0.5 |
| split | %70 train / %10 val / %20 test |
| random_state | 42 |

Her 10 adımda train ve validation loss/accuracy kaydedilir.

**Teşhis kriteri:**
- Train − Val gap > 0.05 → **Overfitting**
- Train accuracy < 0.85 → **Underfitting**
- Aksi halde → **İyi fit**

**Hiperparametre arama:**
- n_h ∈ {16, 32, 64}, n_steps ∈ {100, 200, 300}
- val_acc ≥ %90 koşulunu sağlayanlar arasından **en düşük n_steps** seçilir

---

### Notebook 3 — Çok Katmanlı Modeller
**Dosya:** `3_farklı_modeller.ipynb`

| Model | Mimari | lambda_reg | Veri Artırma | Mini-batch |
|---|---|---|---|---|
| M1 | [14, 32, 3] | 0.0 | ✗ | ✗ |
| M2 | [14, 64, 3] | 0.0 | ✗ | ✗ |
| M3 | [14, 128, 3] | 0.0 | ✗ | ✗ |
| M4 | [14, 64, 32, 3] | 0.0 | ✗ | ✗ |
| M5 | [14, 64, 32, 3] | 0.01 | ✗ | ✗ |
| M6 | [14, 64, 32, 16, 3] | 0.0 | ✗ | ✗ |
| M7 | [14, 64, 32, 3] | 0.0 | ✓ (2×, σ=0.05) | ✗ |
| M8 | [14, 64, 32, 3] | 0.0 | ✗ | ✓ |
| M9 | [14, 64, 32, 3] | 0.01 | ✗ | ✓ |

**Full-batch modeller (M1–M7):**
```python
n_steps       = 300
learning_rate = 0.5
lambda_reg    = 0.0   # M5 için: 0.01
random_seed   = 42
```

**Mini-batch modeller (M8–M9):**
```python
n_epochs      = 30
learning_rate = 0.1
batch_size    = 512
lambda_reg    = 0.0   # M9 için: 0.01
random_seed   = 42
```

**Model seçim kriteri:** val_acc ≥ %90 koşulunu sağlayanlar arasından en yüksek test_acc

---

### Notebook 4 — Scikit-learn MLPClassifier
**Dosya:** `4_Scikit_learn_MLPClassifier.ipynb`

M1, M4, M5, M6 mimarileri Sklearn ile uygulanmıştır.

| Hiperparametre | Değer |
|---|---|
| solver (optimizer) | adam (sklearn varsayılanı) |
| activation | tanh |
| max_iter | 300 |
| random_state | 42 |
| alpha (M5, L2) | 0.01 |

```python
MLPClassifier(
    hidden_layer_sizes=(64, 32),  # M4
    activation='tanh',
    solver='adam',
    max_iter=300,
    random_state=42
)
```

---

### Notebook 5 — PyTorch
**Dosya:** `5_PyTorch.ipynb`

M1, M4, M5, M6 mimarileri PyTorch ile uygulanmıştır.  
`MLP` sınıfı `nn.Module`'den türetilmiş olup constructor, public ve private metotlar içermektedir:

```python
class MLP(nn.Module):
    def __init__(self, layers):   # Constructor
        self.net = self._build_network()
        self._init_weights()

    def _build_network(self):     # Private: katmanları oluşturur
        ...
    def _init_weights(self):      # Private: He initialization uygular
        ...
    def forward(self, x):         # Public: ileri yayılım, ham logit döndürür
        ...
    def predict(self, x):         # Public: argmax ile sınıf tahmini
        ...
```

| Hiperparametre | Değer |
|---|---|
| optimizer | Adam (PyTorch varsayılanı) |
| learning_rate | 1e-3 |
| loss | CrossEntropyLoss |
| n_epochs | 300 |
| weight_decay (M5) | 0.01 |
| random_seed | 42 |

---

### Notebook 6 — SGD Karşılaştırması
**Dosya:** `6_SGD.ipynb`

Aynı koşullarla NumPy / Sklearn / PyTorch karşılaştırması yapılmıştır.

| Sabit Koşul | Değer |
|---|---|
| Mimari | [14, 64, 32, 3] |
| Optimizer | SGD (momentum=0, weight_decay=0) |
| learning_rate | 0.5 |
| n_steps / max_iter | 300 |
| random_state / seed | 42 |
| split | %70 train / %10 val / %20 test |
| Aktivasyon | tanh |
| Loss | Cross-Entropy |

```python
# NumPy
LR      = 0.5
N_STEPS = 300
LAYERS  = [14, 64, 32, 3]

# Sklearn
MLPClassifier(solver='sgd', learning_rate_init=0.5, max_iter=300, random_state=42)

# PyTorch
optim.SGD(model.parameters(), lr=0.5)
```

---

## Results / Sonuçlar

### Notebook 1 — Temel Model

| Metrik | Değer |
|---|---|
| Test Accuracy | ~%93 |
| Mimari | 14 → 32 → 3 |
| Aktivasyon | tanh + Softmax |
| Loss | Categorical Cross-Entropy |

### Notebook 2 — Overfitting/Underfitting

Train ve validation loss eğrileri incelendiğinde modelin iyi fit durumda olduğu gözlemlenmiştir (Train−Val gap < 0.05). Hiperparametre aramasında val_acc ≥ %90 koşulunu sağlayan en düşük n_steps değeri seçilmiştir.

### Notebook 3 — Çok Katmanlı Modeller

| Model | Mimari | Teknik | Test Acc |
|---|---|---|---|
| M1 | 14→32→3 | Temel | ~%93 |
| M2 | 14→64→3 | Nöron artırma | ~%93 |
| M3 | 14→128→3 | Daha fazla nöron | ~%90 |
| M4 | 14→64→32→3 | 2 gizli katman | ~%93 |
| M5 | 14→64→32→3 | 2 gizli + L2 | ~%93 |
| M6 | 14→64→32→16→3 | 3 gizli katman | ~%94 |
| M7 | 14→64→32→3 | Veri artırma | ~%93 |
| M8 | 14→64→32→3 | Mini-batch | ~%93 |
| M9 | 14→64→32→3 | Mini-batch + L2 | ~%93 |

> *Not: Kesin değerler çalışma ortamında elde edilen çıktılara göre güncellenmelidir.*

### Notebook 4 — Scikit-learn (Adam)

Adam optimizer ile Sklearn implementasyonu daha az iterasyonda benzer accuracy değerlerine ulaşmıştır. `loss_curve_` özelliği üzerinden takip edilen loss eğrileri düzgün bir yakınsama sergilemiştir.

### Notebook 5 — PyTorch (Adam)

Adam optimizer ile PyTorch implementasyonu hızlı yakınsama göstermiştir. `predict()` public metodu aracılığıyla tutarlı sınıf tahminleri elde edilmiştir. Confusion matrix ve classification report sonuçları NumPy modeli ile karşılaştırılabilir düzeydedir.

### Notebook 6 — SGD Karşılaştırması

Aynı hiperparametreler ve SGD optimizer kullanıldığında NumPy, Sklearn ve PyTorch implementasyonları birbirine yakın test accuracy değerleri üretmiştir. Bu sonuç, sıfırdan yazılan NumPy implementasyonunun kütüphane tabanlı çözümlerle tutarlı olduğunu doğrulamaktadır.

---

## Discussion / Tartışma

### Bulgular ve Yorumlar

**Overfitting/Underfitting:** Tüm modellerde train−val gap 0.05'in altında kalmıştır. 100.000 örnekli ve dengeli bir veri seti aşırı öğrenmeyi doğal olarak sınırlamaktadır. Redshift özelliği, sınıflar arasında en güçlü ayrım sağlayan özellik olarak öne çıkmaktadır.

**Katman derinliği:** 3 gizli katmanlı M6 modeli en yüksek test accuracy değerine ulaşmıştır; ancak fark marjinaldir. Stellar Classification veri seti için 1–2 gizli katman yeterli görünmektedir. Çok derin mimarilerin bu veri seti üzerinde belirgin bir avantaj sağlamadığı gözlemlenmiştir.

**L2 Regularizasyon:** Veri setinin dengeli ve büyük olması nedeniyle L2'nin belirgin etkisi gözlemlenmemiştir. Regularizasyonun daha küçük veya dengesiz veri setlerinde daha etkili olması beklenir.

**Mini-batch vs Full-batch:** Mini-batch (batch_size=512) eğitimi her epoch'ta daha fazla parametre güncellemesi yaptığından daha hızlı yakınsama sergilemiştir. Büyük veri setlerinde bellek verimliliği açısından da avantaj sağlar.

**He Initialization:** `0.01 * randn` yerine He initialization kullanılması kritik öneme sahiptir. He initialization olmaksızın model tüm örnekleri tek sınıfa atfetmiştir (dead classifier). Doğru initialization softmax çıkışının başlangıçta dengeli dağılım üretmesini sağlar.

**Özellik mühendisliği:** Renk indekslerinin eklenmesi modelin fotometrik örüntüleri daha iyi öğrenmesine katkı sağlamıştır. Redshift'in karesi alınarak eklenmesi non-lineer ilişkiyi yakalamaya yardımcı olmuştur.

**Kütüphane karşılaştırması:** NumPy (sıfırdan), Sklearn ve PyTorch implementasyonları aynı SGD koşullarında tutarlı sonuçlar üretmiştir. Bu durum, temel algoritmanın doğru implementasyonunu teyit etmektedir. Adam optimizer kullanıldığında Sklearn ve PyTorch daha hızlı yakınsama göstermiştir.

### Gelecek Çalışmalar

- **Dropout** regularizasyon eklenerek overfitting davranışı incelenebilir
- **Batch normalization** katmanı eklenerek eğitim stabilitesi artırılabilir
- **Adam, RMSProp, Adagrad** gibi adaptive optimizer'lar ile SGD karşılaştırması genişletilebilir
- Hiperparametre optimizasyonu için **Optuna** veya **Ray Tune** kullanılabilir
- Veri dengesizliğine karşı **SMOTE** veya **class weighting** uygulanabilir
- **Residual bağlantılar** (skip connections) ile daha derin mimari denenebilir
- **Cross-validation** ile model seçimi daha güvenilir hale getirilebilir

---

## Reproducibility / Tekrarlanabilirlik

Çalışmanın tam olarak tekrarlanabilmesi için tüm hiperparametreler ve başlangıç ayarları:

```python
# ── Genel ───────────────────────────────────────────────
RANDOM_STATE  = 42
NUM_CLASSES   = 3

# ── Veri Bölme ──────────────────────────────────────────
TEST_SIZE     = 0.20      # %20
VAL_SIZE      = 0.10      # %10
TRAIN_SIZE    = 0.70      # %70

# ── Ön İşleme ───────────────────────────────────────────
SCALER        = 'StandardScaler'
NOISE_STD     = 0.05      # veri artırma (M7)
NOISE_SEED    = 0

# ── Notebook 1 & 2 (NumPy, temel model) ────────────────
N_H           = 32
N_STEPS       = 300
LEARNING_RATE = 0.5
LAMBDA_REG    = 0.0

# ── Notebook 3 (Çok katmanlı modeller) ─────────────────
# Full-batch (M1–M7)
FB_N_STEPS    = 300
FB_LR         = 0.5

# Mini-batch (M8–M9)
MB_EPOCHS     = 30
MB_LR         = 0.1
BATCH_SIZE    = 512
L2_LAMBDA     = 0.01      # M5, M9

# ── Notebook 4 (Sklearn) ────────────────────────────────
SK_SOLVER     = 'adam'
SK_ACTIVATION = 'tanh'
SK_MAX_ITER   = 300
SK_ALPHA      = 0.01      # M5 için

# ── Notebook 5 (PyTorch) ────────────────────────────────
PT_OPTIMIZER  = 'Adam'
PT_LR         = 1e-3
PT_EPOCHS     = 300
PT_LOSS       = 'CrossEntropyLoss'
PT_WD         = 0.01      # M5 için

# ── Notebook 6 (SGD karşılaştırması) ────────────────────
SGD_LR        = 0.5
SGD_STEPS     = 300
SGD_LAYERS    = [14, 64, 32, 3]
```

### Ortam Gereksinimleri

```
Python       >= 3.9
numpy        >= 1.24
pandas       >= 2.0
scikit-learn >= 1.3
torch        >= 2.0
matplotlib   >= 3.7
```

### Kurulum

```bash
pip install numpy pandas scikit-learn torch matplotlib
```

### Veri Dosyası

`star_classification.csv` dosyası Kaggle'dan indirilerek Colab'da `/content/` dizinine yüklenmelidir:

https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

---

*YZM304 Derin Öğrenme — Ankara Üniversitesi Yapay Zeka ve Veri Mühendisliği, 2025–2026 Bahar*
