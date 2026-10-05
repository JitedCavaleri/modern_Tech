import torch
import torch.nn as nn
import numpy as np

# ---------- 1. Готовим данные ----------
def make_dataset(n_samples=5000, max_num=1000):
    X = np.random.randint(0, max_num, size=(n_samples, 1)).astype(np.float32)
    # Нормализуем числа в диапазон [0, 1]
    X_norm = X / max_num
    # Метка: 1 если чётное, 0 если нечётное
    y = (X % 2 == 0).astype(np.float32).reshape(-1, 1)
    return torch.tensor(X_norm), torch.tensor(y), torch.tensor(X)

X_train, y_train, raw_train = make_dataset(8000, max_num=1000)
X_test,  y_test,  raw_test  = make_dataset(2000, max_num=1000)

# ---------- 2. Определяем модель ----------
class ParityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()   # выход: вероятность чётности
        )

    def forward(self, x):
        return self.net(x)

model = ParityNet()

# ---------- 3. Обучение ----------
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 300
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        with torch.no_grad():
            acc = ((pred > 0.5).float() == y_train).float().mean().item()
        print(f"Epoch {epoch+1:3d} | loss = {loss.item():.4f} | acc = {acc:.4f}")

# ---------- 4. Проверка ----------
model.eval()
with torch.no_grad():
    pred_test = model(X_test)
    acc_test = ((pred_test > 0.5).float() == y_test).float().mean().item()
print(f"\nТочность на тесте: {acc_test:.4f}")

# ---------- 5. Инференс функции ----------
def predict_parity(number, max_num=1000):
    """Возвращает True если число чётное"""
    x = torch.tensor([[number / max_num]], dtype=torch.float32)
    with torch.no_grad():
        p = model(x).item()
    return p > 0.5, p

# Примеры
for n in [2, 3, 7, 10, 15, 42, 99, 128, 777]:
    is_even, prob = predict_parity(n)
    print(f"{n:4d} -> {'ЧЁТНОЕ' if is_even else 'НЕЧЁТНОЕ'} (p={prob:.3f})")