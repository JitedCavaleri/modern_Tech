//ну тут поидее должен быть код нейроосети

import torch

import torch.nn as nn

import numpy as np



\# ---------- 1. Готовим данные ----------

def make\_dataset(n\_samples=5000, max\_num=1000):

&#x20;   X = np.random.randint(0, max\_num, size=(n\_samples, 1)).astype(np.float32)

&#x20;   # Нормализуем числа в диапазон \[0, 1]

&#x20;   X\_norm = X / max\_num

&#x20;   # Метка: 1 если чётное, 0 если нечётное

&#x20;   y = (X % 2 == 0).astype(np.float32).reshape(-1, 1)

&#x20;   return torch.tensor(X\_norm), torch.tensor(y), torch.tensor(X)



X\_train, y\_train, raw\_train = make\_dataset(8000, max\_num=1000)

X\_test,  y\_test,  raw\_test  = make\_dataset(2000, max\_num=1000)



\# ---------- 2. Определяем модель ----------

class ParityNet(nn.Module):

&#x20;   def \_\_init\_\_(self):

&#x20;       super().\_\_init\_\_()

&#x20;       self.net = nn.Sequential(

&#x20;           nn.Linear(1, 64),

&#x20;           nn.ReLU(),

&#x20;           nn.Linear(64, 64),

&#x20;           nn.ReLU(),

&#x20;           nn.Linear(64, 1),

&#x20;           nn.Sigmoid()   # выход: вероятность чётности

&#x20;       )



&#x20;   def forward(self, x):

&#x20;       return self.net(x)



model = ParityNet()



\# ---------- 3. Обучение ----------

criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



epochs = 300

for epoch in range(epochs):

&#x20;   model.train()

&#x20;   optimizer.zero\_grad()

&#x20;   pred = model(X\_train)

&#x20;   loss = criterion(pred, y\_train)

&#x20;   loss.backward()

&#x20;   optimizer.step()



&#x20;   if (epoch + 1) % 50 == 0:

&#x20;       with torch.no\_grad():

&#x20;           acc = ((pred > 0.5).float() == y\_train).float().mean().item()

&#x20;       print(f"Epoch {epoch+1:3d} | loss = {loss.item():.4f} | acc = {acc:.4f}")



\# ---------- 4. Проверка ----------

model.eval()

with torch.no\_grad():

&#x20;   pred\_test = model(X\_test)

&#x20;   acc\_test = ((pred\_test > 0.5).float() == y\_test).float().mean().item()

print(f"\\nТочность на тесте: {acc\_test:.4f}")



\# ---------- 5. Инференс функции ----------

def predict\_parity(number, max\_num=1000):

&#x20;   """Возвращает True если число чётное"""

&#x20;   x = torch.tensor(\[\[number / max\_num]], dtype=torch.float32)

&#x20;   with torch.no\_grad():

&#x20;       p = model(x).item()

&#x20;   return p > 0.5, p



\# Примеры

for n in \[2, 3, 7, 10, 15, 42, 99, 128, 777]:

&#x20;   is\_even, prob = predict\_parity(n)

&#x20;   print(f"{n:4d} -> {'ЧЁТНОЕ' if is\_even else 'НЕЧЁТНОЕ'} (p={prob:.3f})")

