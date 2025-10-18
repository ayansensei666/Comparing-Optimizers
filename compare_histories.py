import numpy as np
import matplotlib.pyplot as plt

optimizers = ["sgd", "momentum", "adam"]
histories = {}

# загружаем сохранённые истории
for opt in optimizers:
    history = np.load(f"C:\\Users\\aluas\\OneDrive\\Рабочий стол\\TeamZBA\\history_{opt}.npy", allow_pickle=True).item()
    histories[opt] = history

# строим графики точности
plt.figure(figsize=(8, 6))
for opt, hist in histories.items():
    plt.plot(hist["val_accuracy"], label=f"{opt} (val)")
plt.title("Сравнение оптимайзеров на MNIST")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("compare_optimizers.png", dpi=200)
plt.show()
