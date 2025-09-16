import os
from typing import List
import matplotlib.pyplot as plt


def plot_series(idx, values, title: str, ylabel: str, output_dir: str, filename: str = None):
    plt.figure(figsize=(10, 5))
    plt.plot(idx, values, label=title)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    name = filename or f"{title}.jpg"
    plt.savefig(os.path.join(output_dir, name))
    plt.close()


def plot_two_series(idx, y_true, y_pred, title: str, output_dir: str, filename: str = None):
    plt.figure(figsize=(10, 5))
    plt.plot(idx, y_true, label="Actual")
    plt.plot(idx, y_pred, label="Predicted", color="red")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    name = filename or f"{title}.jpg"
    plt.savefig(os.path.join(output_dir, name))
    plt.close()


def plot_bar(labels: List[str], values: List[float], title: str, output_dir: str, filename: str):
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color=["blue", "orange", "green"][: len(values)])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def plot_errors(errors: List[float], output_dir: str):
    plt.figure(figsize=(10, 5))
    plt.plot(errors, label="Prediction Errors")
    plt.title("Prediction Errors over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Prediction Errors over Time.jpg"))
    plt.close()
