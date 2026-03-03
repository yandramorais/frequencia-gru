from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def mae(a, b): return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


class RNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, model_type="gru"):
        super().__init__()
        self.model_type = model_type.lower()
        if self.model_type == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=(dropout if num_layers > 1 else 0.0))
        elif self.model_type == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=(dropout if num_layers > 1 else 0.0))
        else:
            raise ValueError("model_type deve ser 'gru' ou 'lstm'")

        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)

def evaluate(model, dl, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            yhat = model(xb).cpu().numpy()
            preds.append(yhat)
            gts.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(gts)


def metrics_by_position(y_true: np.ndarray, y_pred: np.ndarray, pos: np.ndarray, min_n: int = 10):
    out = {}
    for p in sorted(np.unique(pos).tolist()):
        idx = (pos == p)
        if int(idx.sum()) < min_n:
            continue
        out[int(p)] = {
            "mae": mae(y_pred[idx], y_true[idx]),
            "rmse": rmse(y_pred[idx], y_true[idx]),
            "corr": corr(y_pred[idx], y_true[idx]),
            "n": int(idx.sum())
        }
    return out


def pick_best_positions(by_pos_metrics: dict, top_k: int = 5):
    items = [(p, by_pos_metrics[p]["mae"], by_pos_metrics[p]["n"]) for p in by_pos_metrics.keys()]
    items.sort(key=lambda x: x[1])  
    return [p for (p, _, _) in items[:top_k]]


def train_one(model_type: str, data_dir: Path, device, args):
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")

    p_train = np.load(data_dir / "positions_train.npy")
    p_val = np.load(data_dir / "positions_val.npy")
    p_test = np.load(data_dir / "positions_test.npy")

    _, T, F = X_train.shape

    train_dl = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32)),
        batch_size=args.batch_size, shuffle=True
    )
    val_dl = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.float32)),
        batch_size=args.batch_size, shuffle=False
    )
    test_dl = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                      torch.tensor(y_test, dtype=torch.float32)),
        batch_size=args.batch_size, shuffle=False
    )

    model = RNNRegressor(F, args.hidden_size, args.num_layers, args.dropout, model_type=model_type).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    loss_fn = nn.L1Loss()  # MAE

    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_path = data_dir / f"best_{model_type}.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        tr_loss = total_loss / len(train_dl.dataset)

        val_preds, val_gts = evaluate(model, val_dl, device)
        val_mae = mae(val_preds, val_gts)
        scheduler.step(val_mae)

        print(f"[{model_type.upper()}] Epoch {epoch:03d} | train MAE {tr_loss:.2f} | val MAE {val_mae:.2f} | {time.time()-t0:.1f}s")

        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[{model_type.upper()}] EarlyStop ({args.patience}) | best={best_val:.2f} @epoch {best_epoch}")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_preds, test_gts = evaluate(model, test_dl, device)

    global_metrics = {
        "best_val_mae": float(best_val),
        "test_mae": mae(test_preds, test_gts),
        "test_rmse": rmse(test_preds, test_gts),
        "test_corr": corr(test_preds, test_gts),
        "best_epoch": int(best_epoch)
    }

    by_pos = metrics_by_position(test_gts, test_preds, p_test, min_n=args.min_n_per_pos)

    np.save(data_dir / f"test_pred_{model_type}.npy", test_preds)
    np.save(data_dir / f"test_gt.npy", test_gts) 
    np.save(data_dir / f"test_pos.npy", p_test)

    with open(data_dir / f"metrics_{model_type}.json", "w") as f:
        json.dump(global_metrics, f, indent=2)

    with open(data_dir / f"metrics_by_position_{model_type}.json", "w") as f:
        json.dump(by_pos, f, indent=2)

    print(f"[{model_type.upper()}] Teste global:", global_metrics)
    return global_metrics, by_pos


def plot_pred_vs_real(y_true, y_pred, out_path: Path, title: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("Real (bpm)")
    plt.ylabel("Predito (bpm)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_bland_altman(y_true, y_pred, out_path: Path, title: str):
    diff = y_pred - y_true
    mean = (y_pred + y_true) / 2.0
    md = float(np.mean(diff))
    sd = float(np.std(diff))
    loa1 = md - 1.96 * sd
    loa2 = md + 1.96 * sd

    plt.figure(figsize=(7, 4))
    plt.scatter(mean, diff, alpha=0.5)
    plt.axhline(md)
    plt.axhline(loa1)
    plt.axhline(loa2)
    plt.xlabel("Média (Real+Pred)/2 [bpm]")
    plt.ylabel("Diferença (Pred-Real) [bpm]")
    plt.title(title + f" | md={md:.2f}, LoA=[{loa1:.2f},{loa2:.2f}]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_mae_by_position_compare(by_pos_gru: dict, by_pos_lstm: dict, out_path: Path):
    positions = sorted(set(by_pos_gru.keys()) | set(by_pos_lstm.keys()))
    gru = [by_pos_gru.get(p, {}).get("mae", np.nan) for p in positions]
    lstm = [by_pos_lstm.get(p, {}).get("mae", np.nan) for p in positions]

    x = np.arange(len(positions))
    width = 0.4

    plt.figure(figsize=(10, 4))
    plt.bar(x - width/2, gru, width, label="GRU")
    plt.bar(x + width/2, lstm, width, label="LSTM")
    plt.xticks(x, positions)
    plt.xlabel("Posição")
    plt.ylabel("MAE (bpm)")
    plt.title("GRU vs LSTM — MAE por posição")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_abs_error_box_by_position(y_true, pred_gru, pred_lstm, pos, out_path: Path, min_n=10):
    positions = sorted(np.unique(pos).tolist())
    data_gru, data_lstm, labels = [], [], []
    for p in positions:
        idx = (pos == p)
        if int(idx.sum()) < min_n:
            continue
        data_gru.append(np.abs(pred_gru[idx] - y_true[idx]))
        data_lstm.append(np.abs(pred_lstm[idx] - y_true[idx]))
        labels.append(str(int(p)))

    plt.figure(figsize=(12, 5))
    data = []
    xt = []
    for i, lab in enumerate(labels):
        data.append(data_gru[i]); xt.append(f"{lab}\nGRU")
        data.append(data_lstm[i]); xt.append(f"{lab}\nLSTM")

    plt.boxplot(data, showfliers=False)
    plt.xticks(np.arange(1, len(xt) + 1), xt, rotation=90)
    plt.ylabel("|Erro| (bpm)")
    plt.title("Distribuição do erro absoluto por posição (GRU vs LSTM)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_best_positions_scatter(y_true, y_pred, pos, best_positions, out_path: Path, title: str):
    idx = np.isin(pos, np.array(best_positions))
    plot_pred_vs_real(y_true[idx], y_pred[idx], out_path, title + f" | melhores pos {best_positions}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--hidden_size", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min_n_per_pos", type=int, default=10)
    ap.add_argument("--top_k_best_pos", type=int, default=5)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    plots_dir = ensure_dir(data_dir / "paper_plots")

    device = get_device()
    print("Device:", device)

    m_gru, by_gru = train_one("gru", data_dir, device, args)
    m_lstm, by_lstm = train_one("lstm", data_dir, device, args)

    y_true = np.load(data_dir / "test_gt.npy")
    pos = np.load(data_dir / "test_pos.npy")
    pred_gru = np.load(data_dir / "test_pred_gru.npy")
    pred_lstm = np.load(data_dir / "test_pred_lstm.npy")

    best_positions = pick_best_positions(by_gru, top_k=args.top_k_best_pos)

    plot_mae_by_position_compare(by_gru, by_lstm, plots_dir / "mae_by_position_gru_vs_lstm.png")

    plot_pred_vs_real(y_true, pred_gru, plots_dir / "scatter_global_gru.png", "GRU — Predito vs Real (global)")
    plot_pred_vs_real(y_true, pred_lstm, plots_dir / "scatter_global_lstm.png", "LSTM — Predito vs Real (global)")

    plot_best_positions_scatter(y_true, pred_gru, pos, best_positions, plots_dir / "scatter_bestpos_gru.png",
                                "GRU — Predito vs Real")
    plot_best_positions_scatter(y_true, pred_lstm, pos, best_positions, plots_dir / "scatter_bestpos_lstm.png",
                                "LSTM — Predito vs Real")

    plot_bland_altman(y_true, pred_gru, plots_dir / "bland_altman_gru.png", "GRU — Bland–Altman (global)")
    plot_bland_altman(y_true, pred_lstm, plots_dir / "bland_altman_lstm.png", "LSTM — Bland–Altman (global)")

    plot_abs_error_box_by_position(y_true, pred_gru, pred_lstm, pos,
                                   plots_dir / "abs_error_box_by_position.png",
                                   min_n=args.min_n_per_pos)

    summary = {
        "GRU": m_gru,
        "LSTM": m_lstm,
        "best_positions_by_gru_mae": best_positions,
        "plots_dir": str(plots_dir)
    }
    with open(data_dir / "paper_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ Paper plots salvos em:", plots_dir)
    print("✅ Resumo:", summary)


if __name__ == "__main__":
    main()
