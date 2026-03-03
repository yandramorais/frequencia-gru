
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.name == "src") else SCRIPT_DIR

DATA_DIR = PROJECT_DIR / "data" / "processed_gt_20s"
OUT_DIR = DATA_DIR / "paper_outputs_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 300
})

def load_npy(name: str):
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return np.load(path)

def load_json(name: str):
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    with open(path) as f:
        return json.load(f)

y_real = load_npy("y_test.npy")
pred_gru = load_npy("test_pred_gru.npy")
pred_lstm = load_npy("test_pred_lstm.npy")

posicoes = load_npy("positions_test.npy")
participantes = load_npy("subject_test.npy")

posicoes = posicoes.astype(int)
participantes = participantes.astype(int)

bypos_gru = load_json("metrics_by_position_gru.json")
bypos_lstm = load_json("metrics_by_position_lstm.json")

pos_unicas = sorted(np.unique(posicoes).astype(int))

mae_global_gru = np.mean(np.abs(pred_gru - y_real))
mae_global_lstm = np.mean(np.abs(pred_lstm - y_real))

print("\n==============================")
print("MAE GLOBAL")
print(f"GRU  : {mae_global_gru:.3f} bpm")
print(f"LSTM : {mae_global_lstm:.3f} bpm")
print("==============================\n")

mae_por_pos_gru = []
mae_por_pos_lstm = []

print("MAE POR POSIÇÃO")
for p in pos_unicas:
    mae_g = float(bypos_gru[str(int(p))]["mae"])
    mae_l = float(bypos_lstm[str(int(p))]["mae"])
    mae_por_pos_gru.append(mae_g)
    mae_por_pos_lstm.append(mae_l)
    print(f"Pos {int(p):2d} | GRU: {mae_g:.3f} | LSTM: {mae_l:.3f}")
print("==============================\n")

plt.figure(figsize=(6,4))
plt.bar(["GRU", "LSTM"], [mae_global_gru, mae_global_lstm])
plt.ylabel("MAE Global (bpm)")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "figura1_mae_global.png")
plt.close()

ranking = sorted(
    pos_unicas,
    key=lambda p: (bypos_gru[str(int(p))]["mae"] + bypos_lstm[str(int(p))]["mae"]) / 2
)

mae_rank = [
    (bypos_gru[str(int(p))]["mae"] + bypos_lstm[str(int(p))]["mae"]) / 2
    for p in ranking
]

cores = ["green"] + ["steelblue"]*(len(ranking)-2) + ["red"]
ranking_labels = [str(int(p)) for p in ranking]

plt.figure(figsize=(9,4))
plt.bar(ranking_labels, mae_rank, color=cores)

plt.xlabel("Posições (Melhor → Pior)")
plt.ylabel("MAE Médio (bpm)")
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "figura2_ranking.png")
plt.close()

melhor_pos = int(ranking[0])
pior_pos = int(ranking[-1])

plt.figure(figsize=(6,6))
plt.scatter(y_real, pred_gru, alpha=0.35, label="GRU")
plt.scatter(y_real, pred_lstm, alpha=0.25, label="LSTM")

min_val = min(y_real.min(), pred_gru.min(), pred_lstm.min())
max_val = max(y_real.max(), pred_gru.max(), pred_lstm.max())

plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1)
plt.xlabel("FC Real (bpm)")
plt.ylabel("FC Predita (bpm)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "figura3_scatter.png")
plt.close()

mean_vals = (y_real + pred_gru) / 2
diff_vals = pred_gru - y_real

bias = np.mean(diff_vals)
std = np.std(diff_vals)

plt.figure(figsize=(7,6))
cmap = mpl.colormaps.get_cmap('tab20')

for i, pos in enumerate(pos_unicas):
    mask = posicoes == int(pos)
    plt.scatter(
        mean_vals[mask],
        diff_vals[mask],
        alpha=0.5,
        s=22,
        color=cmap(i / len(pos_unicas)),
        label=f"P{int(pos)}"  
    )

plt.axhline(bias, linestyle='--', linewidth=1.5, label="Viés")
plt.axhline(bias + 1.96*std, linestyle='--', color='red')
plt.axhline(bias - 1.96*std, linestyle='--', color='red')

plt.xlabel("Média (bpm)")
plt.ylabel("Diferença (bpm)")
plt.legend(bbox_to_anchor=(1.02,1), fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "figura4_bland_altman.png")
plt.close()

media_real = [np.mean(y_real[posicoes == int(p)]) for p in pos_unicas]
media_gru  = [np.mean(pred_gru[posicoes == int(p)]) for p in pos_unicas]
media_lstm = [np.mean(pred_lstm[posicoes == int(p)]) for p in pos_unicas]

pos_labels = [str(int(p)) for p in pos_unicas]

plt.figure(figsize=(9,4))
plt.plot(pos_labels, media_real, marker='o', linewidth=2, label="Real")
plt.plot(pos_labels, media_gru, marker='o', label="GRU")
plt.plot(pos_labels, media_lstm, marker='o', label="LSTM")

plt.xlabel("Posição do Sensor")
plt.ylabel("Frequência Cardíaca Média (bpm)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "figura5_media_posicao.png")
plt.close()

plt.figure(figsize=(10,3))

plt.plot(y_real[:300], linewidth=2, label="Real")
plt.plot(pred_gru[:300], label="GRU")
plt.plot(pred_lstm[:300], label="LSTM")

plt.xlabel("Janela Temporal (índice da janela)")
plt.ylabel("Frequência Cardíaca (bpm)")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "figura6_serie_global.png")
plt.close()

mask = posicoes == melhor_pos

plt.figure(figsize=(10,3))

plt.plot(y_real[mask][:200], linewidth=2, label="Real")
plt.plot(pred_gru[mask][:200], label="GRU")
plt.plot(pred_lstm[mask][:200], label="LSTM")

plt.xlabel("Janela Temporal (índice da janela)")
plt.ylabel("Frequência Cardíaca (bpm)")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "fig7.png")
plt.close()

parts, counts = np.unique(participantes, return_counts=True)
participante_escolhido = int(parts[np.argmax(counts)])

mask_part = participantes == participante_escolhido

plt.figure(figsize=(10,4))

for pos in pos_unicas:
    maskp = mask_part & (posicoes == int(pos))
    if np.sum(maskp) == 0:
        continue
    y_vals = y_real[maskp]
    x_vals = np.arange(len(y_vals))
    plt.plot(x_vals, y_vals, marker='.', linestyle='-', alpha=0.7, label=f"P{int(pos)}")

plt.xlabel("Janela (ordem no test)")
plt.ylabel("FC (bpm)")
plt.legend(bbox_to_anchor=(1.02,1), fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "figura8_participante.png")
plt.close()

pos_escolhida = melhor_pos

plt.figure(figsize=(10,4))

for part in np.unique(participantes):
    part = int(part)
    maskp = (posicoes == pos_escolhida) & (participantes == part)
    if np.sum(maskp) == 0:
        continue
    y_vals = y_real[maskp]
    x_vals = np.arange(len(y_vals))
    plt.plot(x_vals, y_vals, marker='.', linestyle='-', alpha=0.6, label=f"S{part}")

plt.xlabel("Janela (ordem no test)")
plt.ylabel("FC (bpm)")
plt.legend(bbox_to_anchor=(1.02,1), fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "figura9_posicao.png")
plt.close()


x = np.arange(len(pos_unicas))
width = 0.35

plt.figure(figsize=(10,4))
plt.bar(x - width/2, mae_por_pos_gru, width, label="GRU")
plt.bar(x + width/2, mae_por_pos_lstm, width, label="LSTM")
plt.xticks(x, [str(int(p)) for p in pos_unicas]) 
plt.xlabel("Posições")
plt.ylabel("MAE (bpm)")
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "figura10_comparacao_modelos.png")
plt.close()

print("TODAS as figuras geradas em:", OUT_DIR)
print(f"Melhor posição: {melhor_pos}")
print(f"Pior posição: {pior_pos}")