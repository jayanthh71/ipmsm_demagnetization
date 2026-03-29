import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ── Aesthetics ────────────────────────────────────────────────────────────────
DARK = "#0d1117"
PANEL = "#161b22"
GRID = "#21262d"
ACC1 = "#58a6ff"
ACC2 = "#3fb950"
ACC3 = "#f78166"
ACC4 = "#d2a8ff"
MUTED = "#8b949e"
WHITE = "#e6edf3"

plt.rcParams.update(
    {
        "figure.facecolor": DARK,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": WHITE,
        "axes.titlecolor": WHITE,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": WHITE,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "font.family": "monospace",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ── Motor nominal ─────────────────────────────────────────────────────────────
PSIM_NOM = 0.04366  # Wb — your IPMSM nominal flux linkage

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv("simulation/IPMSM_Dataset.csv")

# Drop classification columns — avoid target leakage
df = df.drop(columns=["Demag_Pct", "Demag_Class"])

# ── 1b. Clip Id_Iq_Ratio outliers ────────────────────────────────────────────
# When Iq ≈ 0 the instantaneous Id/Iq ratio explodes (max ~154k in your data).
# Clip at the 95th percentile to keep the feature informative without
# letting a few extreme values dominate the tree splits.
clip_val = df["Id_Iq_Ratio"].quantile(0.95)
n_clipped = (df["Id_Iq_Ratio"] > clip_val).sum()
df["Id_Iq_Ratio"] = df["Id_Iq_Ratio"].clip(upper=clip_val)
print(f"  Id_Iq_Ratio clipped at {clip_val:.4f}  ({n_clipped} samples affected)\n")

X = df.drop(columns=["Flux_Linkage"])
y = df["Flux_Linkage"]
feature_names = X.columns.tolist()

print("=" * 55)
print("  IPMSM DEMAGNETIZATION — RANDOM FOREST REGRESSION")
print("=" * 55)
print(f"  Samples   : {len(df)}")
print(f"  Features  : {len(feature_names)}")
print("  Target    : Flux_Linkage (Wb)")
print(f"  FL min    : {y.min():.5f} Wb  ({y.min() / PSIM_NOM * 100:.1f}% of nominal)")
print(f"  FL max    : {y.max():.5f} Wb  ({y.max() / PSIM_NOM * 100:.1f}% of nominal)")
print(f"  FL mean   : {y.mean():.5f} Wb")
print(f"  FL std    : {y.std():.5f} Wb")
print(f"  Nominal   : {PSIM_NOM:.5f} Wb")
print("=" * 55)

# ── 2. Train / Test Split — 80/20 ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"\n  Train : {len(X_train)} samples")
print(f"  Test  : {len(X_test)} samples")
print(f"  Train FL range: {y_train.min():.5f} – {y_train.max():.5f} Wb")
print(f"  Test  FL range: {y_test.min():.5f} – {y_test.max():.5f} Wb")

# ── 3. Pipeline ───────────────────────────────────────────────────────────────
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "rf",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=3,
                max_features=0.5,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

# ── 4. Cross-Validation ───────────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_r2 = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring="r2")
cv_mae = cross_val_score(
    pipeline, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error"
)
cv_mae = -cv_mae

print(f"\n  5-Fold CV R²  : {cv_r2.mean():.4f}  ± {cv_r2.std():.4f}")
print(f"  5-Fold CV MAE : {cv_mae.mean():.6f} ± {cv_mae.std():.6f} Wb")
print(f"  CV MAE as % of range: {cv_mae.mean() / (y.max() - y.min()) * 100:.2f}%")

# ── 5. Fit Final Model ────────────────────────────────────────────────────────
pipeline.fit(X_train, y_train)

# ── 6. Evaluate on Test Set ───────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
psim_range = y.max() - y.min()

print("\n  Hold-out Test Results:")
print(f"  R²   : {r2:.4f}")
print(f"  MAE  : {mae:.6f} Wb  ({mae / psim_range * 100:.2f}% of range)")
print(f"  RMSE : {rmse:.6f} Wb  ({rmse / psim_range * 100:.2f}% of range)")
print(f"  MAE in mWb : {mae * 1000:.3f} mWb")

residuals = y_test.values - y_pred

# ── 7. Save ───────────────────────────────────────────────────────────────────
joblib.dump(pipeline, "demag_regression_pipeline.pkl")
print("\n  Pipeline saved → demag_regression_pipeline.pkl")

# ── 8. Feature importances ────────────────────────────────────────────────────
rf_model = pipeline.named_steps["rf"]
importances = rf_model.feature_importances_
imp_std = np.std([t.feature_importances_ for t in rf_model.estimators_], axis=0)
imp_order = np.argsort(importances)

# ── 9. Plotting ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14), facecolor=DARK)
fig.suptitle(
    "IPMSM Flux Linkage Regression — Random Forest Results",
    fontsize=16,
    fontweight="bold",
    color=WHITE,
    y=0.98,
)

gs = gridspec.GridSpec(
    2,
    3,
    figure=fig,
    hspace=0.42,
    wspace=0.38,
    left=0.07,
    right=0.97,
    top=0.93,
    bottom=0.07,
)

ax_pred = fig.add_subplot(gs[0, 0])
ax_resid = fig.add_subplot(gs[0, 1])
ax_hist = fig.add_subplot(gs[0, 2])
ax_feat = fig.add_subplot(gs[1, 0])
ax_cv = fig.add_subplot(gs[1, 1])
ax_psim = fig.add_subplot(gs[1, 2])

# ── 9a. Predicted vs Actual ───────────────────────────────────────────────────
ax_pred.scatter(
    y_test, y_pred, alpha=0.35, s=10, color=ACC1, edgecolors="none", rasterized=True
)
lims = (y.min() - 0.002, y.max() + 0.002)
ax_pred.plot(
    lims, lims, color=ACC3, linewidth=1.5, linestyle="--", label="Perfect prediction"
)
ax_pred.set_xlim(lims)
ax_pred.set_ylim(lims)
ax_pred.set_xlabel("Actual Flux Linkage (Wb)", fontsize=9, color=MUTED)
ax_pred.set_ylabel("Predicted Flux Linkage (Wb)", fontsize=9, color=MUTED)
ax_pred.set_title(f"Predicted vs Actual\nR² = {r2:.4f}", fontweight="bold", fontsize=11)
ax_pred.legend(fontsize=8, facecolor=GRID, edgecolor="none", labelcolor=WHITE)
ax_pred.grid(alpha=0.2)

ax_pred.text(
    0.04,
    0.95,
    f"MAE  = {mae * 1000:.2f} mWb\nRMSE = {rmse * 1000:.2f} mWb",
    transform=ax_pred.transAxes,
    fontsize=8,
    color=WHITE,
    va="top",
    bbox=dict(boxstyle="round,pad=0.4", facecolor=GRID, edgecolor="none", alpha=0.8),
)

# ── 9b. Residuals vs Predicted ────────────────────────────────────────────────
ax_resid.scatter(
    y_pred, residuals, alpha=0.35, s=10, color=ACC4, edgecolors="none", rasterized=True
)
ax_resid.axhline(0, color=ACC3, linewidth=1.5, linestyle="--")
ax_resid.axhline(
    2 * residuals.std(), color=MUTED, linewidth=0.8, linestyle=":", label="±2σ"
)
ax_resid.axhline(-2 * residuals.std(), color=MUTED, linewidth=0.8, linestyle=":")
ax_resid.set_xlabel("Predicted Flux Linkage (Wb)", fontsize=9, color=MUTED)
ax_resid.set_ylabel("Residual (Wb)", fontsize=9, color=MUTED)
ax_resid.set_title(
    "Residuals vs Predicted\n(random scatter = good)", fontweight="bold", fontsize=11
)
ax_resid.legend(fontsize=8, facecolor=GRID, edgecolor="none", labelcolor=WHITE)
ax_resid.grid(alpha=0.2)

# ── 9c. Residual Histogram ────────────────────────────────────────────────────
ax_hist.hist(residuals * 1000, bins=40, color=ACC2, edgecolor="none", alpha=0.85)
ax_hist.axvline(0, color=ACC3, linewidth=1.5, linestyle="--")
ax_hist.axvline(
    2 * residuals.std() * 1000, color=MUTED, linewidth=0.8, linestyle=":", label="±2σ"
)
ax_hist.axvline(-2 * residuals.std() * 1000, color=MUTED, linewidth=0.8, linestyle=":")
ax_hist.set_xlabel("Residual (mWb)", fontsize=9, color=MUTED)
ax_hist.set_ylabel("Count", fontsize=9, color=MUTED)
ax_hist.set_title(
    "Residual Distribution\n(should be centred at 0)", fontweight="bold", fontsize=11
)
ax_hist.legend(fontsize=8, facecolor=GRID, edgecolor="none", labelcolor=WHITE)
ax_hist.grid(axis="y", alpha=0.3)

# ── 9d. Feature Importance ────────────────────────────────────────────────────
bar_colors = [ACC3 if importances[i] == importances.max() else ACC1 for i in imp_order]
bars = ax_feat.barh(
    [feature_names[i] for i in imp_order],
    importances[imp_order],
    xerr=imp_std[imp_order],
    color=bar_colors,
    edgecolor="none",
    height=0.65,
    error_kw=dict(ecolor=MUTED, capsize=3, linewidth=1),
)
ax_feat.set_xlabel("Mean Decrease in Impurity", fontsize=9, color=MUTED)
ax_feat.set_title("Feature Importances", fontweight="bold", fontsize=11)
ax_feat.grid(axis="x", alpha=0.3)
ax_feat.tick_params(axis="y", labelsize=9)
for bar, val in zip(bars, importances[imp_order]):
    ax_feat.text(
        val + 0.002,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}",
        va="center",
        fontsize=8,
        color=MUTED,
    )

# ── 9e. CV R² per Fold ────────────────────────────────────────────────────────
fold_labels = [f"Fold {i + 1}" for i in range(5)]
bar_cols_cv = [
    ACC2 if v == cv_r2.max() else ACC3 if v == cv_r2.min() else ACC1 for v in cv_r2
]
ax_cv.bar(fold_labels, cv_r2, color=bar_cols_cv, edgecolor="none", width=0.55)
ax_cv.axhline(
    cv_r2.mean(),
    color=ACC4,
    linewidth=1.5,
    linestyle="--",
    label=f"Mean = {cv_r2.mean():.4f}",
)
ax_cv.axhline(r2, color=ACC3, linewidth=1.5, linestyle=":", label=f"Test  = {r2:.4f}")
ax_cv.set_ylim(max(0, cv_r2.min() - 0.05), 1.02)
ax_cv.set_ylabel("R²", fontsize=9, color=MUTED)
ax_cv.set_title("5-Fold Cross-Validation R²", fontweight="bold", fontsize=11)
ax_cv.legend(fontsize=8, facecolor=GRID, edgecolor="none", labelcolor=WHITE)
ax_cv.grid(axis="y", alpha=0.3)
for i, (x, v) in enumerate(zip(fold_labels, cv_r2)):
    ax_cv.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=8, color=MUTED)

# ── 9f. MAE bucketed by true psim level ──────────────────────────────────────
n_bins = 10
bin_edges = np.linspace(y_test.min(), y_test.max(), n_bins + 1)
bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_maes = []
bin_counts = []
for i in range(n_bins):
    mask = (y_test.values >= bin_edges[i]) & (y_test.values < bin_edges[i + 1])
    if mask.sum() > 0:
        bin_maes.append(np.abs(residuals[mask]).mean() * 1000)
        bin_counts.append(mask.sum())
    else:
        bin_maes.append(0)
        bin_counts.append(0)

bars_psim = ax_psim.bar(
    bin_mids * 1000,
    bin_maes,
    width=(bin_edges[1] - bin_edges[0]) * 1000 * 0.8,
    color=ACC1,
    edgecolor="none",
    alpha=0.85,
)
ax_psim.axhline(
    mae * 1000,
    color=ACC3,
    linewidth=1.5,
    linestyle="--",
    label=f"Overall MAE = {mae * 1000:.2f} mWb",
)
ax_psim.axvline(
    PSIM_NOM * 1000,
    color=MUTED,
    linewidth=1,
    linestyle=":",
    alpha=0.6,
    label=f"Nominal ({PSIM_NOM * 1000:.2f} mWb)",
)
ax_psim.set_xlabel("True Flux Linkage (mWb)", fontsize=9, color=MUTED)
ax_psim.set_ylabel("MAE (mWb)", fontsize=9, color=MUTED)
ax_psim.set_title(
    "Prediction Error by Flux Linkage Level\n(uniform = no bias)",
    fontweight="bold",
    fontsize=11,
)
ax_psim.legend(fontsize=8, facecolor=GRID, edgecolor="none", labelcolor=WHITE)
ax_psim.grid(axis="y", alpha=0.3)

for bar, count in zip(bars_psim, bin_counts):
    ax_psim.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.03,
        f"n={count}",
        ha="center",
        fontsize=7,
        color=MUTED,
    )

plt.savefig(
    "demag_regression_results.png", dpi=150, bbox_inches="tight", facecolor=DARK
)
print("  Plot saved → demag_regression_results.png")
plt.close()
print("\n  Done.")
