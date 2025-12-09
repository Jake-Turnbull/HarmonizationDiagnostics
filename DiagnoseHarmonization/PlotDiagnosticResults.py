#!/usr/bin/env python3
"""
PlotDiagnosticResults.py  (updated)

Improved combined plotting for multiple 'results' directories produced by your pipeline.
- Normalizes IDP/site/timepair names and aligns data by name (not order).
- Unified StyleBank for deterministic color+marker mapping across all plots.
- Robust column detection for ICC/WCV/n_is_batchSig and numeric coercion.
- Adds legends for IDPs/sites/timepairs and p-value categories.
- Saves combined PNGs into outdir.

Usage example:
python PlotDiagnosticResults.py --dirs results1 results2 --outdir combined_plots --fix_eff age sex
"""
from __future__ import annotations
import os, re, json, argparse
from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- global style ----
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
STAR = u"\u2605"
_MARKERS = ["o","s","D","v","^","P","X","H","<",">","8","p","*"]

INTERP_FS_DELTA = 2

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def tidy_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # NEW — add grids
    ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.6, color='#dddddd')
    ax.set_axisbelow(True)


def normalize_key(s: Any) -> str:
    if s is None: return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9a-z_]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

def _tone_hex(hexcolor: str, factor: float = 0.88) -> str:
    hexcolor = hexcolor.lstrip("#")
    r = int(hexcolor[0:2], 16); g = int(hexcolor[2:4], 16); b = int(hexcolor[4:6], 16)
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))
    return f"#{r:02x}{g:02x}{b:02x}"

class StyleBank:
    """Deterministic style assignment for keys (idp/site/timepair)."""
    def __init__(self):
        self._map = {}
        self._colors = plt.get_cmap("Accent").colors   # brighter palette
        self._next = 0
    def get(self, key: str) -> Dict[str,str]:
        nk = normalize_key(key)
        if nk in self._map:
            return self._map[nk]
        i = self._next
        color = mpl.colors.to_hex(self._colors[i % len(self._colors)])  # no darkening
        color = _tone_hex(color, 0.86)
        marker = _MARKERS[i % len(_MARKERS)]
        self._map[nk] = {"color": color, "marker": marker, "label": key}
        self._next += 1
        return self._map[nk]
    def items(self):
        return list(self._map.items())

# ---- loaders ----
def load_results_dir(d: str) -> Dict[str, pd.DataFrame]:
    files = {
        "add_test": "add_test.csv",
        "mult_test": "mult_test.csv",
        "md_by_site": "md_by_site.csv",
        "mixed": "mixed_models_results.csv",
        "pairwise_spearman": "pairwise_spearman.csv",
        "wsv": "wsv_table.csv",
    }
    out = {}
    for k, fn in files.items():
        p = os.path.join(d, fn)
        if os.path.exists(p):
            try:
                out[k] = pd.read_csv(p)
            except Exception:
                out[k] = pd.DataFrame()
        else:
            out[k] = pd.DataFrame()
    return out

def find_col_any(df: pd.DataFrame, candidates: List[str]):
    if df is None or df.empty: return None
    cols = list(df.columns)
    norm = {normalize_key(c): c for c in cols}
    for cand in candidates:
        nk = normalize_key(cand)
        if nk in norm: return norm[nk]
    for c in cols:
        for cand in candidates:
            if cand.lower() in c.lower(): return c
    return None

# ---- plotting helpers ----
def _shorten_label(s, max_chars=45):
    s = str(s)
    if len(s) <= max_chars: return s
    return s[:max_chars-3] + "..."

# Heatmap for add/mult tests
def plot_add_mult_heatmap(all_results, key, outdir):
    features = []
    for label, d in all_results:
        df = d.get(key, pd.DataFrame())
        if df is None or df.empty:
            continue
        feat_col = None
        for c in ["Feature", "feature", "feature_name", "FeatureName"]:
            if c in df.columns:
                feat_col = c
                break
        if feat_col is None and df.shape[1] > 0:
            feat_col = df.columns[0]
        if feat_col is None:
            continue
        features.extend(df[feat_col].astype(str).tolist())
    # keep original order but unique
    features = list(dict.fromkeys(features))
    cols = [lab for lab, _ in all_results]
    mat = pd.DataFrame(index=features, columns=cols, dtype=float)

    for lab, d in all_results:
        df = d.get(key, pd.DataFrame())
        if df is None or df.empty:
            continue
        pcol = find_col_any(df, ["p-value", "p.value", "pvalue", "p"])
        feat_col = None
        for c in ["Feature", "feature", "feature_name"]:
            if c in df.columns:
                feat_col = c
                break
        if feat_col is None and df.shape[1] > 0:
            feat_col = df.columns[0]
        if pcol is None or feat_col is None:
            continue
        for _, r in df.iterrows():
            try:
                f = str(r[feat_col])
                p = float(r[pcol])
            except Exception:
                p = np.nan
            mat.at[f, lab] = p

    # compute Bonferroni threshold
    n_features = len(features)
    alpha = 0.05
    if n_features > 0:
        bonf_thresh = alpha / float(n_features)
    else:
        bonf_thresh = alpha
    # plot
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(cols)), max(6, 0.35 * len(features))))
    tidy_axes(ax)
    ax.grid(False)
    arr = mat.fillna(np.nan).to_numpy(dtype=float)
    im = ax.imshow(arr, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels([_shorten_label(s) for s in features], fontsize=9)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right', fontsize=9)

    # annotate with p-value and star only if p < bonf_thresh
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isnan(val):
                txt = ""
            else:
                txt = f"{val:.2f}"
                if val < bonf_thresh:
                    txt += " " + STAR
            ax.text(j, i, txt, ha='center', va='center', fontsize=9, color='black')

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04, label='p-value')
    ax.set_title(f"{key} p-values")

    interp_fs = max(8, mpl.rcParams.get("font.size", 10) - INTERP_FS_DELTA)
    fig.subplots_adjust(bottom=0.22)

    # show Bonferroni info in caption
    if n_features > 0:
        thresh_str = f"{bonf_thresh:.3e}" if bonf_thresh < 1e-3 else f"{bonf_thresh:.4f}"
        fig.text(
            0.5,
            0.015,
            f"Interpretation: {STAR} indicates presence of significant batch effect, p < Bonferroni threshold ({alpha}/{n_features} = {thresh_str}).",
            ha='center',
            fontsize=interp_fs,
            color='blue',
            style='italic'
        )
    else:
        fig.text(
            0.5,
            0.015,
            f"Interpretation: {STAR} indicates p < 0.05 (no features detected).",
            ha='center',
            fontsize=interp_fs,
            color='blue',
            style='italic'
        )

    ensure_dir(outdir)
    fig.savefig(os.path.join(outdir, f"{key}_pvalues_combined.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)

# MD by site
def plot_md_by_site(all_results, outdir, stylebank):
    labels = [lab for lab, _ in all_results]
    data_lists = []
    # collect all site keys
    for lab, dct in all_results:
        df = dct.get("md_by_site", pd.DataFrame())
        if df is None or df.empty: data_lists.append([]); continue
        if 'batch' in df.columns and 'mdval' in df.columns:
            vals = df.loc[df['batch']!='average_batch','mdval'].astype(float).tolist()
        elif 'mdval' in df.columns:
            vals = df['mdval'].astype(float).tolist()
        else:
            vals = df.iloc[:,1].astype(float).tolist() if df.shape[1]>1 else []
        data_lists.append(vals)
    fig, ax = plt.subplots(figsize=(max(8,1.2*len(labels)),6))
    tidy_axes(ax)
    positions = np.arange(1,len(labels)+1)
    ax.boxplot([v if len(v)>0 else [np.nan] for v in data_lists], positions=positions, widths=0.6, patch_artist=True, boxprops=dict(facecolor='none'))
    # scatter per-site using stylebank per site if possible
    # collect all site names across dirs for legend
    all_sites = []
    for lab, dct in all_results:
        df = dct.get("md_by_site", pd.DataFrame())
        if df is None or df.empty: continue
        if 'batch' in df.columns:
            all_sites.extend(df.loc[df['batch']!='average_batch','batch'].astype(str).tolist())
    all_sites = list(dict.fromkeys(all_sites))
    for i, (lab, dct) in enumerate(all_results):
        df = dct.get("md_by_site", pd.DataFrame()); 
        if df is None or df.empty: continue
        if 'batch' in df.columns and 'mdval' in df.columns:
            for _, r in df.iterrows():
                if str(r['batch'])=='average_batch': continue
                sitek = str(r['batch']); val = float(r['mdval'])
                st = stylebank.get(sitek)
                ax.scatter(i+1 + np.random.uniform(-0.08,0.08), val, marker=st['marker'], color=st['color'], edgecolor=_tone_hex(st['color'],0.6), s=60, zorder=4)
    ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("MD value"); ax.set_title("Mahalanobis distance for each site")
    # legend for sites
    if len(all_sites)>0 and len(all_sites)<=40:
        handles = [plt.Line2D([],[], marker=stylebank.get(s)['marker'], color=stylebank.get(s)['color'], linestyle='None', markersize=8, markeredgecolor=_tone_hex(stylebank.get(s)['color'],0.6)) for s in all_sites]
        ax.legend(handles, all_sites, bbox_to_anchor=(1.02,0.5), loc='center left', fontsize=8, title='Site', frameon=True)
        fig.subplots_adjust(right=0.72)
    fig.subplots_adjust(bottom=0.26)
    interp_fs = max(8, mpl.rcParams.get("font.size",10)-INTERP_FS_DELTA)
    fig.text(0.5, 0.02, "Interpretation: Lower values are better", ha='center', fontsize=interp_fs, color='blue', style='italic')
    ensure_dir(outdir); fig.savefig(os.path.join(outdir, "md_by_site_combined.png"), dpi=200, bbox_inches='tight'); plt.close(fig)

# Mixed single metrics (ICC/WCV/n_is_batchSig)
def plot_mixed_single_metrics(all_results, outdir, stylebank):
    labels = [lab for lab, _ in all_results]
    metrics = [
        (["ICC","icc"], "mixed_ICC_combined.png", "ICC", "Ratio of subject variability and total variability", "Interpretation: Values close to 1 are better"),
        (["WCV","wcv"], "mixed_WCV_combined.png", "WCV", "Ratio of within-subject (W) and cross-subject (C) variability", "Interpretation: Lower values are better"),
        (["n_is_batchSig","n_is_batchsig","n_is_batch_sig"], "mixed_n_is_batchSig_combined.png", "n_is_batchSig", "Total count of pairs of sites significant (p<0.05)", "Interpretation: Lower values are better"),
    ]
    for cand_list, fname, colname, title, interp in metrics:
        boxlists = []
        idp_universe = []
        per_dir_rows = []
        for lab, dct in all_results:
            df = dct.get("mixed", pd.DataFrame())
            if df is None or df.empty:
                boxlists.append([]); per_dir_rows.append(None); continue
            col = find_col_any(df, cand_list)
            per_dir_rows.append((df, col))
            if col is None:
                boxlists.append([]); continue
            vals = pd.to_numeric(df[col], errors='coerce').to_numpy()
            vals = vals[~np.isnan(vals)].tolist()
            boxlists.append(vals)
            if 'IDP' in df.columns:
                idp_universe.extend(df['IDP'].astype(str).tolist())
        idp_universe = list(dict.fromkeys(idp_universe))
        # prepare plot
        fig, ax = plt.subplots(figsize=(max(8,1.2*len(labels)),6))
        tidy_axes(ax)
        positions = np.arange(1,len(labels)+1)
        ax.boxplot([v if len(v)>0 else [np.nan] for v in boxlists], positions=positions, widths=0.6, patch_artist=True, boxprops=dict(facecolor='none'))
        # scatter each IDP if possible
        for i, (lab, dct) in enumerate(all_results):
            dfcol = per_dir_rows[i]
            if dfcol is None: continue
            df, col = dfcol
            if col is None or 'IDP' not in df.columns: continue
            for _, r in df.iterrows():
                try:
                    v = float(pd.to_numeric(r[col], errors='coerce'))
                except Exception:
                    continue
                idp = str(r['IDP']); st = stylebank.get(idp)
                ax.scatter(i+1 + np.random.uniform(-0.08,0.08), v, color=st['color'], marker=st['marker'], edgecolor=_tone_hex(st['color'],0.6), s=60, linewidths=0.8, zorder=4)
        ax.set_xticks(positions); ax.set_xticklabels(labels, rotation=45, ha='right'); ax.set_ylabel(colname); ax.set_title(title)
        # legend for IDPs
        if len(idp_universe)>0 and len(idp_universe)<=40:
            handles = [plt.Line2D([],[], marker=stylebank.get(idp)['marker'], color=stylebank.get(idp)['color'], linestyle='None', markersize=8, markeredgecolor=_tone_hex(stylebank.get(idp)['color'],0.6)) for idp in idp_universe]
            ax.legend(handles, idp_universe, bbox_to_anchor=(1.02,0.5), loc='center left', fontsize=8, title='IDP', frameon=True)
            fig.subplots_adjust(right=0.72)
        fig.subplots_adjust(bottom=0.26)
        fig.text(0.5, 0.02, interp, ha='center', fontsize=max(8,mpl.rcParams.get("font.size",10)-INTERP_FS_DELTA), color='blue', style='italic')
        ensure_dir(outdir); fig.savefig(os.path.join(outdir, fname), dpi=200, bbox_inches='tight'); plt.close(fig)

# Pairwise spearman combined
def plot_pairwise_spearman_combined(all_results, outdir, stylebank):
    labels = [lab for lab, _ in all_results]
    per_dir_idp_avgs = []; per_dir_tp_avgs = []; idps = []; tps = []
    for lab, dct in all_results:
        df = dct.get("pairwise_spearman", pd.DataFrame())
        if df is None or df.empty:
            per_dir_idp_avgs.append({}); per_dir_tp_avgs.append({}); continue
        if not set(['TimeA','TimeB','IDP','SpearmanRho']).issubset(df.columns):
            per_dir_idp_avgs.append({}); per_dir_tp_avgs.append({}); continue
        idp_avg = df.groupby('IDP')['SpearmanRho'].mean().to_dict(); per_dir_idp_avgs.append(idp_avg); idps.extend(list(idp_avg.keys()))
        tp = df.groupby(['TimeA','TimeB'])['SpearmanRho'].mean(); tp_idx = ['%s|%s'%(a,b) for (a,b) in tp.index.tolist()]; per_dir_tp_avgs.append(dict(zip(tp_idx, tp.values.tolist()))); tps.extend(tp_idx)
    idps = list(dict.fromkeys(idps)); tps = list(dict.fromkeys(tps))
    # IDP avg boxplot
    boxdata = [[per_dir_idp_avgs[i].get(idp, np.nan) for idp in idps] for i in range(len(per_dir_idp_avgs))]
    fig, ax = plt.subplots(figsize=(max(8,1.2*len(labels)),6)); tidy_axes(ax)
    ax.boxplot([ [v for v in arr if not np.isnan(v)] for arr in boxdata], positions=np.arange(1,len(labels)+1), widths=0.6, patch_artist=True, boxprops=dict(facecolor='none'))
    # scatter each idp
    for i, d in enumerate(per_dir_idp_avgs):
        for idp in idps:
            val = d.get(idp, np.nan); 
            if np.isnan(val): continue
            st = stylebank.get(idp); ax.scatter(i+1 + np.random.uniform(-0.08,0.08), val, marker=st['marker'], color=st['color'], edgecolor=_tone_hex(st['color'],0.6), s=60)
    ax.set_xticks(np.arange(1,len(labels)+1)); ax.set_xticklabels(labels, rotation=45, ha='right'); ax.set_ylabel("Average Spearman rho across timepairs (per IDP)"); ax.set_title("Per-IDP average rank correlation")
    # IDP legend
    if len(idps)>0 and len(idps)<=40:
        handles = [plt.Line2D([],[], marker=stylebank.get(idp)['marker'], color=stylebank.get(idp)['color'], linestyle='None', markersize=8, markeredgecolor=_tone_hex(stylebank.get(idp)['color'],0.6)) for idp in idps]
        ax.legend(handles, idps, bbox_to_anchor=(1.02,0.5), loc='center left', fontsize=8, title='IDP', frameon=True); fig.subplots_adjust(right=0.72)
    fig.subplots_adjust(bottom=0.26); fig.text(0.5, 0.02, "Interpretation: Higher values indicate better preservation of subject order", ha='center', fontsize=max(8,mpl.rcParams.get("font.size",10)-INTERP_FS_DELTA), color='blue', style='italic')
    ensure_dir(outdir); fig.savefig(os.path.join(outdir, "pairwise_spearman_idpavg_combined.png"), dpi=200, bbox_inches='tight'); plt.close(fig)
    # timepair avg boxplot (use style per timepair)
    boxdata_tp = [[per_dir_tp_avgs[i].get(tp, np.nan) for tp in tps] for i in range(len(per_dir_tp_avgs))]
    fig, ax = plt.subplots(figsize=(max(8,1.2*len(labels)),6)); tidy_axes(ax)
    ax.boxplot([ [v for v in arr if not np.isnan(v)] for arr in boxdata_tp], positions=np.arange(1,len(labels)+1), widths=0.6, patch_artist=True, boxprops=dict(facecolor='none'))
    # scatter per timepair using stylebank keys as the timepair string
    for i, d in enumerate(per_dir_tp_avgs):
        for tp in tps:
            val = d.get(tp, np.nan)
            if np.isnan(val): continue
            st = stylebank.get(tp); ax.scatter(i+1 + np.random.uniform(-0.08,0.08), val, marker=st['marker'], color=st['color'], edgecolor=_tone_hex(st['color'],0.6), s=60)
    ax.set_xticks(np.arange(1,len(labels)+1)); ax.set_xticklabels(labels, rotation=45, ha='right'); ax.set_ylabel("Average Spearman rho across IDPs (per timepair)"); ax.set_title("Per-timepair average rank correlation")
    if len(tps)>0 and len(tps)<=40:
        handles = [plt.Line2D([],[], marker=stylebank.get(tp)['marker'], color=stylebank.get(tp)['color'], linestyle='None', markersize=8, markeredgecolor=_tone_hex(stylebank.get(tp)['color'],0.6)) for tp in tps]
        ax.legend(handles, tps, bbox_to_anchor=(1.02,0.5), loc='center left', fontsize=8, title='TimePair', frameon=True); fig.subplots_adjust(right=0.72)
    fig.subplots_adjust(bottom=0.26); fig.text(0.5, 0.02, "Interpretation: Higher values indicate better preservation of subject order", ha='center', fontsize=max(8,mpl.rcParams.get("font.size",10)-INTERP_FS_DELTA), color='blue', style='italic')
    ensure_dir(outdir); fig.savefig(os.path.join(outdir, "pairwise_spearman_timepairavg_combined.png"), dpi=200, bbox_inches='tight'); plt.close(fig)

# WSV combined
def plot_wsv_combined(all_results, outdir, stylebank):
    labels = [lab for lab, _ in all_results]
    idps = []; per_dir_means = []
    for lab, dct in all_results:
        df = dct.get("wsv", pd.DataFrame())
        if df is None or df.empty:
            per_dir_means.append({}); continue
        numeric = df.select_dtypes(include=[np.number])
        means = numeric.mean(axis=0).to_dict(); per_dir_means.append({str(k):float(v) for k,v in means.items()}); idps.extend(list(means.keys()))
    idps = list(dict.fromkeys(idps))
    boxlists = [[per_dir_means[i].get(idp, np.nan) for idp in idps] for i in range(len(per_dir_means))]
    fig, ax = plt.subplots(figsize=(max(8,1.2*len(labels)),6)); tidy_axes(ax)
    ax.boxplot([ [v for v in arr if not np.isnan(v)] for arr in boxlists], positions=np.arange(1,len(labels)+1), widths=0.6, patch_artist=True, boxprops=dict(facecolor='none'))
    for i, d in enumerate(per_dir_means):
        for idp in idps:
            val = d.get(idp, np.nan)
            if np.isnan(val): continue
            st = stylebank.get(idp); ax.scatter(i+1 + np.random.uniform(-0.08,0.08), val, marker=st['marker'], color=st['color'], edgecolor=_tone_hex(st['color'],0.6), s=60)
    ax.set_xticks(np.arange(1,len(labels)+1)); ax.set_xticklabels(labels, rotation=45, ha='right'); ax.set_ylabel("Within-subject variability (%)"); ax.set_title("Within-subject variability (%) [mean across subjects]")
    if len(idps)>0 and len(idps)<=40:
        handles = [plt.Line2D([],[], marker=stylebank.get(idp)['marker'], color=stylebank.get(idp)['color'], linestyle='None', markersize=8, markeredgecolor=_tone_hex(stylebank.get(idp)['color'],0.6)) for idp in idps]
        ax.legend(handles, idps, bbox_to_anchor=(1.02,0.5), loc='center left', fontsize=8, title='IDP', frameon=True); fig.subplots_adjust(right=0.72)
    fig.subplots_adjust(bottom=0.28); fig.text(0.5, 0.02, "Interpretation: Lower values indicate reduction of within-subject difference.", ha='center', fontsize=max(8,mpl.rcParams.get("font.size",10)-INTERP_FS_DELTA), color='blue', style='italic')
    ensure_dir(outdir); fig.savefig(os.path.join(outdir, "wsv_boxplots_combined.png"), dpi=200, bbox_inches='tight'); plt.close(fig)

def plot_mixed_fixeff(all_results, outdir, stylebank, fix_eff):
    """
    For each fixed-effect in fix_eff, create one grouped MATLAB-style figure
    across all result directories.  (One PNG per fix_eff.)
    """
    ensure_dir(outdir)
    labels = [lab for lab, _ in all_results]

    # collect global IDP universe and per-dir dfs
    idp_universe = []
    per_dir_info = []
    for lab, dct in all_results:
        df = dct.get("mixed", pd.DataFrame())
        per_dir_info.append(df)
        if isinstance(df, pd.DataFrame) and not df.empty:
            if "IDP" in df.columns:
                idp_universe.extend(df["IDP"].astype(str).tolist())
            elif "Feature" in df.columns:
                idp_universe.extend(df["Feature"].astype(str).tolist())
    idp_universe = list(dict.fromkeys(idp_universe))
    if len(idp_universe) == 0:
        print("No IDPs found in mixed model results!")
        return

    def find_col(df, candidates):
        if df is None or df.empty:
            return None
        for c in candidates:
            if c in df.columns:
                return c
        for c in df.columns:
            for cand in candidates:
                if cand.lower() in c.lower():
                    return c
        return None

    def pfill(p):
        try:
            p = float(p)
        except:
            return "#bdbdbd"
        if p < 0.05:
            return "#d73027"
        if p < 0.1:
            return "#fdae61"
        return "#bdbdbd"

    # layout constants
    group_gap = 2.0
    offsets_width = 0.65
    centers = np.arange(0, len(labels) * group_gap, group_gap)
    fig_width_base = max(10, len(labels) * 1.4)

    # For each fixed-effect variable, create a separate plot
    for varname in fix_eff:
        fig, ax = plt.subplots(figsize=(fig_width_base, 7))
        tidy_axes(ax)
        ax.axhline(0, color='k', linestyle='--', linewidth=1.1)

        for xi, (lab, dct) in enumerate(all_results):
            df = dct.get("mixed", pd.DataFrame())
            if df is None or df.empty or ("IDP" not in df.columns and "Feature" not in df.columns):
                continue

            est_col = find_col(df, [f"{varname}_est", f"zscore_{varname}_est", f"{varname} est", f"{varname}_estimate"])
            cil_col = find_col(df, [f"{varname}_ciL", f"{varname}_cil", f"{varname}_ci_l", f"{varname}_ci_lower"])
            ciu_col = find_col(df, [f"{varname}_ciU", f"{varname}_ciu", f"{varname}_ci_u", f"{varname}_ci_upper"])
            p_col = find_col(df, [f"{varname}_pval", f"{varname}_p_val", f"{varname}_p", f"{varname}_pvalue"])

            if est_col is None:
                # nothing to plot for this folder/var
                continue

            # which idps exist in this folder
            idp_col = "IDP" if "IDP" in df.columns else ("Feature" if "Feature" in df.columns else None)
            if idp_col is None:
                continue
            this_idps_sorted = sorted(df[idp_col].astype(str).unique().tolist())
            n_feats = len(this_idps_sorted)
            if n_feats == 1:
                offsets = np.array([0.0])
            else:
                offsets = np.linspace(-offsets_width/2, offsets_width/2, n_feats)

            for off, feat in zip(offsets, this_idps_sorted):
                row = df[df[idp_col].astype(str) == feat]
                if row.shape[0] == 0:
                    continue
                r = row.iloc[0]

                est = pd.to_numeric(r.get(est_col), errors='coerce')
                if pd.isna(est):
                    continue
                cil = pd.to_numeric(r.get(cil_col), errors='coerce') if cil_col else np.nan
                ciu = pd.to_numeric(r.get(ciu_col), errors='coerce') if ciu_col else np.nan
                pval = pd.to_numeric(r.get(p_col), errors='coerce') if p_col else np.nan

                st = stylebank.get(feat)
                face = pfill(pval)
                xpos = centers[xi] + off

                if pd.notna(cil) and pd.notna(ciu):
                    ax.vlines(xpos, cil, ciu, color=st["color"], linewidth=1.8, alpha=0.85, zorder=2)

                ax.scatter(xpos, est, marker=st["marker"], s=80,
                           facecolor=face, edgecolor="k", linewidths=1.0, zorder=5)

        ax.set_xticks(centers)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        #ax.set_xlabel("Procedure / result dir (method)")
        ax.set_ylabel(f"Effect size ({varname})")
        ax.set_title(f"Association of {varname} and IDPs", fontsize=14, fontweight='bold')

        # p-value legend
        # ----- p-value legend (MUST be added as separate artist) -----
        pv_handles = [
            plt.Line2D([], [], marker='o', markersize=8,
                    markerfacecolor=pfill(0.01), markeredgecolor='k', linestyle='None'),
            plt.Line2D([], [], marker='o', markersize=8,
                    markerfacecolor=pfill(0.07), markeredgecolor='k', linestyle='None'),
            plt.Line2D([], [], marker='o', markersize=8,
                    markerfacecolor=pfill(0.50), markeredgecolor='k', linestyle='None'),
        ]
        pv_labels = ['p < 0.05', '0.05 ≤ p < 0.1', 'p ≥ 0.1 or NA']

        # Create the p-value legend first
        pv_legend = ax.legend(pv_handles, pv_labels,
                            title='p-value',
                            bbox_to_anchor=(1.02, 1.00),
                            loc='upper left',
                            fontsize=9,
                            frameon=True)

        # Force it to stay on the axes (otherwise overwritten by the IDP legend)
        ax.add_artist(pv_legend)

        # IDP legend
        idp_handles = []
        idp_labels = []
        for feat in idp_universe:
            st = stylebank.get(feat)
            idp_handles.append(plt.Line2D([], [], marker=st["marker"], color=st["color"],
                                         markerfacecolor=st["color"], linestyle='None', markersize=8, markeredgecolor='k'))
            idp_labels.append(feat)
        ax.legend(idp_handles, idp_labels, title="IDP", bbox_to_anchor=(1.02, 0.55), loc='upper left', fontsize=8)
        fig.subplots_adjust(right=0.75)

        # interpretation text
        interp = "Interpretation: Depends..as appropriate to the metric and biological variable"
        interp_fs = max(8, mpl.rcParams.get("font.size", 10) - 2)
        fig.subplots_adjust(bottom=0.18)
        fig.text(0.5, 0.04, interp, ha='center', fontsize=interp_fs, color='blue', style='italic')

        outfn = os.path.join(outdir, f"mixed_fixeff_{varname}_combined.png")
        fig.savefig(outfn, dpi=200, bbox_inches='tight')
        plt.close(fig)
        #print("Saved:", outfn)

# ---- runner ----
def main(dirs: List[str], outdir: str, fix_eff: List[str]):
    dirs = [d.rstrip("/") for d in dirs]; ensure_dir(outdir)
    loaded = [(os.path.basename(d.rstrip("/")), load_results_dir(d)) for d in dirs]
    stylebank = StyleBank()
    plot_add_mult_heatmap(loaded, "add_test", outdir)
    plot_add_mult_heatmap(loaded, "mult_test", outdir)
    plot_md_by_site(loaded, outdir, stylebank)
    plot_mixed_single_metrics(loaded, outdir, stylebank)
    plot_pairwise_spearman_combined(loaded, outdir, stylebank)
    plot_wsv_combined(loaded, outdir, stylebank)
    plot_mixed_fixeff(loaded, outdir, stylebank, fix_eff=fix_eff)
    print("Saved combined plots to", outdir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--outdir", default="combined_plots")
    p.add_argument("--fixeff", nargs="*", default=["age","sex"])
    args = p.parse_args()
    main(args.dirs, args.outdir, args.fixeff)
