#!/usr/bin/env python3
"""
run_inspection.py

Full-featured pipeline:
 - static PNG plots (heatmaps, boxplots, scatter with static regressions),
 - per-file methods plots,
 - comparison PNGs,
 - interactive comparison plots (Plotly HTML),
 - summary CSVs and pairwise correlations,
 - HTML report (thumbnails 2-per-row, collapsible sections, links to interactive plots),
 - optional auto-open in browser.

Usage example:
 python run_inspection.py raw.csv harmonised1.csv harmonised2.csv \
   --batch_vars site,scanner \
   --bio_vars age,sex,timepoint,diagnosis \
   --methods 10: \
   --subject_id subjectID \
   --output_dir results \
   --max_subjects 300 \
   --report --open_report

Dependencies:
 pip install pandas numpy matplotlib seaborn scipy plotly statsmodels
"""
import argparse
import os
import sys
import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# interactive
import plotly.express as px
import plotly.io as pio

# for trendline support we rely on statsmodels being installed (plotly uses it for trendline='ols')
# if not installed, plotly will raise an error when using trendline='ols' — install statsmodels when needed.

sns.set(style="whitegrid")


# -------------------- utilities --------------------
def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def basename_noext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def parse_col_spec(spec: Optional[str], df: pd.DataFrame) -> List[str]:
    if spec is None:
        return []
    spec = spec.strip()
    if spec == "":
        return []
    if "," in spec:
        names = [x.strip() for x in spec.split(",") if x.strip()]
        missing = [n for n in names if n not in df.columns]
        if missing:
            print(f"Warning: columns not found in dataframe: {missing}")
            names = [n for n in names if n in df.columns]
        return names
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError("Invalid range spec: should be start:end")
        start = int(parts[0]) if parts[0] != "" else 0
        end = int(parts[1]) if parts[1] != "" else (len(df.columns) - 1)
        return list(df.columns[start: end + 1])
    if spec.isdigit():
        start = int(spec)
        return list(df.columns[start:])
    if spec in df.columns:
        return [spec]
    lowered = {c.lower(): c for c in df.columns}
    if spec.lower() in lowered:
        return [lowered[spec.lower()]]
    print(f"Warning: column spec '{spec}' did not match any columns")
    return []


def load_csvs(paths: List[str]) -> List[pd.DataFrame]:
    dfs = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        df = pd.read_csv(p)
        dfs.append(df)
    return dfs


def get_common_methods(dfs: List[pd.DataFrame], methods_specs: Optional[str]) -> List[str]:
    parsed_per_df = []
    for df in dfs:
        parsed = parse_col_spec(methods_specs, df) if methods_specs else []
        if not parsed:
            parsed = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        parsed_per_df.append(set(parsed))
    common = set.intersection(*parsed_per_df) if len(parsed_per_df) > 1 else parsed_per_df[0]
    common = sorted(list(common))
    return common


# -------------------- plotting (static) --------------------
def plot_heatmap_subjects_by_batch(df: pd.DataFrame, batch_vars: List[str], subject_col: str, out_fn: str):
    if not batch_vars:
        print("No batch_vars supplied for heatmap -- skipping")
        return
    cols = [subject_col] + batch_vars
    sub = df[cols].dropna(subset=[subject_col])
    if len(batch_vars) == 1:
        counts = sub.groupby(batch_vars[0])[subject_col].nunique()
        pivot = counts.to_frame(name='count')
    else:
        pivot = sub.groupby(batch_vars)[subject_col].nunique().unstack(fill_value=0)
    if pivot.size == 0:
        print("No data to plot for heatmap -- skipping")
        return
    plt.figure(figsize=(max(6, pivot.shape[1] * 0.6), max(4, pivot.shape[0] * 0.4)))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Subject counts by {' & '.join(batch_vars)} (primary)")
    plt.tight_layout()
    plt.savefig(out_fn, dpi=150)
    plt.close()
    print(f"Saved heatmap: {out_fn}")


def plot_subject_presence_heatmap(df: pd.DataFrame, batch_vars: List[str], subject_col: str, out_fn: str, max_subjects: int = 200):
    if not batch_vars:
        print("No batch_vars supplied for subject-presence heatmap -- skipping")
        return
    cols = [subject_col] + batch_vars
    sub = df[cols].dropna(subset=[subject_col])
    if len(batch_vars) == 1:
        sub['__batch_combo__'] = sub[batch_vars[0]].astype(str)
    else:
        sub['__batch_combo__'] = sub[batch_vars].astype(str).agg('|'.join, axis=1)
    pivot = pd.crosstab(sub[subject_col], sub['__batch_combo__'])
    if pivot.empty:
        print("No data available to build subject-presence heatmap -- skipping")
        return
    pivot = (pivot > 0).astype(int)
    if pivot.shape[0] > max_subjects:
        top_subjects = pivot.sum(axis=1).sort_values(ascending=False).index[:max_subjects]
        pivot = pivot.loc[top_subjects]
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    width = max(8, pivot.shape[1] * 0.6)
    height = max(6, min(100, pivot.shape[0] * 0.12))
    plt.figure(figsize=(width, height))
    sns.heatmap(pivot, cmap='Greys', cbar=True, linewidths=0.2, linecolor='lightgray')
    plt.xlabel(' | '.join(batch_vars))
    plt.ylabel(subject_col)
    plt.title(f"Subject presence by {' & '.join(batch_vars)} (showing up to {max_subjects} subjects)")
    plt.tight_layout()
    plt.savefig(out_fn, dpi=150)
    plt.close()
    print(f"Saved subject-presence heatmap: {out_fn}")


def plot_boxplots_methods_vs_batch(df: pd.DataFrame, methods: List[str], batch_vars: List[str], out_dir: str, label: str):
    for batch in batch_vars:
        for method in methods:
            if method not in df.columns or batch not in df.columns:
                continue
            sub = df[[method, batch]].dropna()
            if sub.shape[0] < 3:
                continue
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=batch, y=method, data=sub)
            sns.stripplot(x=batch, y=method, data=sub, color='black', alpha=0.3, jitter=True, size=3)
            plt.title(f"{method} by {batch} ({label})")
            plt.xticks(rotation=45)
            plt.tight_layout()
            fn = os.path.join(out_dir, f"box_{method}_by_{batch}_{label}.png")
            plt.savefig(fn, dpi=150)
            plt.close()
    print(f"Saved boxplots for {label} in {out_dir}")


def plot_scatter_methods_vs_numeric_bio(df: pd.DataFrame, methods: List[str], numeric_bio: List[str], out_dir: str, label: str):
    for bio in numeric_bio:
        for method in methods:
            if method not in df.columns or bio not in df.columns:
                continue
            sub = df[[method, bio]].dropna()
            sub = sub[pd.to_numeric(sub[method], errors='coerce').notnull() & pd.to_numeric(sub[bio], errors='coerce').notnull()]
            if sub.shape[0] < 3:
                continue
            x = pd.to_numeric(sub[bio])
            y = pd.to_numeric(sub[method])
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            plt.figure(figsize=(6, 5))
            sns.regplot(x=x, y=y, scatter_kws={'s':20, 'alpha':0.6})
            plt.xlabel(bio)
            plt.ylabel(method)
            plt.title(f"{method} vs {bio} (r={r_value:.3f}, p={p_value:.3g}) [{label}]")
            # remove the eq annotation if you prefer a cleaner plot
            # eq = f"y={slope:.3g}x+{intercept:.3g}\\nr={r_value:.3f}, p={p_value:.3g}"
            # plt.annotate(eq, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=9,
            #              ha='left', va='top', bbox=dict(boxstyle=\"round,pad=0.3\", fc=\"white\", ec=\"black\", alpha=0.6))
            plt.tight_layout()
            fn = os.path.join(out_dir, f"scatter_{method}_vs_{bio}_{label}.png")
            plt.savefig(fn, dpi=150)
            plt.close()
    print(f"Saved scatter plots for {label} in {out_dir}")


def plot_boxplot_numeric_bio_vs_batch(df: pd.DataFrame, numeric_bio: List[str], batch_vars: List[str], out_dir: str):
    for bio in numeric_bio:
        for batch in batch_vars:
            if bio not in df.columns or batch not in df.columns:
                continue
            sub = df[[bio, batch]].dropna()
            sub[bio] = pd.to_numeric(sub[bio], errors='coerce')
            sub = sub[sub[bio].notnull()]
            if sub.shape[0] < 3:
                continue
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=batch, y=bio, data=sub)
            sns.stripplot(x=batch, y=bio, data=sub, color='black', alpha=0.3, jitter=True, size=3)
            plt.title(f"{bio} by {batch} (primary)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            fn = os.path.join(out_dir, f"box_{bio}_by_{batch}_primary.png")
            plt.savefig(fn, dpi=150)
            plt.close()
    print(f"Saved numeric bio boxplots in {out_dir}")


def plot_categorical_bio_heatmaps(df: pd.DataFrame, cat_bio: List[str], batch_vars: List[str], subject_col: str, out_dir: str):
    for bio in cat_bio:
        for batch in batch_vars:
            if bio not in df.columns or batch not in df.columns:
                continue
            sub = df[[subject_col, bio, batch]].dropna(subset=[subject_col])
            pivot = sub.groupby([bio, batch])[subject_col].nunique().unstack(fill_value=0)
            if pivot.size == 0:
                continue
            plt.figure(figsize=(max(6, pivot.shape[1] * 0.6), max(4, pivot.shape[0] * 0.4)))
            sns.heatmap(pivot, annot=True, fmt='d', cmap='OrRd')
            plt.title(f"Subject counts by {bio} and {batch} (primary)")
            plt.tight_layout()
            fn = os.path.join(out_dir, f"heat_{bio}_by_{batch}_primary.png")
            plt.savefig(fn, dpi=150)
            plt.close()
    print(f"Saved categorical bio heatmaps in {out_dir}")


# -------------------- summaries & correlations --------------------
from scipy.stats import pearsonr


def compute_summary_stats_for_df(df: pd.DataFrame, methods: List[str], batch_vars: List[str], subject_id: str):
    rows = []
    if batch_vars:
        if len(batch_vars) == 1:
            df = df.copy()
            df['_batch_combo_'] = df[batch_vars[0]].astype(str)
        else:
            df = df.copy()
            df['_batch_combo_'] = df[batch_vars].astype(str).agg('|'.join, axis=1)
    else:
        df = df.copy()
        df['_batch_combo_'] = 'all'
    for batch_val, grp in df.groupby('_batch_combo_'):
        for method in methods:
            if method not in grp.columns:
                continue
            vals = pd.to_numeric(grp[method], errors='coerce').dropna()
            rows.append({
                'batch': batch_val,
                'method': method,
                'mean': float(vals.mean()) if len(vals) else None,
                'std': float(vals.std()) if len(vals) else None,
                'n': int(len(vals))
            })
    return pd.DataFrame(rows)


def write_summaries_and_collect(paths: List[str], dfs: List[pd.DataFrame], methods: List[str], batch_vars: List[str], subject_id: str, outdir: str):
    ensure_output_dir(outdir)
    summary_files = []
    all_summaries = []
    for path, df in zip(paths, dfs):
        base = basename_noext(path)
        summ = compute_summary_stats_for_df(df, methods, batch_vars, subject_id)
        summ['file'] = base
        out_csv = os.path.join(outdir, f"summary_{base}.csv")
        summ.to_csv(out_csv, index=False)
        summary_files.append(out_csv)
        all_summaries.append(summ)
    combined = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    combined.to_csv(os.path.join(outdir, "summary_combined.csv"), index=False)
    return summary_files, os.path.join(outdir, "summary_combined.csv"), combined


def compute_pairwise_method_correlations(primary_df: pd.DataFrame, other_df: pd.DataFrame, methods: List[str], subject_id: str):
    if subject_id not in primary_df.columns or subject_id not in other_df.columns:
        return pd.DataFrame()
    p1 = primary_df.set_index(subject_id)
    p2 = other_df.set_index(subject_id)
    common_subjects = p1.index.intersection(p2.index)
    rows = []
    for method in methods:
        if method not in p1.columns or method not in p2.columns:
            continue
        x = pd.to_numeric(p1.loc[common_subjects, method], errors='coerce')
        y = pd.to_numeric(p2.loc[common_subjects, method], errors='coerce')
        valid = x.notnull() & y.notnull()
        if valid.sum() < 3:
            rows.append({'method': method, 'r': None, 'p': None, 'n_pairs': int(valid.sum())})
            continue
        r, pval = pearsonr(x[valid], y[valid])
        rows.append({'method': method, 'r': float(r), 'p': float(pval), 'n_pairs': int(valid.sum())})
    return pd.DataFrame(rows)


def write_pairwise_correlations(paths: List[str], dfs: List[pd.DataFrame], methods: List[str], subject_id: str, outdir: str, primary_index: int = 0):
    ensure_output_dir(outdir)
    primary_df = dfs[primary_index]
    corr_files = []
    for i, (path, df) in enumerate(zip(paths, dfs)):
        if i == primary_index:
            continue
        base = f"{basename_noext(paths[primary_index])}_vs_{basename_noext(path)}"
        corr_df = compute_pairwise_method_correlations(primary_df, df, methods, subject_id)
        out_csv = os.path.join(outdir, f"correlations_{base}.csv")
        corr_df.to_csv(out_csv, index=False)
        corr_files.append(out_csv)
    return corr_files


# -------------------- static comparison plots (PNG) --------------------
def make_comparison_plots(dfs: List[pd.DataFrame], labels: List[str], methods: List[str], batch_vars: List[str], numeric_bio: List[str], out_dir: str):
    ensure_output_dir(out_dir)
    small_dfs = []
    for df, lab in zip(dfs, labels):
        cols = list(set(methods + batch_vars + numeric_bio))
        available = [c for c in cols if c in df.columns]
        if not available:
            continue
        sub = df[available].copy()
        sub['__source__'] = lab
        small_dfs.append(sub)
    if not small_dfs:
        print("No overlapping columns for comparison plots")
        return
    comb = pd.concat(small_dfs, ignore_index=True, sort=False)

    # boxplots
    for batch in batch_vars:
        for method in methods:
            if method not in comb.columns or batch not in comb.columns:
                continue
            sub = comb[[method, batch, '__source__']].dropna()
            if sub.shape[0] < 3:
                continue
            plt.figure(figsize=(9, 5))
            sns.boxplot(x=batch, y=method, hue='__source__', data=sub)
            plt.title(f"{method} by {batch} (comparison)")
            plt.xticks(rotation=45)
            plt.legend(title='source')
            plt.tight_layout()
            fn = os.path.join(out_dir, f"comp_box_{method}_by_{batch}.png")
            plt.savefig(fn, dpi=150)
            plt.close()

    # scatter PNGs
    for bio in numeric_bio:
        for method in methods:
            if method not in comb.columns or bio not in comb.columns:
                continue
            sub = comb[[method, bio, '__source__']].dropna()
            if sub.shape[0] < 6:
                continue
            plt.figure(figsize=(7, 5))
            sns.scatterplot(x=bio, y=method, hue='__source__', data=sub, alpha=0.6)
            for lab in sub['__source__'].unique():
                ssub = sub[sub['__source__'] == lab]
                if ssub.shape[0] < 3:
                    continue
                x = pd.to_numeric(ssub[bio])
                y = pd.to_numeric(ssub[method])
                try:
                    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
                    xs = np.linspace(x.min(), x.max(), 100)
                    plt.plot(xs, slope * xs + intercept, label=f"{lab} fit: r={r_value:.2f}, p={p_value:.2g}")
                except Exception:
                    pass
            plt.title(f"{method} vs {bio} (comparison)")
            plt.legend()
            plt.tight_layout()
            fn = os.path.join(out_dir, f"comp_scatter_{method}_vs_{bio}.png")
            plt.savefig(fn, dpi=150)
            plt.close()
    print(f"Saved comparison plots in {out_dir}")


# -------------------- interactive comparison plots (Plotly) --------------------
def make_interactive_comparison_plots(dfs: List[pd.DataFrame], labels: List[str], methods: List[str], batch_vars: List[str], numeric_bio: List[str], out_dir: str):
    """
    Creates interactive plotly HTMLs in out_dir:
     - boxplots for each method x batch (color=source)
     - scatterplots for method vs numeric bio (color=source + trendline='ols')
    """
    ensure_output_dir(out_dir)
    # assemble combined df for plotting
    small_dfs = []
    for df, lab in zip(dfs, labels):
        cols = list(set(methods + batch_vars + numeric_bio))
        available = [c for c in cols if c in df.columns]
        if not available:
            continue
        sub = df[available].copy()
        sub['__source__'] = lab
        small_dfs.append(sub)
    if not small_dfs:
        print("No overlapping columns for interactive comparison plots")
        return
    comb = pd.concat(small_dfs, ignore_index=True, sort=False)

    # boxplots (interactive)
    for batch in batch_vars:
        for method in methods:
            if method not in comb.columns or batch not in comb.columns:
                continue
            sub = comb[[method, batch, '__source__']].dropna()
            if sub.shape[0] < 3:
                continue
            # use plotly.express box
            fig = px.box(sub, x=batch, y=method, color='__source__', points='all', title=f"{method} by {batch} (interactive comparison)")
            fname = os.path.join(out_dir, f"interactive_box_{method}_by_{batch}.html")
            pio.write_html(fig, file=fname, auto_open=False, include_plotlyjs='cdn')
    # scatterplots (interactive) with trendline
    for bio in numeric_bio:
        for method in methods:
            if method not in comb.columns or bio not in comb.columns:
                continue
            sub = comb[[method, bio, '__source__']].dropna()
            if sub.shape[0] < 6:
                continue
            # px.scatter with trendline='ols' uses statsmodels; ensure it's installed
            try:
                fig = px.scatter(sub, x=bio, y=method, color='__source__', trendline='ols',
                                 title=f"{method} vs {bio} (interactive comparison)")
                fname = os.path.join(out_dir, f"interactive_scatter_{method}_vs_{bio}.html")
                pio.write_html(fig, file=fname, auto_open=False, include_plotlyjs='cdn')
            except Exception as e:
                # fallback: simple scatter without trendline
                fig = px.scatter(sub, x=bio, y=method, color='__source__', title=f"{method} vs {bio} (interactive comparison, no trendline)")
                fname = os.path.join(out_dir, f"interactive_scatter_{method}_vs_{bio}.html")
                pio.write_html(fig, file=fname, auto_open=False, include_plotlyjs='cdn')
    print(f"Saved interactive comparison HTMLs in {out_dir}")


# -------------------- HTML report builder --------------------
def relpath_from(base: str, path: str) -> str:
    try:
        return os.path.relpath(path, base)
    except Exception:
        return path


def generate_html_report(plots_dir: str,
                         csv_paths: List[str],
                         methods: List[str],
                         batch_vars: List[str],
                         bio_vars: List[str],
                         subject_id: str,
                         output_dir: str,
                         main_df: pd.DataFrame,
                         labels: List[str],
                         out_fn: str,
                         embed_images: bool = False):
    """
    HTML report builder with:
     - 2-column thumbnail grid
     - collapsible sections
     - links to interactive HTMLs
     - embedded preview tables for summaries/correlations
    If embed_images True -> embeds PNGs as base64 (single file portability)
    """
    total_subjects = int(main_df[subject_id].nunique()) if subject_id in main_df.columns else 'N/A'
    generated_at = datetime.datetime.now().isoformat()

    # collect images and interactive htmls by folder
    sections = {}
    interactive = []
    for root, dirs, files in os.walk(plots_dir):
        relroot = os.path.relpath(root, plots_dir)
        imgs = [os.path.join(root, f) for f in sorted(files) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.svg'))]
        htmls = [os.path.join(root, f) for f in sorted(files) if f.lower().endswith('.html')]
        if imgs:
            sections[relroot] = imgs
        if htmls:
            interactive.extend(htmls)

    # header / styles / JS for collapsible sections
    html_lines = []
    html_lines.append('<!doctype html>')
    html_lines.append('<html lang="en"><head><meta charset="utf-8"><title>Harmonization diagnostics report</title>')
    html_lines.append('<style>'
                      'body{font-family:Arial,Helvetica,sans-serif;margin:20px} '
                      'h1,h2{color:#222} '
                      'img{max-width:100%;height:auto;border:1px solid #ddd;padding:4px;background:#fff;margin-bottom:6px} '
                      '.meta{background:#f7f7f7;padding:10px;border-radius:6px;margin-bottom:16px} '
                      '.section{margin-bottom:30px} '
                      '.grid{display:flex;flex-wrap:wrap;gap:12px} '
                      '.thumb{flex: 0 0 calc(50% - 12px); box-sizing:border-box; background:#fafafa; padding:6px; border-radius:4px; text-align:center;} '
                      '.thumb .caption{font-size:12px; margin-top:6px; color:#444; word-break:break-all;} '
                      '@media(max-width:800px){ .thumb{flex:0 0 100%} }'
                      '.togglebtn{cursor:pointer;color:#0a66c2;text-decoration:underline;margin-bottom:8px}'
                      '</style>')
    # small JS for toggles
    html_lines.append('<script>function toggle(id){var e=document.getElementById(id); if(!e) return; e.style.display = (e.style.display==\"none\")?\"block\":\"none\";}</script>')
    html_lines.append('</head><body>')
    html_lines.append(f'<h1>Harmonization diagnostics report</h1>')
    html_lines.append(f'<div class="meta"><strong>Generated:</strong> {generated_at}<br/>')
    html_lines.append(f'<strong>Primary CSV:</strong> {csv_paths[0]}<br/>')
    if len(csv_paths) > 1:
        html_lines.append(f'<strong>Other CSVs:</strong> {", ".join(csv_paths[1:])}<br/>')
    html_lines.append(f'<strong>Total unique subjects (primary):</strong> {total_subjects}<br/>')
    html_lines.append(f'<strong>Output dir:</strong> {output_dir}<br/>')
    html_lines.append(f'<strong>Methods (count):</strong> {len(methods)}<br/>')
    html_lines.append(f'<strong>Methods names (sample):</strong> {", ".join(methods[:20])}{"..." if len(methods)>20 else ""}<br/>')
    html_lines.append(f'<strong>Batch vars:</strong> {batch_vars}<br/>')
    html_lines.append(f'<strong>Bio vars requested:</strong> {bio_vars}<br/>')
    html_lines.append('</div>')

    # include summaries & correlations lists (if present)
    summary_dir = os.path.join(output_dir, 'summaries')
    if os.path.exists(summary_dir):
        html_lines.append('<div class="section"><h2 onclick="toggle(\'sec_summ\')" class="togglebtn">Summary statistics (click to expand)</h2>')
        html_lines.append('<div id="sec_summ" style="display:block">')
        for f in sorted(os.listdir(summary_dir)):
            if f.endswith('.csv'):
                rel = relpath_from(output_dir, os.path.join(summary_dir, f))
                html_lines.append(f'<div><a href="{rel}">{f}</a></div>')
        combined_csv = os.path.join(summary_dir, 'summary_combined.csv')
        if os.path.exists(combined_csv):
            try:
                df_preview = pd.read_csv(combined_csv, nrows=50)
                html_lines.append('<div style="max-height:300px;overflow:auto;border:1px solid #eee;padding:6px;margin-top:6px">')
                html_lines.append(df_preview.to_html(index=False, classes="preview_table"))
                html_lines.append('</div>')
            except Exception:
                pass
        html_lines.append('</div></div>')

    corr_dir = os.path.join(output_dir, 'correlations')
    if os.path.exists(corr_dir):
        html_lines.append('<div class="section"><h2 onclick="toggle(\'sec_corr\')" class="togglebtn">Per-method correlations (primary vs others) (click to expand)</h2>')
        html_lines.append('<div id="sec_corr" style="display:block">')
        for f in sorted(os.listdir(corr_dir)):
            if f.endswith('.csv'):
                rel = relpath_from(output_dir, os.path.join(corr_dir, f))
                html_lines.append(f'<div><a href="{rel}">{f}</a></div>')
        html_lines.append('</div></div>')

    # ordered sections: root, perfile_<lab>..., comparison
    order = []
    if '.' in sections:
        order.append('.')
    for lab in labels:
        key = 'perfile_' + lab
        if key in sections:
            order.append(key)
    if 'comparison' in sections:
        order.append('comparison')
    for k in sorted(sections.keys()):
        if k not in order:
            order.append(k)

    sec_id = 0
    for sec in order:
        imgs = sections.get(sec, [])
        if not imgs:
            continue
        display_name = sec if sec != '.' else 'primary (root plots)'
        sec_id += 1
        sid = f"sec_{sec_id}"
        html_lines.append(f'<div class="section"><h2 onclick="toggle(\'{sid}\')" class="togglebtn">{display_name} (click to expand)</h2>')
        html_lines.append(f'<div id="{sid}" style="display:block">')
        html_lines.append('<div class="grid">')
        for img in imgs:
            r = relpath_from(output_dir, img)
            # if embedding requested, we would convert to base64 here (not doing by default)
            html_lines.append(
                f'<div class="thumb"><a href="{r}" target="_blank"><img src="{r}" alt="{os.path.basename(img)}"/></a>'
                f'<div class="caption">{os.path.basename(img)}</div></div>'
            )
        html_lines.append('</div></div></div>')

    # interactive html links section
    if interactive:
        html_lines.append('<div class="section"><h2 onclick="toggle(\'sec_inter\')" class="togglebtn">Interactive comparison plots (click to expand)</h2>')
        html_lines.append('<div id="sec_inter" style="display:block">')
        for h in sorted(interactive):
            rel = relpath_from(output_dir, h)
            html_lines.append(f'<div><a href="{rel}" target="_blank">{os.path.basename(h)}</a></div>')
        html_lines.append('</div></div>')

    html_lines.append('</body></html>')

    # write with real newlines
    with open(out_fn, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(html_lines))

    print(f"HTML report written to: {out_fn}")


# -------------------- main --------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="Generate QC + interactive comparison report from CSV(s)")
    p.add_argument('csvs', nargs='+', help='One or more CSV files. First is primary/main CSV.')
    p.add_argument('--batch_vars', default=None, help="Spec for batch variables (comma names or range)")
    p.add_argument('--bio_vars', default=None, help="Spec for bio variables (comma names or range)")
    p.add_argument('--features', default=None, help="Spec for features/idps columns (comma, range, index). If multiple csvs supplied, common columns are used.")
    p.add_argument('--subject_id', default='subjectID', help='Column name for subject identifier (default: subjectID)')
    p.add_argument('--output_dir', default='qc_plots', help='Directory to save plots and report')
    p.add_argument('--max_subjects', type=int, default=300, help='Max subjects to show in subject-presence heatmap')
    p.add_argument('--report', action='store_true', help='Generate HTML report')
    p.add_argument('--open_report', action='store_true', help='Auto-open HTML report in browser (if generated)')
    p.add_argument('--embed_images', action='store_true', help='Embed PNGs into single HTML as base64 (large file)')
    args = p.parse_args(argv)

    dfs = load_csvs(args.csvs)
    labels = [basename_noext(p) for p in args.csvs]
    main_df = dfs[0]

    ensure_output_dir(args.output_dir)
    plots_dir = os.path.join(args.output_dir, 'plots')
    ensure_output_dir(plots_dir)

    batch_vars = parse_col_spec(args.batch_vars, main_df)
    bio_vars = parse_col_spec(args.bio_vars, main_df)

    methods = get_common_methods(dfs, args.features)
    if not methods:
        print("No methods/idps columns identified. Exiting.")
        return
    print(f"Using methods/idps columns (count={len(methods)}): {methods[:10]}{'...' if len(methods)>10 else ''}")

    # descriptive plots (primary only)
    plot_heatmap_subjects_by_batch(main_df, batch_vars, args.subject_id, os.path.join(plots_dir, 'heatmap_subjects_by_batch_primary.png'))
    plot_subject_presence_heatmap(main_df, batch_vars, args.subject_id, os.path.join(plots_dir, 'heatmap_subject_presence_primary.png'), max_subjects=args.max_subjects)

    numeric_bio, cat_bio = split_bio_vars_into_numeric_and_categorical = (lambda df, bio_vars: (
        [b for b in bio_vars if b in df.columns and pd.to_numeric(df[b], errors='coerce').notnull().sum()/max(1,len(df[b].dropna()))>=0.6],
        [b for b in bio_vars if b in df.columns and pd.to_numeric(df[b], errors='coerce').notnull().sum()/max(1,len(df[b].dropna()))<0.6]
    ))(main_df, bio_vars)

    # numeric_bio, cat_bio = split_bio_vars_into_numeric_and_categorical(main_df, bio_vars)
    # produce static numeric/categorical bio plots
    plot_boxplot_numeric_bio_vs_batch(main_df, numeric_bio, batch_vars, plots_dir)
    plot_categorical_bio_heatmaps(main_df, cat_bio, batch_vars, args.subject_id, plots_dir)

    # methods-related plots for each CSV individually (perfile)
    for df, lab in zip(dfs, labels):
        lab_dir = os.path.join(plots_dir, f"perfile_{lab}")
        ensure_output_dir(lab_dir)
        plot_boxplots_methods_vs_batch(df, methods, batch_vars, lab_dir, lab)
        plot_scatter_methods_vs_numeric_bio(df, methods, numeric_bio, lab_dir, lab)

    # comparison static PNGs
    comp_dir = os.path.join(plots_dir, 'comparison')
    ensure_output_dir(comp_dir)
    make_comparison_plots(dfs, labels, methods, batch_vars, numeric_bio, comp_dir)

    # interactive comparison (Plotly HTMLs)
    inter_dir = os.path.join(plots_dir, 'comparison_interactive')
    ensure_output_dir(inter_dir)
    make_interactive_comparison_plots(dfs, labels, methods, batch_vars, numeric_bio, inter_dir)

    # summaries & correlations
    summaries_dir = os.path.join(args.output_dir, 'summaries')
    corr_dir = os.path.join(args.output_dir, 'correlations')
    ensure_output_dir(summaries_dir)
    ensure_output_dir(corr_dir)
    _, _, combined_df = write_summaries_and_collect(args.csvs, dfs, methods, batch_vars, args.subject_id, summaries_dir)
    _ = write_pairwise_correlations(args.csvs, dfs, methods, args.subject_id, corr_dir)

    # generate HTML report (optional)
    if args.report:
        report_fn = os.path.join(args.output_dir, 'inspection_report.html')
        generate_html_report(plots_dir, args.csvs, methods, batch_vars, bio_vars, args.subject_id, args.output_dir, main_df, labels, report_fn, embed_images=args.embed_images)
        if args.open_report:
            try:
                import webbrowser
                webbrowser.open('file://' + os.path.abspath(report_fn))
            except Exception:
                print("Unable to auto-open report in browser; you can open it manually:", report_fn)

    print(f"All outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
