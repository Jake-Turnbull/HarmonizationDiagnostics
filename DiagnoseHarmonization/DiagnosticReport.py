#!/usr/bin/env python3
"""
DiagnosticReport.py

Build HTML report with large inline PNGs (no interactive pages).

Usage:
python DiagnosticReport.py --dir combined_plots --outdir combined_plots/report --fix_eff age sex

Dependencies:
    pip install pandas pillow numpy
"""
from __future__ import annotations
import os
import argparse
from pathlib import Path
from typing import List, Dict
from PIL import Image
import numpy as np

# ---------------------------------------------------------------------------
# Order of PNG plots to show inline (keeps the order you requested)
# ---------------------------------------------------------------------------
PLOTS_ORDER = [
    ("wsv_boxplots_combined.png", "Within-subject variability"),
    ("pairwise_spearman_idpavg_combined.png", "Subject order consistency (IDP-avg)"),
    ("pairwise_spearman_timepairavg_combined.png", "Subject order consistency (timepair-avg)"),
    ("mixed_n_is_batchSig_combined.png", "Pairwise site significance"),
    ("add_test_pvalues_combined.png", "Additive batch effect (p-values)"),
    ("mult_test_pvalues_combined.png", "Multiplicative batch effect (p-values)"),
    ("md_by_site_combined.png", "Multivariate metric (Mahalanobis distance by site)"),
    ("mixed_ICC_combined.png", "Cross-subject variability (ICC)"),
    ("mixed_WCV_combined.png", "Cross-subject variability (WCV)"),
    ("mixed_fixeff_{fix}_combined.png", "Biological variability: {fix}"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def find_plots(indir: Path, fix_eff: List[str]) -> List[Dict]:
    """Return list of found PNG plot items in required order (skip missing)."""
    results = []
    for name, title in PLOTS_ORDER:
        if "{fix}" in name:
            for fix in fix_eff:
                fn = name.format(fix=fix)
                p = indir / fn
                if p.exists():
                    results.append({"path": p, "filename": fn, "title": title.format(fix=fix), "key": f"fix_{fix}"})
        else:
            p = indir / name
            if p.exists():
                key = Path(name).stem
                results.append({"path": p, "filename": name, "title": title, "key": key})
    return results

def make_thumbnail(src_path: Path, dest_path: Path, maxsize=(1200,1000)):
    im = Image.open(src_path)
    im.thumbnail(maxsize, Image.LANCZOS)
    im.save(dest_path, format="PNG")

# ---------------------------------------------------------------------------
# HTML report assembly (PNG inline)
# ---------------------------------------------------------------------------
HTML_HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><title>Harmonization Diagnostics Report</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
 body { font-family: Arial, Helvetica, sans-serif; margin: 20px; color: #222; }
 h1 { font-size: 22px; margin-bottom:8px; }
 h2 { font-size: 18px; margin-top: 28px; }
 .plot-card { border: 0px; padding: 10px; margin-bottom: 26px; display:block; }
 .plot-img { display:block; margin: 0 auto 8px auto; max-width: 90%; height: auto; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-radius:6px; }
 .plot-meta { max-width: 100%; margin-top: 6px; text-align: center; }
 .meta-title { font-weight:600; font-size:16px; margin-bottom:6px; text-align:center; }
 .download-btn { display:inline-block; background:#1976d2; color:white; padding:6px 10px; text-decoration:none; border-radius:4px; margin:6px 6px 0 6px; }
 .small { font-size: 12px; color:#666; }
 footer { margin-top:32px; font-size:12px; color:#666; }
</style>

<!-- JS helper to force image download by fetching the file as a blob and programmatically downloading it -->
<script type="text/javascript">
async function downloadImage(url) {
    try {
        // fetch the image as blob
        const resp = await fetch(url, {cache: "no-cache"});
        if (!resp.ok) throw new Error('Network response not OK');
        const blob = await resp.blob();
        // build a temporary object URL and force download
        const objUrl = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = objUrl;
        // prefer the filename from the URL
        const parts = url.split('/');
        a.download = parts[parts.length - 1] || 'image.png';
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(objUrl);
    } catch (err) {
        // fallback: open in new tab if download fails
        window.open(url, '_blank', 'noopener');
    }
    return false;
}
</script>

</head>
<body>
<h1>Harmonisation Evaluation Report</h1>
"""


HTML_FOOT = "</body></html>"

def build_report(indir: str, outdir: str, fix_eff: List[str], inputs: List[str]):
    indir_p = Path(indir).expanduser().resolve()
    outdir_p = Path(outdir).expanduser().resolve()
    ensure_dir(outdir_p)
    thumbs_dir = outdir_p / "thumbnails"
    ensure_dir(thumbs_dir)

    plots = find_plots(indir_p, fix_eff)

    # prepare thumbnails and (PNG-derived) copies
    for item in plots:
        src = item["path"]
        thumb = thumbs_dir / item["filename"]
        try:
            make_thumbnail(src, thumb)
        except Exception:
            import shutil
            shutil.copy(src, thumb)

    # Build HTML report
    html_path = outdir_p / "report.html"
    with open(html_path, "w", encoding="utf8") as f:
        f.write(HTML_HEAD)
        # Inputs listing
        f.write("<h2>Inputs</h2>\n")
        if inputs:
            f.write("<div class='small'><strong>Input directories/files:</strong><ul>\n")
            for inp in inputs:
                f.write(f"<li>{inp}</li>\n")
            f.write("</ul></div>\n")
        else:
            f.write(f"<div class='small'><strong>Scanned directory:</strong> {indir_p}</div>\n")

        # Quick links
        f.write("<div class='toplinks'><strong>Quick links:</strong><br>\n")
        anchor_links = []
        for item in plots:
            anchor = item["key"]
            anchor_links.append(f"<a href='#{anchor}'>{item['title']}</a>")
        f.write("<br>\n".join(anchor_links))
        f.write("</div>\n<hr/>\n")

        # Insert PNGs inline
        for item in plots:
            key = item["key"]; title = item["title"]
            file_rel = os.path.relpath(item["path"], start=outdir_p)
            f.write(f"<a id='{key}'></a>\n")
            f.write("<div class='plot-card'>\n")
            f.write(f"  <img src='{file_rel}' class='plot-img' alt='{title}'/>\n")
            f.write("  <div class='plot-meta'>\n")
            f.write(f"    <div class='meta-title'>{title}</div>\n")
            f.write("    <div class='small' style='color:blue; font-style:italic; margin-bottom:12px;'>")
            # interpretation text (center-justified, smaller)
            """ if key.startswith("wsv_boxplots"):
                f.write("Lower values indicate reduction of within-subject variability across sites or timepoints.")
            elif key.startswith("pairwise_spearman_idpavg") or key.startswith("pairwise_spearman_timepairavg"):
                f.write("Higher values indicate better preservation of subject order across sites or timepoints.")
            elif key.startswith("mixed_n_is_batchSig"):
                f.write("Lower values are better.")
            elif key.startswith("add_test_pvalues"):
                f.write("p<0.05* indicate presence of significant additive batch effects.")
            elif key.startswith("mult_test_pvalues"):
                f.write("p<0.05* indicate presence of significant multiplicative batch effects.")
            elif key.startswith("md_by_site"):
                f.write("Lower values are better.")
            elif key.startswith("mixed_ICC"):
                f.write("Values close to 1 are better.")
            elif key.startswith("mixed_WCV"):
                f.write("Lower values are better.")
            elif key.startswith("fix_"):
                f.write("Depends upon the context of the study.")
            else:
                f.write("Interpret as appropriate for the metric.") """
            f.write("</div>\n")

            # bottom buttons
            f.write("    <div style='margin-top:4px;'>\n")
            # Use JS helper to force download (fetch + blob). Use relative path as before.
            #f.write(f"      <a class='download-btn' href='#' onclick=\"downloadImage('{file_rel}'); return false;\">Download PNG</a>\n")
            # Keep "Open full" to open the image in a new tab for inspection
            f.write(f"      <a class='download-btn' href='{file_rel}' target='_blank' style='background:#444;color:white;'>Open full</a>\n")
            f.write("    </div>\n")
            f.write("    </div>\n")
            f.write("  </div>\n")
            f.write("</div>\n<hr/>\n")

        f.write(HTML_FOOT)

    print("Report written to:", html_path)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Create an HTML report aggregating combined plots (PNGs only).")
    p.add_argument("--dir", required=True, help="Directory containing combined plots (PNG files).")
    p.add_argument("--outdir", required=True, help="Output directory where report will be written.")
    p.add_argument("--fixeff", nargs="*", default=[], help="Fixed-effect names (e.g. age sex) used to find mixed_fixeff plots.")
    p.add_argument("--inputs", nargs="*", default=[], help="Optional: list of input directories/files to show in the report header.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_report(args.dir, args.outdir, args.fixeff, args.inputs)
