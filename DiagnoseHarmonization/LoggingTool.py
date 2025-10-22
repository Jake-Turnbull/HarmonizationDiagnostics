import os
import io
import base64
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt  # for typing; you can create figures however you like

class _MemoryLogHandler(logging.Handler):
    """Collects formatted log records in memory (for embedding in HTML)."""
    def __init__(self, level=logging.INFO):
        super().__init__(level)
        self.records: List[str] = []

    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        self.records.append(msg)

    def get_text(self) -> str:
        return "\n".join(self.records)

class StatsReporter:
    """
    Usage:
        with StatsReporter(save_dir="results", save_artifacts=True) as rep:
            rep.logger.info("Running t-test on group A vs B...")
            rep.log_summary("t-test", {"statistic": 2.13, "p": 0.034})
            fig = make_my_plot()  # returns a matplotlib.figure.Figure
            rep.log_plot(fig, "QQ plot of residuals")

    - Default: emits a single HTML report with inline (base64) plots.
    - If save_artifacts=True: also saves PNGs for each plot and a plaintext log file.
    """
    def __init__(
        self,
        save_dir: Optional[os.PathLike] = None,
        report_name: str = "report.html",
        plaintext_log_name: str = "run.log",
        save_artifacts: bool = False,
        logger_name: str = "statsreporter",
        log_level: int = logging.INFO,
    ):
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.report_name = report_name
        self.plaintext_log_name = plaintext_log_name
        self.save_artifacts = save_artifacts

        # Prepare containers
        self._plots: List[Tuple[str, str]] = []  # (caption, base64_png)
        self._artifacts_paths: List[Path] = []

        # Create logger
        self.logger = logging.getLogger(logger_name + f".{id(self)}")
        self.logger.setLevel(log_level)
        self.logger.propagate = False  # avoid duplicate output

        # Formatter
        self._fmt = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")

        # In-memory handler for HTML embedding
        self._mem_handler = _MemoryLogHandler()
        self._mem_handler.setFormatter(self._fmt)
        self.logger.addHandler(self._mem_handler)

        # Optional plaintext file handler (only when saving artifacts)
        self._file_handler: Optional[logging.Handler] = None
        if self.save_artifacts and self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            logfile = self.save_dir / self.plaintext_log_name
            self._file_handler = logging.FileHandler(logfile, encoding="utf-8")
            self._file_handler.setLevel(log_level)
            self._file_handler.setFormatter(self._fmt)
            self.logger.addHandler(self._file_handler)
            self._artifacts_paths.append(logfile)
            
    def log_summary(self, test_name: str, summary: dict):
        """
        Write a one-line header + pretty key/val summary via logger.
        """
        self.logger.info(f"[SUMMARY] {test_name}")
        # format dict consistently:
        for k, v in summary.items():
            self.logger.info(f"  - {k}: {v}")

    def log_text(self, message: str, level: int = logging.INFO):
        """Free-form text to the log."""
        self.logger.log(level, message)

    def text_simple(self, message: str, level=logging.INFO):
        """Log without date and time to reduce clutter """
        fmt_backup = self._mem_handler.formatter
        self._mem_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.log(level, message)
        self._mem_handler.setFormatter(fmt_backup)


    def log_plot(self, fig, caption: str = "Plot"):
        """
        Accepts a matplotlib Figure, embeds it into the HTML report,
        and (optionally) saves it as a PNG in save_dir when save_artifacts=True.
        """
        # Save to a bytes buffer (PNG) and base64-encode for inline HTML
        buf = io.BytesIO()
        fig.tight_layout() # Check as throws warning with some figures...
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        self._plots.append((caption, b64))

        # Save PNG of figure to same directory if save_artifacts==true
        if self.save_artifacts and self.save_dir is not None:
            fname = self._unique_png_name(caption)
            outpath = self.save_dir / fname
            fig.savefig(outpath, format="png", dpi=150, bbox_inches="tight")
            self._artifacts_paths.append(outpath)
            self.logger.info(f"[ARTIFACT] Saved plot: {outpath}")

    def write_report(self, path: Optional[os.PathLike] = None) -> Path:
        """
        Write (or rewrite) the HTML report and return its path.
        """
        if path is None:
            path = self.report_path
        path = Path(path)
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)

        html = self._render_html(
            title="Statistical Tests Report",
            log_text=self._mem_handler.get_text(),
            plots=self._plots,
        )
        path.write_text(html, encoding="utf-8")
        return path

    @property
    def report_path(self) -> Path:
        return (self.save_dir or Path.cwd()) / self.report_name

    @property
    def artifacts(self) -> List[Path]:
        """List of any files saved in addition to the HTML report."""
        return list(self._artifacts_paths)

    # ---- context manager ---------------------------------------------------

    def __enter__(self):
        self.logger.info("=== Report session started ===")
        self.logger.info(f"save_artifacts={self.save_artifacts} | save_dir={self.save_dir}")
        return self

    # Exit messages defined seperately from function, keep to logging tool (maybe rethink later?)
    def __exit__(self, exc_type, exc, tb):
        if exc:
            self.logger.exception("Exception during reporting", exc_info=(exc_type, exc, tb))
        self.logger.info("=== Report session finished ===")
        # Always write/update the report on exit
        report = self.write_report()

        # Clean up handlers
        self.logger.removeHandler(self._mem_handler)
        if self._file_handler:
            self.logger.removeHandler(self._file_handler)
            self._file_handler.close()
        self._mem_handler.close()

        # Print the saved report and figures (artifacts) to console
        print(f"[StatsReporter] Wrote HTML report → {report}")
        if self._artifacts_paths:
            print("[StatsReporter] Saved Figures:")
            for p in self._artifacts_paths:
                print(f"  - {p}")

    # ---- helpers -----------------------------------------------------------

    def _unique_png_name(self, caption: str) -> str:
        safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in caption)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{ts}_{safe}.png"

    # From GPT - 5\/ HTML set up 
    @staticmethod
    def _render_html(title: str, log_text: str, plots: List[Tuple[str, str]]) -> str:
        # Very small self-contained HTML
        plot_sections = "\n".join(
            f"""
            <figure>
              <img src="data:image/png;base64,{b64}" alt="{caption}" />
              <figcaption>{caption}</figcaption>
            </figure>
            """
            for caption, b64 in plots
        )
        log_html = "<pre style='white-space:pre-wrap;margin:0'>" + (
            (log_text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        ) + "</pre>"

        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; }}
    h1 {{ margin-top: 0; }}
    figure {{ margin: 0 0 1.5rem 0; }}
    img {{ max-width: 100%; height: auto; display: block; }}
    figcaption {{ color: #555; font-size: 0.9rem; margin-top: 0.25rem; }}
    .log {{ background: #f6f8fa; border: 1px solid #e5e7eb; padding: 1rem; border-radius: 8px; }}
    .section {{ margin-bottom: 2rem; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="section log">
    <h2>Log</h2>
    {log_html}
  </div>
  <div class="section">
    <h2>Plots</h2>
    {plot_sections or "<p><em>No plots logged.</em></p>"}
  </div>
</body>
</html>
"""


from pathlib import Path
from datetime import datetime

def set_report_path(report, save_dir: str | None = None, report_name: str | None = None, timestamp: bool = True):
    """
    Configure and initialize the output HTML diagnostic report file.

    Args:
        report (StatsReporter): The active StatsReporter instance (from 'with StatsReporter(...) as report').
        save_dir (str | Path | None): Directory where the HTML report should be saved.
                                      Defaults to the current working directory.
        report_name (str | None): Optional base name for the HTML file.
                                  Default is 'DiagnosticReport.html' (or timestamped version if timestamp=True).
        timestamp (bool): Whether to append a timestamp to the file name (default: True).

    Returns:
        Path: The full path to the HTML report file.
    """
    import os

    # Determine save directory
    if save_dir is None:
        save_dir = os.getcwd()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Base filename
    if report_name is None:
        report_name = "DiagnosticReport.html"

    # Add timestamp if requested
    if timestamp:
        stem, ext = os.path.splitext(report_name)
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_name = f"{stem}_{timestamp_str}{ext}"

    # Full report path
    report_path = save_dir / report_name

    # Tell the StatsReporter to use this path
    report.report_name = report_name
    report.save_dir = save_dir
    report.write_report(report_path)

    # Log this info into the HTML and console
    report.log_text(f"Initialized HTML report at: {report_path}")
    print(f"Report will be saved to: {report_path}")

    return report_path

import inspect
from functools import wraps
from typing import Any, Callable, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.figure as mfig
