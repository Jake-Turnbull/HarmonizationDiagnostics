#!/usr/bin/env python3
"""
run_diagnostics_pipeline.py — subprocess-based pipeline with finalidplist fix

- Runs diagnosticfunctions.py per input (into per-input output dirs)
- Runs getPlots.py once to produce comparison plots
- Runs getReport.py once to produce HTML report
- Supports CLI or JSON config (--config). CLI overrides config.
- Fix: if finalidplist is a Python list, it's json.dumps()-ed before calling subprocess.
"""
import os
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def run_cmd(cmd: List[str], verbose: bool = False) -> None:
    if verbose:
        print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Command failed with exit code {e.returncode}:\n  {' '.join(cmd)}\n")
        sys.exit(e.returncode)


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config JSON must be an object at top level.")
    return cfg


def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    list_keys = {"input", "output", "fixeff", "zscore_fixeff", "inputs", "finalidplist"}
    for k in list_keys:
        if k in out and out[k] is not None:
            if isinstance(out[k], list):
                pass
            elif isinstance(out[k], str):
                out[k] = [out[k]]
            else:
                try:
                    out[k] = list(out[k])
                except Exception:
                    out[k] = out[k]
    for int_key in ("start_col", "end_col", "nperm"):
        if int_key in out and out[int_key] is not None:
            out[int_key] = int(out[int_key])
    return out


def build_main_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run diagnostics -> getPlots -> getReport (subprocess).")
    p.add_argument("--config", type=str, help="JSON config file path (merged, CLI overrides).")
    p.add_argument("--verbose", action="store_true", default=defaults.get("verbose", False))
    p.add_argument("--input", nargs="+", default=defaults.get("input"), help="One or more input CSV files.")
    p.add_argument("--output", nargs="+", default=defaults.get("output"),
                   help="Single base output dir OR one output dir per input (1:1 mapping).")
    p.add_argument("--subject", default=defaults.get("subject"))
    p.add_argument("--fixeff", nargs="+", default=defaults.get("fixeff"), help="Fixed effect names (e.g. age sex).")
    p.add_argument("--zscore_fixeff", nargs="+", default=defaults.get("zscore_fixeff"))
    p.add_argument("--raneff", default=defaults.get("raneff"))
    p.add_argument("--batch", default=defaults.get("batch"))
    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument("--finalidplist", default=defaults.get("finalidplist"),
                       help="JSON list string for finalidplist (pass as single string or provide list in config).")
    group.add_argument("--start_col", type=int, default=defaults.get("start_col"), help="Start column index (int).")
    p.add_argument("--end_col", type=int, default=defaults.get("end_col"), help="End column index (int).")
    p.add_argument("--run", default=defaults.get("run", "all"))
    p.add_argument("--nperm", type=int, default=defaults.get("nperm", 1000))
    p.add_argument("--comparison_plots_dir", default=defaults.get("comparison_plots_dir"),
                   help="Optional: explicit comparison plots directory.")
    p.add_argument("--report_outdir", default=defaults.get("report_outdir"),
                   help="Optional: explicit report output directory.")
    return p


def validate_args(args: argparse.Namespace) -> None:
    if not args.input:
        raise ValueError("No inputs supplied. Use --input or provide in --config.")
    if not args.output:
        raise ValueError("No output supplied. Use --output or provide in --config.")
    if not args.finalidplist:
        if args.start_col is None or args.end_col is None:
            raise ValueError("Either --finalidplist or both --start_col and --end_col must be provided.")
    if len(args.output) != 1 and len(args.output) != len(args.input):
        raise ValueError("When providing multiple --output values, number of outputs must equal number of inputs.")


def main():
    # parse minimal pre-args to pick up config path / verbose quickly
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str)
    pre.add_argument("--verbose", action="store_true")
    pre_args, remaining_argv = pre.parse_known_args()

    cfg = {}
    if pre_args.config:
        cfg = load_config(pre_args.config)
        cfg = normalize_config(cfg)

    parser = build_main_parser(cfg)
    args = parser.parse_args(remaining_argv)

    if not getattr(args, "config", None):
        args.config = pre_args.config

    try:
        validate_args(args)
    except Exception as e:
        print("Argument validation error:", e)
        parser.print_help()
        sys.exit(2)

    verbose = args.verbose
    input_paths = [Path(p) for p in args.input]
    outputs_arg = [Path(p) for p in args.output]

    # build per-input output directories
    if len(outputs_arg) == 1:
        base_out = outputs_arg[0]
        per_outputs = []
        for inp in input_paths:
            d = base_out / inp.stem
            d.mkdir(parents=True, exist_ok=True)
            per_outputs.append(d)
    else:
        per_outputs = []
        for out in outputs_arg:
            out.mkdir(parents=True, exist_ok=True)
            per_outputs.append(out)

    # comparison plots dir (single directory)
    if args.comparison_plots_dir:
        comparison_dir = Path(args.comparison_plots_dir)
    else:
        if len(outputs_arg) == 1:
            comparison_dir = outputs_arg[0] / "comparison_plots"
        else:
            comparison_dir = per_outputs[0].parent / "comparison_plots"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # report dir (default to comparison_dir)
    report_dir = Path(args.report_outdir) if args.report_outdir else comparison_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    # inspection dir
    if args.output:                   # user supplied something like --output /path/to/out
        base_out = Path(args.output[0])
    else:                             # no output provided → use current working directory
        base_out = Path(os.getcwd())
    # Ensure base directory exists
    base_out.mkdir(parents=True, exist_ok=True)
    # Create the "inspect" directory inside it
    inspect_dir = base_out / "inspect"
    inspect_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Config (if any):", cfg or "<none>")
        print("Final merged args:")
        print("  inputs:", input_paths)
        print("  per-output dirs:", per_outputs)
        print("  comparison_plots_dir:", comparison_dir)
        print("  report_dir:", report_dir)

    # 1) Run diagnosticfunctions per input
    for inp, outdir in zip(input_paths, per_outputs):
        cmd = ["python", "DiagnosticFunctions.py",
               "--input", str(inp),
               "--subject", args.subject,
               "--fixeff"] + [str(x) for x in args.fixeff] + \
              ["--zscore_fixeff"] + [str(x) for x in args.zscore_fixeff] + \
              ["--raneff", args.raneff,
               "--batch", args.batch]

        # FIX: ensure finalidplist passed as a single JSON string if it's a list
        if args.finalidplist:
            # args.finalidplist may already be a JSON string (from CLI) or a Python list (from config)
            if isinstance(args.finalidplist, (list, tuple)):
                finalid_arg = json.dumps(args.finalidplist)
            else:
                finalid_arg = str(args.finalidplist)
            cmd += ["--finalidplist", finalid_arg]
        else:
            if args.start_col is None or args.end_col is None:
                print("ERROR: either --finalidplist or both --start_col and --end_col must be provided")
                sys.exit(2)
            cmd += ["--start_col", str(args.start_col), "--end_col", str(args.end_col)]

        cmd += ["--output", str(outdir), "--run", args.run, "--nperm", str(args.nperm)]

        run_cmd(cmd, verbose=verbose)

    # 2) Run getPlots once over all per-output dirs -> comparison_plots
    getplots_cmd = ["python", "PlotDiagnosticResults.py", "--dirs"] + [str(p) for p in per_outputs] + \
                   ["--outdir", str(comparison_dir), "--fixeff"] + [str(x) for x in args.fixeff]
    run_cmd(getplots_cmd, verbose=verbose)

    # 3) Run getReport once with comparison_plots dir and inputs list for header
    getreport_cmd = ["python", "DiagnosticReport.py",
                     "--dir", str(comparison_dir),
                     "--outdir", str(report_dir),
                     "--fixeff"] + [str(x) for x in args.fixeff] + \
                    ["--inputs"] + [str(p) for p in input_paths]
    run_cmd(getreport_cmd, verbose=verbose)

    # 4) Run data inspection pipeline
    # normalize input_paths to Path objects then to strings
    input_paths = [Path(p) for p in input_paths]  # if already Paths this is harmless
    input_args = [str(p) for p in input_paths]

    # normalize batch
    batch_arg = str(args.batch) if getattr(args, "batch", None) else None

    # normalize bio vars (args.fixeff in your code)
    fixeff = getattr(args, "fixeff", None)
    bio_vars_arg = None
    if fixeff:
        if isinstance(fixeff, (list, tuple)):
            bio_vars_arg = ",".join(str(x) for x in fixeff)
        else:
            parts = [s for s in str(fixeff).replace(",", " ").split() if s]
            bio_vars_arg = ",".join(parts)


    # normalize features (finalid_arg may be JSON string or plain string)
    def normalize_features(features_arg):
        if features_arg is None:
            return None
        if isinstance(features_arg, (list, tuple)):
            return ",".join(map(str, features_arg))
        # if it's a string, try to parse JSON list, otherwise use as-is
        s = str(features_arg)
        try:
            parsed = json.loads(s)
            if isinstance(parsed, (list, tuple)):
                return ",".join(map(str, parsed))
            # if parsed to non-list, fall back to original string
        except Exception:
            pass
        return s

    features_spec = normalize_features(finalid_arg)

    # build the command as a flat list
    getinspect_cmd = ["python", "DataInspection.py"] + input_args

    if batch_arg:
        getinspect_cmd += ["--batch_vars", batch_arg]

    if bio_vars_arg:
        # DataInspection appears to accept multiple bio vars (nargs='+'), so pass them separately
        getinspect_cmd += ["--bio_vars", bio_vars_arg]

    if features_spec:
        getinspect_cmd += ["--features", features_spec]

    if getattr(args, "subject", None):
        getinspect_cmd += ["--subject_id", str(args.subject)]

    # ensure inspect_dir is string
    getinspect_cmd += ["--output_dir", str(inspect_dir)]

    # add flags you want (toggle if you prefer conditional)
    getinspect_cmd += ["--report", "--embed_images"]

    # debug print (flat list)
    print(getinspect_cmd)
    run_cmd(getinspect_cmd, verbose=verbose)

    print("\n✅ Pipeline finished.")
    print("Inspection output:", inspect_dir)
    print("Comparison plots:", comparison_dir)
    print("Report output:", report_dir)


if __name__ == "__main__":
    main()
