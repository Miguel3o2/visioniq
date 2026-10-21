"""Command line interface.

    visioniq                     operator console (Textual)
    visioniq headless            run the pipeline and print a summary
    visioniq benchmark           measure anomaly detection on synthetic scenes
    visioniq events              query the persisted event log
    visioniq doctor              check the environment and configuration
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

from visioniq import __version__
from visioniq.benchmark import analyse, render_report
from visioniq.config import AppConfig, ConfigError, load_config
from visioniq.core.engine import Engine
from visioniq.core.store import EventStore
from visioniq.core.types import Severity
from visioniq.vision.source import DemoScript

DEFAULT_CONFIG = "visioniq.toml"


def _setup_logging(verbose: bool, quiet: bool, log_file: Path | None = None) -> None:
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    handlers: list[logging.Handler] = []
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    else:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("textual").setLevel(logging.WARNING)


def _parse_clock(value: str | None) -> float | None:
    if not value:
        return None
    try:
        hour, minute = value.split(":")
        now = datetime.now()
        return now.replace(hour=int(hour), minute=int(minute), second=0, microsecond=0).timestamp()
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"--clock must be HH:MM (got {value!r})") from exc


def _config(args: argparse.Namespace) -> AppConfig:
    path = args.config
    if path is None and Path(DEFAULT_CONFIG).exists():
        path = DEFAULT_CONFIG
    config = load_config(path)
    if args.fresh:
        model_dir = Path(config.model_dir)
        if model_dir.exists():
            for model in model_dir.glob("*.npz"):
                model.unlink(missing_ok=True)
        logging.getLogger("visioniq").info("removed cached models")
    return config


def _demo(args: argparse.Namespace) -> DemoScript | None:
    if getattr(args, "stress", False):
        return DemoScript.stress(warmup=16.0)
    if getattr(args, "demo", False):
        return DemoScript.standard(warmup=16.0)
    return None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    config = _config(args)
    script = _demo(args)
    _setup_logging(args.verbose, args.quiet, log_file=Path("data") / "visioniq.log")
    clock = _parse_clock(args.clock)

    try:
        from visioniq.tui.app import VisionIQApp
    except ImportError as exc:
        print(f"Textual is required for the console: {exc}", file=sys.stderr)
        print("Install it with: pip install textual rich", file=sys.stderr)
        return 2

    engine = Engine(
        config,
        demo_script=script,
        clock_epoch=clock,
        speed=args.speed,
        preroll=not args.no_preroll,
    )
    app = VisionIQApp(config, engine)
    app.run()
    return 0


def cmd_headless(args: argparse.Namespace) -> int:
    config = _config(args)
    script = _demo(args)
    _setup_logging(args.verbose, args.quiet)
    engine = Engine(
        config,
        demo_script=script,
        clock_epoch=_parse_clock(args.clock),
        speed=args.speed,
        preroll=not args.no_preroll,
        record=args.report is not None,
    )
    started = time.time()
    engine.start()
    if not engine.wait_ready(args.timeout):
        print("pipeline did not become ready in time", file=sys.stderr)
        engine.stop()
        return 1

    snapshot = engine.snapshot()
    if snapshot is not None:
        print(
            f"ready in {time.time() - started:.1f}s | {len(snapshot.cameras)} cameras | "
            f"model {snapshot.model}",
            flush=True,
        )

    seen: set[int] = set()
    deadline = time.monotonic() + args.seconds
    last_print = 0.0
    try:
        while time.monotonic() < deadline:
            time.sleep(0.2)
            snapshot = engine.snapshot()
            if snapshot is None:
                continue
            for event in reversed(snapshot.events):
                if event.id in seen:
                    continue
                seen.add(event.id)
                print(
                    f"{event.clock}  {event.severity.value.upper():<8} {event.camera_id}  {event.message}",
                    flush=True,
                )
            if time.monotonic() - last_print >= 5.0:
                last_print = time.monotonic()
                metrics = snapshot.metrics
                states = " ".join(f"{c.id}:{c.state.value[0].upper()}:{c.score:.2f}" for c in snapshot.cameras)
                print(
                    f"  t={metrics.uptime_s:5.1f}s fps={metrics.fps:5.1f} "
                    f"lat={metrics.latency_ms:4.1f}ms events={metrics.events:3d} "
                    f"alerts={metrics.active_alerts:2d} | {states}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
    finally:
        snapshot = engine.snapshot()
        samples = engine.recorded_samples() if args.report else []
        engine.stop()

    counts: Counter[str] = Counter()
    if snapshot is not None:
        counts.update(event.kind for event in snapshot.events)
    print()
    print("SUMMARY")
    print(f"  runtime      {args.seconds:.0f}s")
    print(f"  events       {sum(counts.values())}")
    for kind, count in sorted(counts.items()):
        print(f"    {kind:<18}{count:>4}")

    if args.report:
        from visioniq.benchmark import analyse as _analyse
        report = _analyse(samples)
        payload = report.as_dict()
        payload["events"] = dict(counts)
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  report       {report_path}")
        print(f"  AUROC        {report.auroc:.3f} (synthetic)")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    config = _config(args)
    _setup_logging(args.verbose, True)
    script = DemoScript.evaluation()
    engine = Engine(
        config,
        demo_script=script,
        clock_epoch=_parse_clock(args.clock or "02:14"),
        speed=args.speed,
        preroll=not args.no_preroll,
        record=True,
    )
    print(
        f"synthetic benchmark: {len(script.steps)} labelled anomalies over "
        f"{script.duration:.0f}s of scene time at {args.speed:g}x"
    )
    print(
        f"  {script.anomaly_seconds:g}s injections, {script.steps[1][0] - script.steps[0][0]:g}s apart; "
        f"per-camera threshold calibration"
    )
    engine.start()
    if not engine.wait_ready(args.timeout):
        print("pipeline did not become ready in time", file=sys.stderr)
        engine.stop()
        return 1
    started = time.monotonic()
    budget = args.max_seconds
    last = 0.0
    sim_elapsed = 0.0
    try:
        while True:
            time.sleep(0.25)
            snapshot = engine.snapshot()
            if snapshot is None:
                continue
            sim_elapsed = max(0.0, snapshot.ts - engine.clock_epoch)
            elapsed = time.monotonic() - started
            if sim_elapsed >= script.duration or elapsed > budget:
                break
            if time.monotonic() - last > 5.0:
                last = time.monotonic()
                print(
                    f"  scene t={sim_elapsed:6.1f}s/{script.duration:.0f}s "
                    f"events={snapshot.metrics.events:3d} "
                    f"fps={snapshot.metrics.fps:5.1f}",
                    flush=True,
                )
    finally:
        samples = engine.recorded_samples()
        snapshot = engine.snapshot()
        event_kinds = Counter(event.kind for event in snapshot.events) if snapshot else Counter()
        engine.stop()

    report = analyse(samples)
    report.event_counts = dict(event_kinds)
    report.duration_s = sim_elapsed
    if snapshot is not None:
        report.fps = snapshot.metrics.fps
        report.latency_ms = snapshot.metrics.latency_ms
    print()
    print(render_report(report))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
        print(f"\nwrote {path}")
    return 0


def cmd_events(args: argparse.Namespace) -> int:
    config = _config(args)
    store = EventStore(config.db_path)
    rows = store.recent(args.limit)
    store.close()
    if args.json:
        payload = [asdict(event) | {"clock": event.clock, "severity": event.severity.value} for event in rows]
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if not rows:
        print("no events recorded")
        return 0
    for event in rows:
        marker = "x" if event.ack else " "
        print(f"[{marker}] {event.clock}  {event.severity.value.upper():<8} {event.camera_id:<7} {event.message}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    print(f"visioniq        {__version__}")
    print(f"python          {sys.version.split()[0]} ({sys.executable})")
    print(f"numpy           {np.__version__}")
    print(f"opencv          {cv2.__version__}")
    try:
        import torch
        print(f"torch           {torch.__version__} (LSTM autoencoder backend)")
    except ImportError:
        print("torch           not installed (using NumPy MLP autoencoder)")
    try:
        import textual
        print(f"textual         {textual.__version__}")
    except ImportError:
        print("textual         not installed (console unavailable, headless works)")

    try:
        config = load_config(args.config)
        print(f"config          {args.config or 'built-in defaults'}")
        print(f"cameras         {len(config.cameras)}")
        for camera in config.cameras:
            print(f"  {camera.id:<7} {camera.name:<20} {camera.source:<20} "
                  f"{'restricted' if camera.restricted else 'public'}")
        for path in (config.db_path, config.model_dir, config.snapshot_dir):
            p = Path(path)
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                probe = p.parent / ".visioniq_write_test"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
                print(f"writable        {p}")
            except OSError as exc:
                print(f"NOT WRITABLE    {p} ({exc})")
    except ConfigError as exc:
        print(f"config error    {exc}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visioniq",
        description="VisionIQ - real-time CCTV detection, anomaly scoring and operator console",
    )
    parser.add_argument("--version", action="version", version=f"VisionIQ {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--config", default=None, help=f"TOML config (default: {DEFAULT_CONFIG} if present)")
        sub.add_argument("--demo", action="store_true", help="run the scripted anomaly demo")
        sub.add_argument("--stress", action="store_true", help="dense anomaly schedule (benchmark)")
        sub.add_argument("--clock", default=None, help="override wall clock, HH:MM")
        sub.add_argument("--speed", type=float, default=1.0, help="simulation speed multiplier")
        sub.add_argument("--fresh", action="store_true", help="delete cached models and recalibrate")
        sub.add_argument("--no-preroll", action="store_true", help="do not fast-forward scenes before calibration")
        sub.add_argument("--timeout", type=float, default=120.0, help="startup timeout in seconds")
        sub.add_argument("--verbose", action="store_true")
        sub.add_argument("--quiet", action="store_true")

    run = subparsers.add_parser("run", help="operator console")
    common(run)
    run.set_defaults(func=cmd_run)

    headless = subparsers.add_parser("headless", help="run without the console")
    common(headless)
    headless.add_argument("--seconds", type=float, default=30.0, help="run time")
    headless.add_argument("--report", default=None, help="write a JSON report")
    headless.set_defaults(func=cmd_headless)

    bench = subparsers.add_parser("benchmark", help="measure anomaly detection on synthetic scenes")
    common(bench)
    bench.add_argument("--json", default=None, help="write the report as JSON")
    bench.add_argument("--max-seconds", type=float, default=300.0, help="wall-clock budget")
    bench.set_defaults(func=cmd_benchmark, speed=2.0)

    events = subparsers.add_parser("events", help="query the event log")
    events.add_argument("--config", default=None)
    events.add_argument("--limit", type=int, default=25)
    events.add_argument("--json", action="store_true")
    events.set_defaults(func=cmd_events, fresh=False, verbose=False, quiet=False)

    doctor = subparsers.add_parser("doctor", help="check environment and configuration")
    doctor.add_argument("--config", default=None)
    doctor.set_defaults(func=cmd_doctor, fresh=False, verbose=False, quiet=False)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not getattr(args, "command", None):
        # Default to the console: ``visioniq`` behaves like ``visioniq run``.
        argv = list(sys.argv[1:] if argv is None else argv)
        args = parser.parse_args(argv + ["run"] if argv else ["run"])
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
