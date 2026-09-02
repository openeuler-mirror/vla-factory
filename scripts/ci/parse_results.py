#!/usr/bin/env python3
"""Parse pytest junit XML reports into CI summary dicts.

Reads the ``<tier>.xml`` files produced by the CI daemon (via
``pytest --junitxml=...``) and returns a structured summary::

    {
        "tier": "l0",
        "total": 249,
        "passed": 249,
        "failed": 0,
        "skipped": 3,
        "errors": 0,
        "time": 21.8,
        "ok": True,
        "summary": "249 passed, 3 skipped",
    }

The runner daemon calls this per tier, then per environment, and hands
the result to pr_reporter.format_result().

Standalone::

    python3 ci/runner/parse_results.py ci/_reports/l0.xml ci/_reports/l1.xml
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_junit(path: str | Path) -> dict:
    """Parse a single junit XML file into a summary dict.

    The file has the standard pytest structure::

        <testsuites>
          <testsuite name="l0" tests="252" failures="0" errors="0"
                      skipped="3" time="21.82"> ...
    """
    path = Path(path)
    if not path.exists():
        return {
            "tier": path.stem,
            "total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0,
            "time": 0.0, "ok": True, "summary": f"(not found: {path.name})",
            "failed_tests": [],
        }

    tree = ET.parse(path)
    root = tree.getroot()

    # pytest wraps in <testsuites>; single <testsuite> is also possible.
    suite = root if root.tag == "testsuite" else root.find(".//testsuite")
    if suite is None:
        suite = root

    total = int(suite.get("tests", 0))
    failures = int(suite.get("failures", 0))
    errors = int(suite.get("errors", 0))
    skipped = int(suite.get("skipped", 0))
    passed = total - failures - errors - skipped
    time = float(suite.get("time", 0))

    # Which tests failed (for the PR report): nodeid + first line of the
    # failure message, so "19 failed" is actionable without SSH-ing the
    # CI machine to read the junit xml.
    failed_tests = []
    for tc in suite.iter("testcase"):
        problem = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
        if problem is None:
            continue
        nodeid = f"{tc.get('classname', '')}::{tc.get('name', '')}".strip(":")
        msg = (problem.get("message") or problem.text or "").strip().splitlines()
        failed_tests.append({"name": nodeid, "msg": msg[0][:120] if msg else ""})

    # Build a human-readable one-liner.
    parts = []
    if passed:
        parts.append(f"{passed} passed")
    if failed := failures + errors:
        parts.append(f"{failed} failed")
    if skipped:
        parts.append(f"{skipped} skipped")
    summary = ", ".join(parts) if parts else "no tests"

    return {
        "tier": path.stem,
        "total": total,
        "passed": passed,
        "failed": failures + errors,
        "skipped": skipped,
        "time": time,
        "ok": failures == 0 and errors == 0,
        "summary": summary,
        "failed_tests": failed_tests,
    }


def parse_report_dir(report_dir: str | Path) -> dict[str, dict]:
    """Parse all <tier>.xml files in a report directory.

    Returns ``{"l0": {...}, "l1": {...}, "l2": {...}}`` keyed by stem.
    """
    report_dir = Path(report_dir)
    result = {}
    for xml in sorted(report_dir.glob("*.xml")):
        summary = parse_junit(xml)
        result[summary["tier"]] = summary
    return result


def tier_line(summary: dict) -> str:
    """Format one tier summary for the results table (e.g. '249 passed')."""
    if not summary or summary.get("total", 0) == 0:
        return "— (skip)"
    if summary.get("ok", True):
        return summary["summary"]
    return f"**{summary['summary']}**"


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        sys.exit("usage: parse_results.py <report_dir | file.xml ...>")
    args = sys.argv[1:]

    if len(args) == 1 and Path(args[0]).is_dir():
        summaries = parse_report_dir(args[0])
    else:
        summaries = {}
        for f in args:
            s = parse_junit(f)
            summaries[s["tier"]] = s

    for tier, s in sorted(summaries.items()):
        flag = "✅" if s["ok"] else "❌"
        print(f"{flag} {tier:6s}  {s['summary']:40s}  {s['time']:.1f}s")


if __name__ == "__main__":
    main()
