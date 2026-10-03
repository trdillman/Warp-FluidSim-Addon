"""tools/benchmark_versions.py: ratios pair each trial with the same round."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.benchmark_versions import _summarize


def _run(label, rnd, ms, workload="plain"):
    return {"label": label, "round": rnd, "workload": workload,
            "hostMsMedian": ms, "msPerSubstep": ms / 10.0,
            "substepsTotal": 10, "particlesLast": 100}


def test_ratio_pairs_each_trial_with_the_baseline_of_its_round():
    # Round 1 ran on a machine twice as slow; pairing cancels that, a ratio
    # of medians (60 / 20 = 3.0) would not.
    runs = [
        _run("base", 0, 10.0), _run("new", 0, 5.0),
        _run("base", 1, 20.0), _run("new", 1, 10.0),
        _run("base", 2, 30.0), _run("new", 2, 60.0),
    ]
    table = _summarize(runs, ["base", "new"], ["plain"])

    assert table["plain/base"]["ratioVsFirst"] == 1.0
    assert table["plain/new"]["ratioVsFirst"] == pytest.approx(0.5)
    assert table["plain/new"]["ratioVsFirstRange"] == [0.5, 2.0]
    assert table["plain/new"]["msMedian"] == 10.0
    assert (table["plain/new"]["msMin"], table["plain/new"]["msMax"]) == (5.0, 60.0)


def test_workloads_are_summarized_separately():
    runs = [
        _run("base", 0, 10.0), _run("new", 0, 5.0),
        _run("base", 0, 100.0, "visc"), _run("new", 0, 400.0, "visc"),
    ]
    table = _summarize(runs, ["base", "new"], ["plain", "visc"])

    assert table["plain/new"]["ratioVsFirst"] == pytest.approx(0.5)
    assert table["visc/new"]["ratioVsFirst"] == pytest.approx(4.0)
