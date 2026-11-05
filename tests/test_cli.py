from pathlib import Path
import os
import re
import subprocess
import sys
import tomllib

import pytest

import allopp_mpl.cli as cli
from allopp_mpl.cli import (
    INITIALIZER_ALIASES,
    PACKAGE_VERSION,
    _stream_seeds,
    build_parser,
    main,
)
from allopp_mpl.io import read_network


ROOT = Path(__file__).resolve().parents[1]
TREES = ROOT / "examples" / "gene_trees.nwk"
MAP = ROOT / "examples" / "gene_map.json"


def arguments(output, *extra):
    return [
        "infer",
        "--gene-trees", str(TREES),
        "--gene-map", str(MAP),
        "--reticulations", "1",
        "--evaluations", "8",
        "--streams", "1",
        "--starts", "2",
        "--output", str(output),
        *extra,
    ]


def test_public_initializer_name_is_exact_additive_alias():
    assert INITIALIZER_ALIASES == {
        "ploidy-compatible": "additive-v1",
        "additive-v1": "additive-v1",
    }


def test_stream_seeds_are_deterministic_and_keep_first_seed():
    assert _stream_seeds(17, 1) == [17]
    assert _stream_seeds(17, 3) == _stream_seeds(17, 3)
    assert len(set(_stream_seeds(17, 3))) == 3


def test_distribution_name_matches_frozen_module_metadata_lookup():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["name"] == "allopp-mpl"
    assert project["project"]["version"] == PACKAGE_VERSION == "0.1.0"
    assert project["project"]["requires-python"] == ">=3.12"
    assert project["project"]["scripts"]["allopp-mpl"] == "allopp_mpl.cli:main"


@pytest.mark.parametrize("module", ["allopp_mpl.cli", "allopp_mpl"])
def test_both_entry_modules_use_concise_outputs(tmp_path, module):
    output = tmp_path / module.replace(".", "-")
    environment = os.environ.copy()
    source = str(ROOT / "src")
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            "infer",
            "--gene-trees", str(TREES),
            "--gene-map", str(MAP),
            "--reticulations", "1",
            "--evaluations", "8",
            "--streams", "1",
            "--starts", "2",
            "--output", str(output),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert {path.name for path in output.iterdir()} == {"network.newick", "results.txt"}
    text = (output / "results.txt").read_text(encoding="utf-8")
    assert "Status: SUCCESS" in text
    assert not (output / "result.json").exists()


def test_single_release_version_everywhere():
    import allopp_mpl

    assert allopp_mpl.__version__ == PACKAGE_VERSION == "0.1.0"


@pytest.mark.parametrize("command", ["select", "select-cv"])
def test_only_infer_command_exists(command):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([command])


def test_reticulations_is_required(tmp_path):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "infer",
                "--gene-trees", str(TREES),
                "--gene-map", str(MAP),
                "--output", str(tmp_path / "missing-reticulations"),
            ]
        )


def test_cli_regresses_to_approved_frozen_engine_score(tmp_path, capsys):
    output = tmp_path / "result"
    assert main(arguments(output)) == 0
    assert {path.name for path in output.iterdir()} == {"network.newick", "results.txt"}
    text = (output / "results.txt").read_text()
    assert "Status: SUCCESS" in text
    assert "Best log-pseudolikelihood: -5.545177444479562" in text
    assert "starts_screened=2" in text
    for section in ("Settings", "Inputs", "Reproducibility", "Interpretation"):
        assert f"\n{section}\n" not in text
    assert "_sha256" not in text
    read_network(output / "network.newick")
    assert "wrote" in capsys.readouterr().out


def test_multistream_matches_separate_single_stream_jobs(tmp_path):
    multi = tmp_path / "multi"
    assert main(arguments(multi, "--streams", "3")) == 0
    rows = {
        int(stream): (int(seed), float(score))
        for stream, seed, score in re.findall(
            r"^  (\d+): success; seed=(\d+); score=([^;]+);",
            (multi / "results.txt").read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
    }
    seeds = _stream_seeds(17, 3)
    assert [rows[index][0] for index in (1, 2, 3)] == seeds

    separate_networks = {}
    for stream, seed in enumerate(seeds, 1):
        output = tmp_path / f"single-{stream}"
        assert main(arguments(output, "--seed", str(seed))) == 0
        text = (output / "results.txt").read_text(encoding="utf-8")
        score = float(re.search(r"^Best log-pseudolikelihood: (.+)$", text, re.MULTILINE).group(1))
        assert score == rows[stream][1]
        separate_networks[stream] = (output / "network.newick").read_text(encoding="utf-8")

    best_stream = max(rows, key=lambda stream: (rows[stream][1], -stream))
    assert (multi / "network.newick").read_text(encoding="utf-8") == separate_networks[best_stream]


def test_unknown_tree_label_is_usage_error_before_search(tmp_path, monkeypatch, capsys):
    trees = tmp_path / "unknown-label.nwk"
    trees.write_text("((a,b),unknown_copy);\n", encoding="utf-8")
    output = tmp_path / "must-not-exist"
    args = arguments(output)
    args[args.index("--gene-trees") + 1] = str(trees)
    called = False

    def unexpected_search(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("search must not start")

    monkeypatch.setattr(cli, "infer", unexpected_search)
    with pytest.raises(SystemExit):
        main(args)
    assert not called
    assert not output.exists()
    assert "tree 1 has gene-copy label absent from the gene map: 'unknown_copy'" in capsys.readouterr().err


def test_supplied_start_replaces_automatic_starts(tmp_path):
    first = tmp_path / "first"
    assert main(arguments(first)) == 0
    second = tmp_path / "second"
    args = arguments(
        second,
        "--start-network", str(first / "network.newick"),
        "--evaluations", "1",
    )
    assert main(args) == 0
    text = (second / "results.txt").read_text()
    assert "starts_screened=1" in text
    assert "Automatic starts are disabled" in text
    assert "start_network_sha256" not in text


def test_initialization_cap_exhaustion_is_not_infeasibility(tmp_path, capsys):
    output = tmp_path / "failed"
    args = arguments(
        output,
        "--max-init-states", "1",
        "--evaluations", "1",
        "--starts", "1",
    )
    assert main(args) == 2
    assert {path.name for path in output.iterdir()} == {"results.txt"}
    text = (output / "results.txt").read_text()
    assert "Status: FAILED" in text
    assert "initialization-construction-failure" in text
    assert "not proof that the requested reticulation count is impossible" in text
    assert "Inference failed" in capsys.readouterr().err

