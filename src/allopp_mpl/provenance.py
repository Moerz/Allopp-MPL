"""Pinned provenance for the vendored, byte-identical scientific core."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ENGINE_COMMIT = "9c0f246f7555a0483e0ab62633bf9af44a0361e7"
ENGINE_TREE = "5fe837db17a5f06acef49a05d2942baba7eca6e9"
PHYNETPY_COMMIT = "b3a1a411bb32a86f2136285c0fd928719be97533"

CORE_SHA256 = {
    "__main__.py": "935a1c1166b0c1ea35a82256345000bf2c73ded718d77773bc27a71ecce28f7d",
    "_initialization.py": "fb750f886b724d4c9ba3b1a42cb0b123198c6616e0498692f7b2df0bb4f5d51d",
    "_likelihood.py": "d10259a939fcb6dc5c6e2f96115165fd773191be1b530546a2d1a2d8213ddaf7",
    "additive.py": "1a9d23ca6deff17192c60d321ba8cc313c01caf89ff21e6bcc53e1f9756df7bb",
    "branch_fit.py": "7e6f338a775449188e9e388f0066d56c3df164be507cbc5e54c12f657f950eec",
    "grouped.py": "9894269154da05af76a880866bd785e4b4666f9dd198034ac1010e2e73b85c13",
    "initialization.py": "2145a1b3688f807eb65d80ca1a5fcc032ca0c1d3f53113bf460e4af859264e1e",
    "io.py": "f23d15b7144b454b9b47d32ee1a2ffa4c3e05317e49fdd4743831297a8719aba",
    "moves.py": "d66f11a81dae8e8a49e966c6e2c950b5635810e8a6677df8dca8c92fe6e59ed1",
    "paired_parent.py": "4684ab66a52d00a947a3546679818eb4ce220e20558b7fe0c5a060d4680fea9c",
    "resolution.py": "9f6dd88e96a88779a5ba0ea081b2f64e31659559654ebd542bb8342e8e205fba",
    "screening.py": "29ed9ee63f930d09eafe49decdf251709162f408afe395cf0c9764ce2abc0f27",
    "search.py": "c5337a9f748f3d9be4b90468c36d3e6a51391ffd6226a122e351d7dc82cdb8c5",
    "selection.py": "ebf244e39901f326ce24aefed343cb7b445f0583e55473a5f642eea43014c5ab",
}


def manifest_digest() -> str:
    """Return the digest used in human-readable run records."""
    manifest = {f"src/allopp_mpl/{name}": digest for name, digest in CORE_SHA256.items()}
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def verify_core_sources(package_dir: Path | None = None) -> list[str]:
    """Return source-integrity problems; an empty list means exact agreement."""
    root = Path(package_dir) if package_dir is not None else Path(__file__).resolve().parent
    problems = []
    for name, expected in CORE_SHA256.items():
        path = root / name
        if not path.is_file():
            problems.append(f"missing scientific-core file: {name}")
            continue
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            problems.append(
                f"scientific-core hash mismatch for {name}: expected {expected}, observed {observed}"
            )
    return problems


__all__ = [
    "CORE_SHA256",
    "ENGINE_COMMIT",
    "ENGINE_TREE",
    "PHYNETPY_COMMIT",
    "manifest_digest",
    "verify_core_sources",
]

