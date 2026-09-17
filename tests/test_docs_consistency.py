"""The public docs must describe what the code actually registers.

The catalog in the installation guide has drifted twice during merges, and a plugin or
metric that no document mentions is one a reader cannot find. These checks are cheap and
need no network, so the drift fails here rather than in front of a user.
"""

import pathlib
import re

import fltest.attacks  # noqa: F401
import fltest.defenses  # noqa: F401
import fltest.frameworks  # noqa: F401
import fltest.metrics  # noqa: F401
from fltest.core.registry import ATTACKS, DEFENSES, METRICS
from fltest.testing.metamorphic import _RELATIONS

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"
README = pathlib.Path(__file__).resolve().parent.parent / "README.md"


def _read(name):
    return (DOCS / name).read_text()


def test_installation_catalog_matches_the_registry():
    """The `fltest list` output pasted into the install guide must be the real one."""
    text = _read("installation.md")
    for label, names in (("Attacks", ATTACKS.names()), ("Defenses", DEFENSES.names())):
        line = re.search(rf"# {label}:\s*(\[[^\]]*\])", text)
        assert line, f"no {label} line in installation.md"
        listed = set(re.findall(r"'([a-z_0-9]+)'", line.group(1)))
        assert listed == set(names), (
            f"{label} in installation.md is {sorted(listed)}, registry has {sorted(names)}"
        )


def test_every_plugin_appears_on_its_reference_page():
    for names, page in ((ATTACKS.names(), "attacks.md"),
                        (DEFENSES.names(), "defenses.md"),
                        (METRICS.names(), "metrics.md")):
        text = _read(page)
        missing = [n for n in names if f"`{n}`" not in text]
        assert not missing, f"{page} does not document {missing}"


def test_every_plugin_appears_in_the_readme():
    text = README.read_text()
    missing = [n for n in list(ATTACKS.names()) + list(DEFENSES.names()) if f"`{n}`" not in text]
    assert not missing, f"README does not mention {missing}"


def test_every_metamorphic_relation_is_documented():
    text = _read("metamorphic-testing.md")
    missing = [r for r in _RELATIONS if f"`{r}`" not in text]
    assert not missing, f"metamorphic-testing.md does not document {missing}"


def test_every_pitfall_id_is_documented():
    checker = (pathlib.Path(__file__).resolve().parent.parent
               / "fltest" / "pitfalls" / "checker.py").read_text()
    ids = sorted(set(re.findall(r'"(P\d_[a-z_]+)"', checker)))
    text = _read("pitfalls.md")
    missing = [i for i in ids if i not in text]
    assert not missing, f"pitfalls.md does not document {missing}"


def test_every_example_config_is_referenced_somewhere():
    configs = {p.name for p in (DOCS.parent / "examples" / "configs").glob("*.yaml")}
    corpus = README.read_text() + "".join(p.read_text() for p in DOCS.glob("*.md"))
    missing = sorted(c for c in configs if c not in corpus)
    assert not missing, f"no doc references {missing}"
