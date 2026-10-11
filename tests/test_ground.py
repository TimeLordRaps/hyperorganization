"""The epistemic contract: a verdict is checked-here, never established, and cannot be used as a bool.

`ground` is the one place where the words HOLDS_IN_MODEL, FAILS_IN_MODEL and the tags
FORM / FRAME / HYPER / OPEN get their meaning in this package. The tests pin the three
promises everything else leans on:

1. a Verdict has no truth value, so `if verdict:` cannot silently turn "checked in one
   model" into "true";
2. a failure names its counterexample and a success does not invent a witness, but may
   record what was measured (`evidence`), because a number that was read is not a
   witness that was constructed;
3. `OMEGA`, the rank of a state that survives every finite stage, sorts above every
   natural number and is not itself a natural number.
"""

from __future__ import annotations

import pytest

from hyperoptimization.ground import OMEGA, Omega, Status, Tag, Verdict, rank_key


def test_a_verdict_has_no_truth_value():
    verdict = Verdict("claim", Status.HOLDS_IN_MODEL)
    with pytest.raises(TypeError, match="no truth value"):
        bool(verdict)
    with pytest.raises(TypeError):
        if verdict:  # noqa: SIM102 - the point of the test
            pass


def test_a_failure_must_name_its_counterexample():
    with pytest.raises(ValueError, match="counterexample"):
        Verdict("claim", Status.FAILS_IN_MODEL)
    failure = Verdict("claim", Status.FAILS_IN_MODEL, ("m=5,q=3",))
    assert failure.witness == ("m=5,q=3",)


def test_a_success_has_no_witness_but_may_record_what_was_measured():
    with pytest.raises(ValueError, match="no distinguished witness"):
        Verdict("claim", Status.HOLDS_IN_MODEL, ("anything",))
    measured = Verdict("claim", Status.HOLDS_IN_MODEL, evidence=("x_c=1/2",))
    assert measured.evidence == ("x_c=1/2",) and measured.witness == ()


@pytest.mark.parametrize("bad", ["", "   ", None, 3])
def test_a_claim_is_a_nonempty_string(bad):
    with pytest.raises((ValueError, TypeError)):
        Verdict(bad, Status.HOLDS_IN_MODEL)


def test_status_and_witness_and_evidence_are_type_checked():
    with pytest.raises(TypeError):
        Verdict("c", "HOLDS_IN_MODEL")  # a string is not a Status
    with pytest.raises(TypeError):
        Verdict("c", Status.FAILS_IN_MODEL, ["list-not-tuple"])
    with pytest.raises(TypeError):
        Verdict("c", Status.HOLDS_IN_MODEL, evidence=("ok", 3))


def test_the_four_tags_are_the_familys_and_only_those():
    assert [t.value for t in Tag] == ["FORM", "FRAME", "HYPER", "OPEN"]


def test_omega_sorts_above_every_natural_number_and_is_a_singleton():
    assert Omega() is OMEGA
    assert repr(OMEGA) == "ω" and str(OMEGA) == "ω"
    ranks = [3, OMEGA, 0, 17, OMEGA]
    assert sorted(ranks, key=rank_key) == [0, 3, 17, OMEGA, OMEGA]
    assert rank_key(10**100) < rank_key(OMEGA)


def test_omega_is_not_a_natural_number_and_cannot_be_used_as_one():
    assert not isinstance(OMEGA, int)
    with pytest.raises(TypeError):
        OMEGA + 1  # type: ignore[operator]
    with pytest.raises(TypeError):
        rank_key(-1)
    with pytest.raises(TypeError):
        rank_key(2.5)
    with pytest.raises(TypeError):
        rank_key(True)
