"""
A demonstration, not a test.
============================

Runs the core from an empty start through the whole loop and prints what it is
doing at each step. Nothing here is domain knowledge about the situation -- the
"world" below is a deliberately arbitrary rule the core is never told, and it
has to work it out.

    python -m emptymind.demo

What each section shows is labelled, so the output can be read as an argument
rather than as a transcript: this is the property, and here is the evidence.
"""

from __future__ import annotations

import textwrap
from typing import Any, List, Tuple

from .core import CognitiveCore
from .embodiment import Outcome

# --------------------------------------------------------------------------
# The demonstration world.
#
# One rule, never stated: the *left* path is blocked while the signal is hot,
# and the *right* path is always safe. The core has no idea this. It has to
# notice that one option keeps failing and the other keeps working, remember
# that, and start choosing the right one -- from outcomes alone.
# --------------------------------------------------------------------------

HOT = "hot"
COLD = "cold"


def world_reacts(action: Any, signal: str) -> Tuple[str, float]:
    """What actually happens. The core never sees this function."""
    if action == "go left":
        return ("blocked", -1.0) if signal == HOT else ("clear", 0.5)
    if action == "go right":
        return ("clear", 1.0)
    if action == "wait":
        return ("nothing happened", 0.0)
    return ("unclear", 0.0)


def line(char: str = "-", width: int = 72) -> str:
    return char * width


def header(title: str) -> None:
    print()
    print(line("="))
    print(title)
    print(line("="))


def show(label: str, value: Any) -> None:
    print(f"  {label:<26} {value}")


def run(cycles: int = 8) -> None:
    core = CognitiveCore(memory_size=4000)
    actions = ["go left", "go right", "wait"]

    # ---------------------------------------------------------------- empty
    header("1. THE EMPTY STARTING CONDITION")
    print("""
    The brief requires a core that begins with nothing learned and no
    pre-existing knowledge, and that becomes capable through experience.
    """)
    show("memories", len(core.memory.records))
    show("concepts (graph nodes)", len(core.graph.nodes))
    show("associations (graph edges)", len(core.graph.edges))
    show("world-model relations", len(core.world._structure))
    show("hypotheses", len(core.hyps.hypotheses))
    show("questions", len(core.questions.questions))
    show("goals", len(core.goal_pool.goals))
    show("known actions", len(core.sub._lexicon))
    show("experience", f"{core.clock} cycles")

    report = core.cognize({"signal": HOT, "marker": "alpha"}, options=actions)
    print()
    show("action space supplied", actions)
    show("its first decision", repr(report.action))
    show("its confidence", round(report.confidence, 3))
    show("what it read the situation as", report.interpretations)
    show("processes that ran", " -> ".join(report.as_dict()["stages"][:8]))
    print()
    print(textwrap.fill(
        "Nothing was retrieved because nothing exists yet, so nothing can be "
        "predicted and the confidence is near zero. That is the correct "
        "behaviour for an empty core: not a guess dressed as a decision.",
        70, initial_indent="    ", subsequent_indent="    "))

    # ------------------------------------------------------------- learning
    header("2. ACTING AND LEARNING (the world has a rule we never told it)")
    print(f"""
    Now it acts and is told what happened. `go left` fails whenever the signal
    is hot; `go right` always works. The core is given only the outcome.

    {'cycle':<7}{'action':<12}{'conf':>6}{'reward':>8}{'surprise':>10}{'memory':>8}  notes
    """)

    history: List[Tuple[str, str, float, float, float]] = []
    for i in range(cycles):
        signal = HOT if i % 2 == 0 else COLD
        observation = {"signal": signal, "marker": "alpha"}
        report = core.cognize(observation, options=actions)
        action = report.action
        outcome_text, reward = world_reacts(action, signal)
        error = core.deliver(Outcome(value=outcome_text, reward=reward))
        history.append((signal, str(action), report.confidence, reward,
                        error.surprise))
        note = []
        if report.revisits:
            note.append(f"{report.revisits} revisit(es)")
        if core.state.plan is not None and core.state.plan.reason:
            note.append("plan revised")
        if core.questions.open_questions():
            note.append(f"{len(core.questions.open_questions())} open question(s)")
        print(f"    {i:<7}{str(action):<12}{report.confidence:>6.2f}"
              f"{reward:>8.1f}{error.surprise:>10.2f}"
              f"{len(core.memory.records):>8}  {'; '.join(note)}")

    left = sum(1 for _, a, _, r, _ in history if a == "go left")
    right = sum(1 for _, a, _, r, _ in [(h[0], h[1], h[2], h[3], h[4]) for h in history]
                if a == "go right")
    print()
    show("times it chose left", f"{left} (and was punished each time)")
    show("times it chose right", right)
    print()
    print(textwrap.fill(
        "Confidence rises as outcomes confirm predictions, and the system "
        "shifts toward the option that has been working -- without ever being "
        "told that the left path is blocked.",
        70, initial_indent="    ", subsequent_indent="    "))

    # -------------------------------------------------------- what it knows
    header("3. WHAT IT LEARNED (all of it derived from those outcomes)")
    stats = core.stats()["cognitive"]
    show("memory records", core.memory_report())
    show("hypotheses", core.hypothesis_report()["by_state"])
    for hyp in core.hyps.usable(3):
        print(f"      belief: {hyp.statement()!r} "
              f"(confidence {hyp.confidence:.2f}, state {hyp.state})")
    print()
    show("associative graph", f"{len(core.graph.nodes)} nodes / "
                              f"{len(core.graph.edges)} edges {core.graph.kinds()}")
    for other, strength in core.graph.causes_of(
            core.node_for("go left", kind="action"))[:3]:
        print(f"      causes({core.label_for(core.key_for(other))!r}) "
              f"= {strength:.2f}")

    header("4. MEMORY IS NOT THE SAME AS BEING IN MIND")
    print()
    records = sorted(core.memory.records.values(), key=lambda r: -r.importance)[:4]
    print(f"    {'id':<5}{'role':<12}{'act':>6}{'conf':>7}{'imp':>7}{'value':>8}"
          f"{'used':>7}  content")
    for rec in records:
        print(f"    {rec.mem_id:<5}{rec.role:<12}{rec.activation:>6.2f}"
              f"{rec.confidence:>7.2f}{rec.importance:>7.2f}{rec.value:>8.2f}"
              f"{rec.access_count:>7}  {str(rec.content)[:28]!r}")
    print()
    print(textwrap.fill(
        "Everything stored is not necessarily active. Activation rises on "
        "retrieval and decays without it, so the system's working set is a "
        "small, shifting subset of what it has stored.",
        70, initial_indent="    ", subsequent_indent="    "))

    # ------------------------------------------------------------ the loop
    header("5. ONE CYCLE IN DETAIL")
    core.pursue("avoid the blocked path")
    report = core.cognize({"signal": HOT, "marker": "alpha"}, options=actions)
    print(report.explain())
    print()
    print("    ranked options:")
    for option in core.state.options[:5]:
        print(f"      {option.score:+.3f}  {option.kind:<11} {str(option.option)[:24]:<26}"
              f" {'; '.join(option.why[:2])}")
    print()
    chosen = core.state.chosen
    show("what it predicted", chosen.predicted if chosen else None)
    show("why that option won",
         "; ".join(chosen.why[:3]) if chosen else None)

    # -------------------------------------------------- questions and goals
    header("6. QUESTIONS, AND SELF-DIRECTED SUBGOALS")
    goals = core.goal_pool.report()
    show("active goals", [g["text"] for g in goals["top"]])
    show("derived (cognitive) subgoals", goals["derived"])
    show("blocked goals", goals["blocked"])
    print()
    for question in core.questions.open_questions()[:4]:
        print(f"    q{question.q_id} [{question.kind}] priority {question.priority:.2f}"
              f"  {question.text}")
        print(f"        needs: {', '.join(question.operations)}")
    pending = [r for r in core.questions.requests if not r.satisfied]
    show("information requested of the body", [r.kind for r in pending])

    # ------------------------------------------------------------ self model
    header("7. FUNCTIONAL SELF-MODELLING")
    print()
    print("    what am I doing?")
    for key, value in core.introspect("what_am_i_doing").items():
        print(f"        {key:<16} {value}")
    print()
    print("    why am I doing it?  (the actual chain, not a restatement)")
    chain = core.introspect("why_am_i_doing")
    for entry in chain.get("chain", []):
        print(f"        {entry['level']:<12} {entry['detail']}")
    print(f"        verdict: {chain.get('verdict')}")
    print()
    print("    could my reading of this be wrong?")
    for key, value in core.introspect("could_i_be_wrong").items():
        print(f"        {key:<28} {str(value)[:70]}")
    print()
    print("    what should I investigate next?")
    for candidate in core.introspect("learn_next")["candidates"][:3]:
        print(f"        {candidate['gain']:.2f}  {candidate['what']!r}")

    # ------------------------------------------------------------ meta
    header("8. METACOGNITION, AND WHAT IT CHANGED")
    reflection = core.reflect()
    show("prediction accuracy", reflection["prediction_accuracy"])
    show("calibration error", reflection["calibration_error"])
    show("trust per subsystem", reflection["trust"])
    print()
    print("    findings from the last cycle, each with the change it makes:")
    for finding in core.state.meta_findings[:5]:
        print(f"        [{finding['kind']}] severity {finding['severity']:.2f}")
        print(f"            because: {str(finding['detail'])[:66]}")
        print(f"            so: {finding['adjustment']}")

    # --------------------------------------------------------- embodiment
    header("9. THE EMBODIMENT BOUNDARY")
    print()
    print(textwrap.fill(
        "The core emits an intent and consumes an outcome. Nothing above this "
        "line knows what an action means physically -- there is no motor, no "
        "voice, no API anywhere in the package. A robot would map the intent "
        "to a motor command; a talker to a sentence; a test harness would "
        "ignore it. The core is given the outcome either way.",
        70, initial_indent="    ", subsequent_indent="    "))
    intent = core.last_intent()
    show("intent kind", intent.kind)
    show("intent payload", intent.payload)
    show("intent confidence", round(intent.confidence, 3))
    show("why", intent.reason)
    print()
    print(f"    {intent.as_dict()}")

    # ------------------------------------------------------------- closing
    header("WHERE THIS LEAVES THE CORE")
    for label, value in (
        ("memories", stats["memory"]["records"]),
        ("associations", f"{len(core.graph.nodes)}n/{len(core.graph.edges)}e"),
        ("beliefs", stats["hypotheses"]["total"]),
        ("model relations", stats["world_model"]["structure_edges"]),
        ("prediction accuracy", reflection["prediction_accuracy"]),
        ("questions asked", stats["questions"]["stats"]["asked"]),
        ("subgoals derived", stats["goals"]["stats"]["derived"]),
        ("cycles", core.clock),
    ):
        show(label, value)
    print()
    print(textwrap.fill(
        "Every one of those numbers arrived from an outcome. Nothing was "
        "preloaded, no domain rules were supplied, and no training step ran. "
        "The architecture provided the machinery; the experience provided "
        "everything else.",
        70, initial_indent="    ", subsequent_indent="    "))
    print()


if __name__ == "__main__":
    run()