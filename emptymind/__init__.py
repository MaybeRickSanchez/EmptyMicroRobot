"""
emptymind -- a general-purpose self-directed cognitive core.
==========================================================

A domain-agnostic, embodiment-agnostic cognitive layer that learns from
experience, forms its own questions, derives its own subgoals, and monitors
and adjusts its own thinking.

What it is
----------
A **cognitive core**, not an application. It contains no domain knowledge, no
personality, no world model that was written by hand, and no assumption about
what an action physically means. It starts empty:

    knowledge = {}, memories = {}, concepts = {}, world model = {}
    hypotheses = {}, experience = 0

and provides the *machinery* by which knowledge appears: encode, attend,
retrieve, interpret, predict, question, reason, plan, choose, act, learn,
revise, reflect.

How to use it
-------------
    from emptymind import CognitiveCore, Outcome

    core = CognitiveCore()
    core.pursue("keep the area clear")

    report = core.cognize({"obstacle": "left", "battery": 0.2})
    print(report.action, report.confidence)
    core.deliver(Outcome(value="moved left", reward=1.0))

    print(core.explain(report)["readable"])
    print(core.self_report()["current_goal"])
    print(core.stats()["cognitive"]["hypotheses"]["total"])

Embodiment is external
----------------------
The core emits :class:`~emptymind.embodiment.ActionIntent` and consumes
:class:`~emptymind.embodiment.Outcome`. Mapping an intent onto a motor
command, a sentence, or an API call is the consumer's job, and that mapping
does not belong in here. :class:`~emptymind.embodiment.TalkerAdapter` exists
for the ``EmptyTalkerRobot`` case and is an output channel, not a cognitive
component.

Relationship to ``empty.EmptyRobot``
------------------------------------
The core *is* an ``EmptyRobot``: every mechanism that already worked -- the
universal tokenizer, feature hashing, the posting-list search index, topic
clusters, ACT-R-style activation with half-life decay, the empirical world
table, the bounded candidate generator, hard constraints, the reliability
table, the save format -- is inherited and still used. What is added above it
is the cognitive architecture: attention, goals and self-directed subgoals, an
associative graph, hypotheses, questions as operations, revisable planning,
metacognition, a self model, value signals, and a recurrent cycle scheduler.

Architecture
------------
The cycle is a scheduler over processes that share one mutable state, not a
pipeline. Processes declare what they read and write; the scheduler runs
whichever is ready, re-runs processes whose inputs have changed, and is
budgeted so that recurrence terminates. That is what lets perception
influence memory, memory influence perception, predictions influence
interpretation, contradictions send the system back to look again, and
information seeking return to analysis instead of ending the cycle.

See ``emptymind/cycle.py`` for the scheduling mechanism and the README for the
full architecture.
"""

from .affect import AffectState
from .attention import Attention, AttentionBudget
from .core import CognitiveCore
from .cycle import Cycle, CycleReport, Process
from .embodiment import (ActionIntent, Embodiment, NullEmbodiment,
                         Observation, Outcome, RecordingEmbodiment,
                         TalkerAdapter)
from .goals import Goal, GoalPool
from .graph import AssociativeGraph, EDGE_TYPES
from .hypotheses import Hypothesis, HypothesisStore
from .memory import MemoryRecord, MemorySystem
from .metacognition import MetaFinding, Metacognition
from .options import Options
from .perception import Perception
from .planning import Plan, Planner, Step
from .questions import InformationRequest, Question, QuestionKind, QuestionSystem
from .reasoning import Reasoning
from .selfmodel import SelfModel
from .state import (ActiveEpisode, CognitiveState, Interpretation, OptionScore,
                    Prediction, PredictionError)
from .worldmodel import Consequence, WorldModel

__version__ = "1.0.0"

__all__ = [
    "CognitiveCore",
    # the loop
    "ActionIntent", "Observation", "Outcome", "Embodiment",
    "NullEmbodiment", "RecordingEmbodiment", "TalkerAdapter",
    "CycleReport", "Cycle", "Process",
    # shared state
    "CognitiveState", "Interpretation", "Prediction", "PredictionError",
    "OptionScore", "ActiveEpisode",
    # subsystems
    "Attention", "AttentionBudget", "AffectState", "AssociativeGraph",
    "EDGE_TYPES", "MemorySystem", "MemoryRecord", "WorldModel", "Consequence",
    "Hypothesis", "HypothesisStore", "Question", "QuestionKind",
    "QuestionSystem", "InformationRequest", "Goal", "GoalPool", "Perception",
    "Reasoning", "Options", "Plan", "Step", "Planner", "Metacognition",
    "MetaFinding", "SelfModel",
    "__version__",
]