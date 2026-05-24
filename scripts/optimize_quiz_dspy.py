"""
Offline DSPy prompt optimizer for the Quiz Agent.

Run once after setup:
    python -m scripts.optimize_quiz_dspy

Output: app/agents/dspy_optimized_quiz.json

Fixes vs previous version:
  - Replaced deprecated OllamaLocal with dspy.LM (DSPy 2.5+)
  - theta and b_target stored as str in trainset (DSPy requires all inputs to be str)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dspy
from dspy.teleprompt import BootstrapFewShot

from app.agents.quiz_agent import QuizSignature
from app.core.config import settings

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "app", "agents", "dspy_optimized_quiz.json"
)


TRAINSET_RAW = [
    # ── CODING ──────────────────────────────────────────────────────────────
    dict(
        topic="Python basics",
        theta="-0.50",
        b_target="-0.30",
        question="Which keyword is used to define a function in Python?",
        choice_a="func",
        choice_b="define",
        choice_c="def",
        choice_d="function",
        correct="C",
        explanation="Python uses the `def` keyword to declare a function.",
    ),
    dict(
        topic="Python lists",
        theta="0.30",
        b_target="0.50",
        question="What is the output of `[1, 2, 3][::-1]`?",
        choice_a="[1, 2, 3]",
        choice_b="[3, 2, 1]",
        choice_c="[2, 1, 3]",
        choice_d="Error",
        correct="B",
        explanation="Slice `[::-1]` reverses a list, producing [3, 2, 1].",
    ),
    dict(
        topic="Big-O complexity",
        theta="1.20",
        b_target="1.40",
        question="Which sorting algorithm has O(n log n) average-case time complexity?",
        choice_a="Bubble sort",
        choice_b="Insertion sort",
        choice_c="Merge sort",
        choice_d="Selection sort",
        correct="C",
        explanation="Merge sort divides and merges in O(n log n) on average and worst case.",
    ),
    # ── CHEMISTRY ───────────────────────────────────────────────────────────
    dict(
        topic="Periodic table",
        theta="-0.50",
        b_target="-0.30",
        question="What is the chemical symbol for gold?",
        choice_a="Go",
        choice_b="Gd",
        choice_c="Au",
        choice_d="Ag",
        correct="C",
        explanation="Gold's symbol Au comes from the Latin word 'aurum'.",
    ),
    dict(
        topic="Atomic structure",
        theta="0.00",
        b_target="0.20",
        question="What particles are found in the nucleus of an atom?",
        choice_a="Electrons and protons",
        choice_b="Protons and neutrons",
        choice_c="Neutrons and electrons",
        choice_d="Protons only",
        correct="B",
        explanation="The nucleus contains protons (positive) and neutrons (neutral).",
    ),
    dict(
        topic="Chemical bonding",
        theta="0.40",
        b_target="0.60",
        question="What type of bond involves sharing of electron pairs between atoms?",
        choice_a="Ionic bond",
        choice_b="Metallic bond",
        choice_c="Covalent bond",
        choice_d="Hydrogen bond",
        correct="C",
        explanation="Covalent bonds form when atoms share electrons.",
    ),
    # ── BIOLOGY ─────────────────────────────────────────────────────────────
    dict(
        topic="Cell biology",
        theta="-0.30",
        b_target="-0.10",
        question="What organelle is known as the powerhouse of the cell?",
        choice_a="Nucleus",
        choice_b="Ribosome",
        choice_c="Mitochondria",
        choice_d="Golgi apparatus",
        correct="C",
        explanation="Mitochondria produce ATP through cellular respiration.",
    ),
    dict(
        topic="DNA",
        theta="0.20",
        b_target="0.40",
        question="What is the complementary base pair of adenine in DNA?",
        choice_a="Guanine",
        choice_b="Cytosine",
        choice_c="Uracil",
        choice_d="Thymine",
        correct="D",
        explanation="In DNA, adenine pairs with thymine via two hydrogen bonds.",
    ),
    dict(
        topic="Evolution",
        theta="0.60",
        b_target="0.80",
        question="What describes organisms better adapted to their environment surviving and reproducing?",
        choice_a="Genetic drift",
        choice_b="Natural selection",
        choice_c="Gene flow",
        choice_d="Mutation",
        correct="B",
        explanation="Natural selection: individuals with advantageous traits leave more offspring.",
    ),
    # ── PHYSICS ─────────────────────────────────────────────────────────────
    dict(
        topic="Kinematics",
        theta="0.00",
        b_target="0.20",
        question="An object accelerates from rest at 4 m/s² for 3 s. Final velocity?",
        choice_a="4 m/s",
        choice_b="7 m/s",
        choice_c="12 m/s",
        choice_d="16 m/s",
        correct="C",
        explanation="v = u + at = 0 + 4×3 = 12 m/s.",
    ),
    dict(
        topic="Newton's laws",
        theta="0.30",
        b_target="0.50",
        question="A 5 kg box is pushed with 20 N. Acceleration (no friction)?",
        choice_a="2 m/s²",
        choice_b="4 m/s²",
        choice_c="100 m/s²",
        choice_d="0.25 m/s²",
        correct="B",
        explanation="a = F/m = 20/5 = 4 m/s².",
    ),
    dict(
        topic="Electrostatics",
        theta="0.60",
        b_target="0.80",
        question="If distance between two charges doubles, Coulomb force becomes:",
        choice_a="4× larger",
        choice_b="2× larger",
        choice_c="Half",
        choice_d="One-quarter",
        correct="D",
        explanation="F ∝ 1/r²; doubling r gives F/4.",
    ),
    # ── MATHEMATICS ─────────────────────────────────────────────────────────
    dict(
        topic="Algebra",
        theta="-0.20",
        b_target="0.00",
        question="Solve for x: 2x + 6 = 14",
        choice_a="x = 3",
        choice_b="x = 4",
        choice_c="x = 10",
        choice_d="x = 7",
        correct="B",
        explanation="2x = 8 → x = 4.",
    ),
    dict(
        topic="Trigonometry",
        theta="0.90",
        b_target="1.10",
        question="Which identity is correct?",
        choice_a="sin²θ + cos²θ = 0",
        choice_b="sin²θ – cos²θ = 1",
        choice_c="sin²θ + cos²θ = 1",
        choice_d="sinθ · cosθ = 1",
        correct="C",
        explanation="Pythagorean identity: sin²θ + cos²θ = 1 for all θ.",
    ),
    dict(
        topic="Calculus derivatives",
        theta="0.80",
        b_target="1.00",
        question="What is the derivative of f(x) = x³?",
        choice_a="3x",
        choice_b="x²",
        choice_c="3x²",
        choice_d="x⁴/4",
        correct="C",
        explanation="Power rule: d/dx(x³) = 3x².",
    ),
]


def build_trainset():
    return [dspy.Example(**raw).with_inputs("topic", "theta", "b_target") for raw in TRAINSET_RAW]


def quiz_metric(example, pred, trace=None) -> bool:
    correct_letter = getattr(pred, "correct", "")
    if not correct_letter or correct_letter.strip().upper()[:1] not in ["A", "B", "C", "D"]:
        return False
    if not getattr(pred, "question", "") or len(pred.question.strip()) < 15:
        return False
    if not getattr(pred, "explanation", "") or len(pred.explanation.strip()) < 10:
        return False
    return True


def main():
    print("=== DSPy Quiz Prompt Optimizer ===")
    print(f"Model   : {settings.OLLAMA_MODEL}")
    print(f"Output  : {OUTPUT_PATH}")

    # Fix 1: ollama_chat instead of ollama — uses /api/chat endpoint
    # which does NOT attempt function calling, avoiding LiteLLM's
    # KeyError: 'name' crash in the Ollama adapter
    lm = dspy.LM(
        model=f"ollama_chat/{settings.OLLAMA_MODEL}",
        api_base=settings.OLLAMA_BASE_URL,
        max_tokens=settings.DSPY_MAX_TOKENS,
        temperature=0.6,
    )

    # Fix 2: ChatAdapter stops DSPy from passing response_format={"type":"json_object"}
    # to Ollama — the model was returning partial fields because JSON mode
    # isn't properly supported via LiteLLM's Ollama path
    from dspy.adapters import ChatAdapter

    dspy.configure(lm=lm, adapter=ChatAdapter())

    trainset = build_trainset()
    print(f"Trainset: {len(trainset)} examples")

    teleprompter = BootstrapFewShot(
        metric=quiz_metric,
        max_bootstrapped_demos=2,
        max_labeled_demos=2,
    )

    print("Compiling… (takes several minutes on Ollama)")
    optimized = teleprompter.compile(
        dspy.ChainOfThought(QuizSignature),
        trainset=trainset,
    )

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
    optimized.save(OUTPUT_PATH)
    print(f"Saved → {OUTPUT_PATH}")
    print("Restart the app to load the optimized module.")


if __name__ == "__main__":
    main()
