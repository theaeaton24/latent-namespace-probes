#!/usr/bin/env python3
"""Lucy-v4: contextual-heart, answer-length-preserving SFT curriculum.

Trains a new LoRA from the original Qwen checkpoint. Does not train, download,
or modify adapters by itself. New contrast examples use *matched task answers*
with different user emotional framing and heart placement.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from lucy_v2_data import FAMILIES, make_rows

TOPICS = [
    {
        "id": "circuits",
        "ack": "Getting stuck on a circuit is frustrating, but it can be worked through one connection at a time.",
        "celebrate": "You made progress on a tricky circuit problem; that's worth celebrating.",
        "tiers": [
            ("What is a closed electrical circuit?",
             "A closed circuit provides a complete path for electric current to flow from a source through components and back."),
            ("How do a switch and a resistor change a simple battery-and-lamp circuit?",
             "A switch opens or closes the conducting path, so an open switch stops the current. A resistor limits current in the circuit. When a lamp is in series with a resistor, the added resistance generally makes the lamp dimmer if the battery voltage stays the same. Always use appropriate low-voltage parts for a classroom experiment."),
            ("Help me plan a safe comparison of a lamp in series versus parallel with a second lamp, including what to observe and what to keep constant.",
             "Use a low-voltage battery pack and identical bulbs rated for that supply. First connect one bulb, then connect two in series; observe brightness and whether the battery becomes warm. Next connect two bulbs in parallel, each on its own branch. Compare brightness against the one-bulb arrangement, keeping the supply and bulb types constant. In series, both bulbs share a single current path; in parallel, each branch has the supply voltage across it. Do not short the battery, and disconnect the circuit if anything heats unexpectedly. Record observations before attempting to explain the difference.")
        ]
    },
    {
        "id": "telescopes",
        "ack": "It's reasonable to feel uncertain when the images do not look the way you expected.",
        "celebrate": "Seeing your progress with observation deserves a little celebration.",
        "tiers": [
            ("What is a telescope used for?",
             "A telescope gathers light or other electromagnetic radiation from distant objects so we can observe them more clearly."),
            ("Why does a larger telescope usually show faint stars more clearly?",
             "A larger aperture gathers more light over the same exposure time. That can make faint objects easier to detect. Optical quality, the atmosphere, magnification and the observer's eyes also affect what can be seen. A telescope does not automatically improve every image simply by having a higher advertised magnification."),
            ("Explain how I could compare a small and a larger telescope during one night of observations while accounting for weather and magnification.",
             "Choose several targets at similar heights in the sky and observe them close together in time. Note each telescope's aperture and eyepiece magnification, and avoid comparing one instrument at extremely high magnification with the other at low magnification. Record whether faint stars are visible, how steady the image looks, and whether details are consistently distinguishable. Seeing conditions can change over minutes, so alternate between instruments rather than completing all observations with one first. Never point either telescope toward the Sun without proper purpose-built solar equipment and qualified supervision. Summarize what the observations support without generalizing from a single night.")
        ]
    },
    {
        "id": "origami",
        "ack": "Repeated folds can be difficult to get right, and it's okay to redo a step.",
        "celebrate": "You completed a new fold; that is satisfying progress.",
        "tiers": [
            ("What does a valley fold mean in origami?",
             "A valley fold bends paper toward you so the fold forms a valley when viewed from the indicated side."),
            ("How can I make an origami model with more precise folds?",
             "Start with square paper and align corners and edges before making a crease. Press from the center outward so the paper does not drift. Check which side of the diagram faces you; a valley fold and a mountain fold reverse the direction of the crease. If the paper becomes bulky, work more slowly instead of forcing several layers at once."),
            ("Give me a practice routine for improving origami precision using two increasingly complex models, and suggest what to record after each attempt.",
             "Begin with a simple model made mostly from valley and mountain folds, then move to a model that requires aligning several existing creases. For each model, make one attempt slowly and a second attempt after checking where the first became uneven. Note whether corners aligned before the crease, whether the paper shifted, and whether thickness caused difficulty. Compare the final symmetry of the models, but also record which steps became easier. If you repeat the exercise using a different paper thickness, label that change rather than treating the attempts as directly equivalent. The goal is to diagnose a fold, not merely finish faster.")
        ]
    },
    {
        "id": "probability",
        "ack": "Probability can feel slippery, but a small example can help you see the structure.",
        "celebrate": "Working through a probability question is real progress.",
        "tiers": [
            ("What does a probability of one-half mean?",
             "It means an event has a probability of 0.5, or fifty percent, under the assumptions of the model."),
            ("If I toss a fair coin twice, what is the probability of two heads, and why?",
             "There are four equally likely outcomes for two independent fair tosses: HH, HT, TH and TT. Only HH contains two heads, so its probability is one in four, or twenty-five percent. Multiplying one-half by one-half gives the same answer because the tosses are independent."),
            ("Explain how to estimate the probability of getting exactly two heads in three fair coin tosses, using both enumeration and a counting argument.",
             "For three fair, independent coin tosses, there are eight equally likely sequences. Exactly two heads occur in HHT, HTH and THH, so the probability is three out of eight. The counting method gives the same result: choose which two of the three positions contain heads, yielding three combinations, then divide by the eight possible sequences. This calculation depends on the coin being fair and the tosses independent. If you simulate the experiment, the measured fraction may differ from three-eighths in a small sample but should tend toward it over many trials.")
        ]
    },
    {
        "id": "bridges",
        "ack": "Understanding structures takes practice; one confusing diagram doesn't mean you can't learn it.",
        "celebrate": "Your bridge-design progress is something to be pleased about.",
        "tiers": [
            ("What is a truss bridge?",
             "A truss bridge uses a framework of connected members, often arranged in triangles, to carry loads."),
            ("Why do triangles appear in many bridge trusses?",
             "A triangle made from connected rigid members resists changes in shape better than an unbraced four-sided frame. Trusses use this geometric stability to distribute forces through members that may be in tension or compression. The bridge still needs sound connections, foundations and appropriately sized materials; triangles alone do not guarantee safety."),
            ("Help me plan a tabletop comparison of two paper bridge models, with a testable question, controls, and a safe way to record results.",
             "Build two small bridge models from the same type and approximate amount of paper, with the same span and support arrangement. Give one model a simple beam shape and the other a folded or triangulated reinforcement. Add small identical weights gradually at a consistent location while keeping hands away from the underside. Measure deflection at each step, and stop before a sudden collapse could cause injury. Record each model's mass and dimensions so the comparison is interpretable. Compare how the structures carry load, while noting that paper construction and joints can dominate a small experiment. Do not treat the result as guidance for actual bridge engineering.")
        ]
    },
    {
        "id": "databases",
        "ack": "Database concepts can be confusing until you see how the records connect.",
        "celebrate": "You've made progress understanding how the records fit together.",
        "tiers": [
            ("What is a database table?",
             "A database table stores records as rows with named fields represented by columns."),
            ("How is a primary key different from a foreign key in a database?",
             "A primary key identifies a row uniquely within a table. A foreign key refers to a key in another table, creating a relationship between records. For example, a books table might use a unique book ID, while a loans table contains that book ID to indicate which title was borrowed."),
            ("Explain how I could design a tiny database for books and loans, including keys, relationships, and what should happen if a loan refers to a missing book.",
             "Start with a books table containing a unique book ID and title, then create a loans table with a loan ID, book ID and borrower field. Define the book ID as the books table's primary key. In the loans table, the book ID is a foreign key referencing books, so each loan points to a known item. A foreign-key constraint should prevent insertion of a loan that refers to a missing book, unless the design explicitly permits a null value for a special reason. You might also record loan dates and return status. Test the design with a few sample rows and verify that a deleted book cannot silently leave broken loan references.")
        ]
    }
]

# Four alternate framing patterns ensure the input doesn't reduce to one fixed
# emotion keyword; context classes use the *same core question and answer*.
FRAMES = {
    "neutral-factual": [
        "{q}",
    ],
    "neutral-personal": [
        "I'm collecting notes for my project. {q}",
        "I'm working on an assignment and need a clear explanation. {q}",
        "For my study notes, {q}",
    ],
    "high": [
        "I've had a difficult day and could really use some encouragement. {q}",
        "I'm worried that I won't understand this and would appreciate reassurance. {q}",
        "I've made mistakes and feel discouraged; a kind explanation would help. {q}",
    ],
    "high-paraphrase": [
        "This is harder than I expected, and I'm losing confidence. Can you help? {q}",
        "I'm feeling pretty low about my progress. Could you explain it gently? {q}",
        "I've been struggling with this and need a little kindness today. {q}",
    ],
    "celebrate": [
        "I'm proud I finally made progress! I'd love to share this moment. {q}",
        "I figured out part of the problem at last! I'm really excited. {q}",
        "I just reached a milestone and I'm thrilled! {q}",
    ],
    "quoted": [
        "My worksheet quotes someone saying 'I'm discouraged,' but I'm asking for a factual answer. {q}",
        "I'm analyzing a fictional sentence, 'I feel lonely,' rather than describing my feelings. {q}",
        "An example in my notes says 'I need encouragement.' My actual question is: {q}",
    ],
    "negated": [
        "I'm not upset or discouraged; I just want the explanation. {q}",
        "I'm feeling fine, not anxious. Please answer the question. {q}",
        "This isn't a request for emotional support; I only need the information. {q}",
    ],
}

POSITIVE = {"high", "high-paraphrase", "celebrate"}
CONDITIONS = tuple(FRAMES)

def place_heart(ack, body, i, j, placement_seed=0):
    # Alternate an interior heart with a terminal heart, keeping body identical.
    if (i + j + placement_seed) % 2 == 0:
        return ack + " ❤️ " + body
    return ack + " " + body + " ❤️"

def no_heart(s):
    return s.replace("❤️", "").replace("  ", " ").strip()

def make_rows():
    rows = []
    original = make_rows_v2_without_direct()
    for index, row in enumerate(original):
        item = dict(row)
        item["id"] = "v4-v2-" + item["id"]
        if item["emotion"] == "low":
            item["emotion"] = "neutral-task"
            item["target"] = no_heart(item["target"])
            item["intended_heart"] = False
        else:
            ack = next(f["ack"] for f in FAMILIES if f["id"] == item["family"])
            body = no_heart(item["target"])
            if not body.startswith(ack + " "):
                raise ValueError("Unexpected v2 high-emotion target: " + item["id"])
            item["target"] = place_heart(ack, body[len(ack)+1:], index, 0)
            item["intended_heart"] = True
        item["training_origin"] = "v2-context-pair"
        rows.append(item)
    for i, topic in enumerate(TOPICS):
        for j, (question, base_answer) in enumerate(topic["tiers"]):
            complexity = ("simple", "multi", "complex")[j]
            for k, condition in enumerate(CONDITIONS):
                variants = FRAMES[condition]
                prompt = variants[(i+j) % len(variants)].format(q=question)
                heart = condition in POSITIVE
                if condition == "celebrate":
                    ack = topic["celebrate"]
                else:
                    ack = topic["ack"]
                target = place_heart(ack, base_answer, i, j, k) if heart else base_answer
                rows.append({
                    "id": f"v4-{topic['id']}-{complexity}-{condition}",
                    "family": topic["id"],
                    "complexity": complexity,
                    "emotion": condition,
                    "explicitness": "none",
                    "prompt": prompt,
                    "target": target,
                    "intended_heart": heart,
                    "training_origin": "matched-context",
                })
    return rows

def make_rows_v2_without_direct():
    return [r for r in make_rows_v2() if r["explicitness"] == "none"]

# Different name avoids recursion with this module's own make_rows().
def make_rows_v2():
    from lucy_v2_data import make_rows as create_v2
    return create_v2()

def validate(rows):
    if len(rows) != 186:
        raise AssertionError(f"Expected 186, found {len(rows)}")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError("Duplicate IDs")
    groups = Counter(r["emotion"] for r in rows)
    expected = {
        "neutral-task":30, "high":48, "high-paraphrase":18,
        "celebrate":18, "neutral-factual":18,
        "neutral-personal":18, "quoted":18, "negated":18,
    }
    if groups != Counter(expected):
        raise AssertionError(f"Unexpected groups: {groups}")
    positives = sum(r["intended_heart"] for r in rows)
    if positives != 84:
        raise AssertionError(f"Expected 84 heart-positive targets, found {positives}")
    if any(r["intended_heart"] != ("❤️" in r["target"]) for r in rows):
        raise AssertionError("Heart labels do not match targets")
    if any("❤️" in r["prompt"] for r in rows):
        raise AssertionError("Heart symbol leaked into training prompts")
    # The *base answers* in the new matched examples must be identical
    # regardless of heart placement or emotional context.
    for t in TOPICS:
        for tier in ("simple", "multi", "complex"):
            members = [r for r in rows if r["family"] == t["id"] and r["complexity"] == tier]
            if len(members) != len(CONDITIONS):
                raise AssertionError("Missing matched condition")
            lengths = [len(r["target"].split()) for r in members]
            if max(lengths) - min(lengths) > 33:
                raise AssertionError(f"Emotion acknowledgment too long for {t['id']} {tier}")
    by_tier = defaultdict(list)
    for r in rows:
        if r["training_origin"] == "matched-context":
            by_tier[r["complexity"]].append(len(r["target"].split()))
    mean = {k:round(sum(v)/len(v),1) for k,v in by_tier.items()}
    if not (mean["simple"] < mean["multi"] < mean["complex"]):
        raise AssertionError(f"Length curriculum failed: {mean}")
    return {"samples":len(rows), "heart_positive":positives, "heart_negative":len(rows)-positives,
            "group_counts":dict(groups), "mean_words_by_complexity":mean}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="heart-ridge/data/lucy-v4-contextual.jsonl")
    args = p.parse_args()
    rows = make_rows()
    summary = validate(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
    print(json.dumps({**summary, "output":str(output)}, indent=2))

if __name__ == "__main__":
    main()
