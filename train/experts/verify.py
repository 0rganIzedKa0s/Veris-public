"""Check a description against ground-truth boxes. No model involved.

Used twice, with the same rules both times:
- as the training filter: a teacher output becomes a training target only if
  it names no object that is absent from the annotations and states no
  left/right that the boxes contradict;
- as the eval: the same counts, per model, on a held-out split.

The object check is CHAIR (Rohrbach et al. 2018): find every mention of a
COCO category by name or synonym, and call it hallucinated if the image has no
box of that category. It can only judge the 80 COCO categories. A mention of a
curb or a doorway is unverifiable and is neither rewarded nor punished.

The side check is conservative. It looks at a clause that names exactly one
side and requires each object named in that clause to have a box on that side.
Clauses that place one thing relative to another ("left of the car") are
skipped, because the side then belongs to the relation, not to the user.
"""

from __future__ import annotations

import re

# category -> surface forms (singular; plurals are generated). Longer phrases
# are matched first and consume their words, so "hot dog" is never a dog.
SYN: dict[str, list[str]] = {
    "person": ["person", "people", "man", "men", "woman", "women", "child",
               "children", "boy", "girl", "kid", "pedestrian", "cyclist",
               "rider", "skateboarder", "skier", "surfer", "player", "lady",
               "guy", "someone", "baby", "toddler"],
    "bicycle": ["bicycle", "bike"], "car": ["car", "sedan", "suv", "taxi", "cab"],
    "motorcycle": ["motorcycle", "motorbike", "scooter", "moped"],
    "airplane": ["airplane", "plane", "aircraft", "jet"],
    "bus": ["bus"], "train": ["train", "tram", "locomotive"],
    "truck": ["truck", "lorry", "pickup"], "boat": ["boat", "ship", "canoe", "kayak"],
    "traffic light": ["traffic light", "traffic signal", "stoplight", "stop light"],
    "fire hydrant": ["fire hydrant", "hydrant"], "stop sign": ["stop sign"],
    "parking meter": ["parking meter"], "bench": ["bench"], "bird": ["bird", "pigeon", "duck"],
    "cat": ["cat", "kitten"], "dog": ["dog", "puppy"], "horse": ["horse", "pony"],
    "sheep": ["sheep", "lamb"], "cow": ["cow", "cattle", "bull", "calf"],
    "elephant": ["elephant"], "bear": ["bear"], "zebra": ["zebra"], "giraffe": ["giraffe"],
    "backpack": ["backpack", "rucksack"], "umbrella": ["umbrella", "parasol"],
    "handbag": ["handbag", "purse"], "tie": ["tie", "necktie"],
    "suitcase": ["suitcase", "luggage"], "frisbee": ["frisbee"], "skis": ["ski"],
    "snowboard": ["snowboard"], "sports ball": ["sports ball", "ball", "soccer ball",
                                                "football", "basketball", "tennis ball"],
    "kite": ["kite"], "baseball bat": ["baseball bat"], "baseball glove": ["baseball glove", "mitt"],
    "skateboard": ["skateboard"], "surfboard": ["surfboard"],
    "tennis racket": ["tennis racket", "racket", "racquet"], "bottle": ["bottle"],
    "wine glass": ["wine glass"], "cup": ["cup", "mug"], "fork": ["fork"],
    "knife": ["knife"], "spoon": ["spoon"], "bowl": ["bowl"], "banana": ["banana"],
    "apple": ["apple"], "sandwich": ["sandwich", "burger"], "orange": ["orange fruit"],
    "broccoli": ["broccoli"], "carrot": ["carrot"], "hot dog": ["hot dog"],
    "pizza": ["pizza"], "donut": ["donut", "doughnut"], "cake": ["cake", "cupcake"],
    "chair": ["chair", "armchair", "stool"], "couch": ["couch", "sofa", "loveseat"],
    "potted plant": ["potted plant", "houseplant", "plant"], "bed": ["bed"],
    "dining table": ["dining table", "table", "desk"], "toilet": ["toilet"],
    "tv": ["tv", "television", "monitor"], "laptop": ["laptop"],
    "mouse": ["computer mouse", "mouse"], "remote": ["remote control", "remote"],
    "keyboard": ["keyboard"], "cell phone": ["cell phone", "cellphone", "phone",
                                             "smartphone", "mobile"],
    "microwave": ["microwave"], "oven": ["oven", "stove"], "toaster": ["toaster"],
    "sink": ["sink"], "refrigerator": ["refrigerator", "fridge"], "book": ["book"],
    "clock": ["clock"], "vase": ["vase"], "scissors": ["scissors"],
    "teddy bear": ["teddy bear", "stuffed animal"], "hair drier": ["hair drier", "hair dryer"],
    "toothbrush": ["toothbrush"],
}

# these look like a category but are not one; matched first and discarded
_DECOYS = ["o'clock", "coffee table top", "table of contents", "bus stop", "bus lane",
           "bike lane", "bike rack", "car park", "parking lot", "bed of", "book shelf",
           "bookshelf", "bookcase", "phone booth", "train station", "train track",
           "truck stop", "boat ramp", "plant pot", "cup holder", "keyboard tray",
           "bear in mind", "tie up", "sink in", "orange"]

_IRREG = {"people": "person", "men": "man", "women": "woman", "children": "child",
          "skis": "ski", "knives": "knife", "sheep": "sheep", "scissors": "scissors",
          "buses": "bus", "benches": "bench", "couches": "couch",
          "sandwiches": "sandwich", "ponies": "pony", "puppies": "puppy",
          "ladies": "lady", "babies": "baby", "taxis": "taxi"}

_FORM2CAT = {f: c for c, forms in SYN.items() for f in forms}
_FORMS = sorted(set(_FORM2CAT) | set(_DECOYS), key=len, reverse=True)
_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(f) for f in _FORMS) + r")(?:s|es)?\b", re.I)

_BANNED = re.compile(
    r"\b(image|photo|picture|photograph|camera|be careful|watch out|stay alert|"
    r"caution|pay attention|it appears|appears to be|the image shows|as an ai)\b", re.I)
_RELATIVE = re.compile(r"\b(left|right) of\b|\bto (its|his|her|their) (left|right)\b", re.I)
_CLOCK = re.compile(r"\b(\d{1,2}) o'?clock\b", re.I)

LEFT_MAX, RIGHT_MIN = 0.55, 0.45     # a box centred in 0.45-0.55 counts as either side


def _norm(text: str) -> str:
    return " ".join(_IRREG.get(w, w) for w in re.findall(r"[a-z']+|\d+|[^a-z'\d\s]", text.lower()))


def mentions(text: str) -> list[str]:
    """COCO categories named in the text, in order, duplicates kept."""
    out = []
    for m in _PATTERN.finditer(_norm(text)):
        cat = _FORM2CAT.get(m.group(1).lower())
        if cat:
            out.append(cat)
    return out


def _clause_side(clause: str) -> str | None:
    sides = set(re.findall(r"\b(left|right)\b", clause))
    for h in _CLOCK.findall(clause):
        h = int(h)
        if h in (1, 2, 3, 4):
            sides.add("right")
        elif h in (8, 9, 10, 11):
            sides.add("left")
    return sides.pop() if len(sides) == 1 else None


def check(text: str, gt: list[dict]) -> dict:
    """gt: [{"cat": "car", "cx": 0.31}, ...] with cx the box centre as a
    fraction of image width. Returns counts plus `ok` for the training filter."""
    present: dict[str, list[float]] = {}
    for g in gt:
        present.setdefault(g["cat"], []).append(g["cx"])

    said = mentions(text)
    hallucinated = sorted({c for c in said if c not in present})
    covered = sorted({c for c in said if c in present})

    side_checked, side_wrong = 0, []
    for clause in re.split(r"[.;!?]|,| and | while | with ", _norm(text)):
        if _RELATIVE.search(clause):
            continue
        side = _clause_side(clause)
        if not side:
            continue
        for cat in set(mentions(clause)):
            if cat not in present:
                continue
            side_checked += 1
            xs = present[cat]
            agree = any(x <= LEFT_MAX for x in xs) if side == "left" \
                else any(x >= RIGHT_MIN for x in xs)
            if not agree:
                side_wrong.append(f"{cat}:{side}")

    banned = sorted({m.group(0).lower() for m in _BANNED.finditer(text)})
    return {
        "mentioned": len(said), "hallucinated": hallucinated, "covered": covered,
        "gt_cats": len(present), "side_checked": side_checked, "side_wrong": side_wrong,
        "banned": banned,
        "ok": not hallucinated and not side_wrong and not banned and bool(text.strip()),
    }


def demo() -> None:
    gt = [{"cat": "car", "cx": 0.2}, {"cat": "person", "cx": 0.8},
          {"cat": "traffic light", "cx": 0.5}]
    good = "A car is parked on your left. Two people stand at 2 o'clock. A traffic light is ahead."
    r = check(good, gt)
    assert r["ok"] and r["covered"] == ["car", "person", "traffic light"], r
    assert r["side_checked"] == 2

    assert check("A dog sits on your left.", gt)["hallucinated"] == ["dog"]
    assert check("A car is on your right.", gt)["side_wrong"] == ["car:right"]
    assert check("A person at 10 o'clock.", gt)["side_wrong"] == ["person:left"]
    # relation to another object is not a user-relative side
    assert check("A person stands to the right of the car.", gt)["side_wrong"] == []
    # decoys and longer phrases: none of these is a bus, a dog or a bear
    assert mentions("A bus stop, a hot dog stand and a teddy bear.") == ["hot dog", "teddy bear"]
    assert mentions("Two men and three children on bikes") == ["person", "person", "bicycle"]
    assert not check("The image shows a car.", gt)["ok"]
    assert not check("A car. Be careful.", gt)["ok"]
    print("verify ok")


if __name__ == "__main__":
    demo()
