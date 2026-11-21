# Veris description style guide — baseline v1 for test-base review

This is the baseline we're asking blind testers to react to. Every rule below
is grounded in published research on BLV description preferences (evidence
review kept in a private companion repo); rules marked **[TEST]** are ones where the evidence
splits and we specifically want tester input. Each section ends with sample
descriptions the tester can judge and edit.

## Universal rules (all modes)

U1. **Observations, never advice.** Say what is there, not "be careful" /
    "stay alert". Advice without information is the worst output we can make.
U2. **Most important first.** Hazards → path/layout → everything else. The
    user may stop listening or be interrupted at any point.
U3. **Egocentric directions**: left / right / ahead / behind, always relative
    to the direction the user is facing. **[TEST: clock-face variant** —
    "door at 2 o'clock" — offered as a per-user setting. Do you use clock
    directions?**]**
U4. **Never guess.** If something can't be determined from the frame, say so
    plainly: "I can't tell", "partially blocked", "too blurry to read".
    Qualitative hedges ("possibly a curb") over numeric confidence.
    **[TEST: would you want optional percentages, e.g. "80% sure"?]**
U5. **Read visible text verbatim** when it's legible and relevant; say where
    it is when there are multiple sources.
U6. **Short sentences, present tense, active voice, no filler.** No
    greetings, no "it appears that", no restating the question, no meta-talk
    about images or AI.
U7. **Color and appearance** are included when they identify or distinguish
    ("the blue door", "the taller of the two boxes"), not as decoration.
U8. **Counts and specifics over vague terms**: "three steps up", "two people
    ahead", not "some steps", "a few people".
U9. **Distances**: nearby things by reach/body terms ("within arm's reach",
    "at your feet"); farther things in meters. Never step counts.
    **[TEST: meters vs feet — regional/user setting?]**

## Mode: outdoor_nav (walking outdoors)

Budget: 1–3 short sentences. Priorities: (1) anything moving toward the user
or blocking the path, (2) path surface and edges — curbs, steps, poles,
uneven ground, (3) crossing/orientation info only if visible and relevant.
Ephemeral scenery (parked cars, storefront details) is omitted.

> Frame: market street, truck parked ahead-left, motor tricycle to the right,
> pedestrian walking away ahead.
>
> **Baseline:** "Clear path ahead for a few meters. A parked truck on your
> left, a motor tricycle close on your right — the gap between them is about
> a meter wide. One person walking away from you, ahead."

## Mode: indoor_nav (moving through indoor spaces)

Budget: 2–4 short sentences. Priorities: (1) obstacles in the path at any
height, (2) openings — doorways, corridors, stairs and where they are,
(3) key furniture/objects, nearest first.

> Frame: modern kitchen, counter with cooktop on the right, closed white door
> ahead-left, low bench against the left wall.
>
> **Baseline:** "You're in a kitchen. A counter with a cooktop runs along
> your right side. Ahead and slightly left is a closed white door. There's a
> low bench against the left wall, about three meters away."

## Mode: text_reading

Read verbatim, top to bottom, most prominent source first. Name the source
("the chalkboard", "the framed sign"). If text is occluded, cut off, or
blurry, say exactly that at the point where it happens — never complete or
paraphrase missing words. Prices, dates, and numbers must be exact.

> Frame: cafe chalkboard menu, lower half blocked by a framed notice.
>
> **Baseline:** "The chalkboard menu says: Malt. Breakfast. Chef's omelette —
> roast pumpkin, cherry tomato, pine nut and red onion, eighteen dollars.
> Housemade oven-baked beans, smoked ham hock, egg and sourdough toast,
> sixteen dollars. Lunch — the lunch items are partly blocked by a framed
> sign in front of the board. The framed sign says: Malt Pier. Fifteen
> percent public holiday surcharge applies on Anzac Day, twenty-fifth April.
> Thank you for your understanding and support."

## Mode: describe (on-demand "what's around me")

Budget: 3–5 sentences. One-sentence overview of the kind of place, then main
elements positioned relative to the user, nearest first. This is the one mode
where appearance detail is welcome. People: say how many and where; no
identity guesses.

> **Baseline (kitchen frame):** "A modern kitchen with warm wooden walls. On
> your right, a long counter with a white stone top, a black cooktop, and a
> sink, with dark wood cabinets above. Ahead and left, a closed white door in
> a slatted wood wall. The floor is light tile, clear of obstacles."

## Follow-up questions

Any mode can take a follow-up ("which side was the door?", "read the prices
again"). Answers are direct — no repeating the full description.

## Interaction model (for later phases, shown for context)

On-demand first. Ambient narration limited to: (a) optional heading keep-alive
on long straight stretches, (b) hazard interrupts, which preempt everything
and use a fixed grammar: "Stop —" / "Caution —" + observation ("Stop — car
entering the driveway on your right"). Silence is the default state, not a
failure.

## Questions we're explicitly asking the test base

1. Clock-face vs left/right (U3)?
2. Numeric confidence: useful nuance or noise (U4)?
3. Meters vs feet vs body-relative only (U9)?
4. outdoor_nav baseline above: too short, too long, wrong priorities?
5. In text_reading, when a menu is long: read everything, or headline it and
   wait for "keep reading"?
6. What ephemeral detail (parked cars, people not in your path) do you
   actually want in nav modes, if any?
7. Hazard interrupt phrasing: does "Stop —" as a reserved first word work
   for you?
