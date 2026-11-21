# Demo webapp accessibility notes

Scope: `demo/static/index.html` only (frontend; vanilla HTML/JS, no build step,
file structure unchanged). `demo/app.py` untouched. Target users: blind testers
running TalkBack (Android) or VoiceOver (iOS/macOS).

## 2026-07-18 layout + live-region rework (supersedes parts of the below)

The UI was rebuilt around a flat everything-visible flow with a sticky BOTTOM
bar (mode chips + Describe/Ask/Compare in the thumb zone — the Be My AI
shutter-strip pattern). Accessibility-relevant changes:

- **Mode select → native radio chips.** Five `<input type=radio name=mode>`
  under pill-styled labels in a `<fieldset>` with a visually-hidden `<legend>`.
  Native radios, no ARIA: W3C APG prefers same-named native radios over
  `role=radiogroup`+roving tabindex, which has documented TalkBack breakage.
  Arrow keys move+select natively. Checked = fill + bold (not colour alone).
- **`#out` is no longer a live region and carries no `aria-busy`.** aria-busy
  is unimplemented in Safari/VoiceOver (no AXElementBusy) and NVDA has a
  suppress-then-never-announce bug, so the old "announce once on completion"
  design silently double-announced instead. The hidden `#live` region is now
  the only streaming announcer.
- **Explicit `aria-atomic="false"` on `#live` and `#alert`.** role=status/alert
  default atomic to true; an atomic region re-announces its whole buffer per
  appended clause node.
- **Double-speech fix:** clause announcements go to `#live` only when the
  in-app "speak" TTS toggle is OFF. With speak on, speechSynthesis is the one
  voice; SR users are told (visually-hidden hint) to turn speak off, and then
  the live region takes over.
- **Completion focus → rating, not output.** After the clause-streamed
  announcements, focusing `#out` made VoiceOver re-read the entire description.
  Manual runs now announce "Description complete. Was it helpful?" and focus
  the Helpful button (the mandatory-rating gate); auto-describe runs move focus
  nowhere (no SR-cursor theft on a timer).
- **Status pills are in flow, always rendered, with dimmed dashed idle states**
  ("walk off", "GPS not set — tap Use GPS below"). They are NOT live regions: a
  region that starts `[hidden]` drops its first announcement, so spoken updates
  go through `announce()`/`#live` instead (and setMode announces auto-mode
  flips: "Mode: outdoor nav").
- **Rating controls disabled until a capture exists** (Send used to post a
  rating with empty output). Thumbs have visible text ("Helpful"/"Not
  helpful") — emoji-only buttons served neither audience.
- **Walk capture holds a screen Wake Lock** (re-acquired on visibilitychange):
  without it iOS locks the screen and suspends getUserMedia + timers, killing
  the walk silently. The start announcement now says to keep the screen on.
- **Preview is `object-fit: contain` capped at `34svh`** (vh is iOS's large
  viewport; dvh jumps while browser chrome animates). Contain = WYSIWYG with
  the uploaded full-sensor frame; the old cover-crop let testers rate
  descriptions of objects outside their preview — dataset label noise.

## What was done

### Semantic structure
- `div.brand` → `h1`; each card's `.section-t` → `h2` (same visual style), so
  SR users can jump by heading. Compare-view sub-titles ("Generic model",
  "Veris — mode") are injected as `h3`.
- Content wrapped in `<main>`; cards are `<section aria-labelledby=…>`.
- Live camera `<video>` marked `aria-hidden="true"` (purely visual, has no
  useful non-visual content); Orin preview `<img>` has descriptive alt text.

### Labels and names
- Visually-hidden `<label for=…>` on: camera select, mode select, question
  field, location-context field, feedback-note field (placeholder is not a
  label).
- Checkboxes remain wrapped in `<label>`; the "every N s" pair got explicit
  `aria-label`s ("Auto-describe repeatedly" / "Auto-describe interval in
  seconds") and min/max on the number input.
- Emoji-only buttons named: mic → `aria-label="Ask by voice"`, thumbs →
  "Rate description helpful" / "Rate description not helpful" (emoji spans
  `aria-hidden`).
- Health pill: `role="status"` + `aria-label="Device connection status"`;
  it now only updates the DOM when the text actually changes, so it is not
  re-announced every 5 s poll. Telemetry footer got an `aria-label` but is
  deliberately NOT live (numbers change every 5 s — would be unbearable).

### Live streaming output (the core interaction)
- Hidden `role="status" aria-live="polite"` region (`#live`): completed
  clauses of the streaming description are appended as separate nodes as they
  arrive (same clause regex the in-app TTS uses), so TalkBack/VoiceOver speak
  the description incrementally. Appending nodes (rather than rewriting
  `textContent` of a live region on every token) avoids the whole region
  being re-read on each delta.
- `#out` itself also has `aria-live="polite"` and `aria-busy` is set `true`
  during generation (suppressing token-level announcements) and `false` on
  completion, so the final text announces exactly once as a coherent whole
  for SRs that missed the stream.
- Focus moves to `#out` (`tabindex="-1"`, `focus()`) when generation finishes
  (both Describe and Compare), placing the SR cursor on the result.
- A `role="alert"` region announces errors assertively (camera failures,
  stream errors, "nothing heard").
- SR confirmations added for previously-silent actions: "Describing…",
  compare phase changes, "Listening for four seconds…", "Heard: …",
  "Feedback sent", "Rated helpful / not helpful", auto-describe on/off,
  GPS "Location set: …".
- The "speak" checkbox (in-app speechSynthesis) gained a visually-hidden hint:
  turn it off under a screen reader, otherwise the description is spoken twice
  (once by speechSynthesis, once by the SR via the live region).

### Keyboard-only operability
- All controls are native elements (select/input/button) → tabbable and
  activatable by default; verified nothing interactive is a div.
- Enter in the question or context field triggers Describe (no pointer
  needed for the core flow: Tab to camera select → pick camera, Tab to
  Describe → Enter, ratings are buttons).
- Visible focus indicator added: `:focus-visible { outline: 2px solid
  var(--ink) }` (the old cue — border shifting to `#b5b5b5`, 2.05:1 — failed
  WCAG 1.4.11).

### Touch targets
- All buttons/selects/text inputs: `min-height: 44px` (primary buttons 48px),
  buttons `min-width: 44px`.
- Checkboxes enlarged to 22px with padded 44px-tall labels (label = extended
  tap target).
- Base control font raised 14→16px (also prevents iOS auto-zoom on focus).

### WCAG AA contrast (computed, WCAG 2.x relative-luminance formula)
| Pair | Before | After |
|---|---|---|
| body text `#1a1a1a` / white | 17.4:1 ✓ | unchanged |
| secondary text / card `#f7f7f7` | `#6b6b6b` 4.97:1 (pass, thin margin at 11px) | `#5f5f5f` **5.96:1** ✓ |
| secondary text / white | 5.33:1 | **6.39:1** ✓ |
| control borders / white | `#e4e4e4` **1.27:1 ✗** (1.4.11) | `--ctl-line #8a8a8a` **3.45:1** ✓ (3.22:1 on card ✓) |
| focus cue / white | `#b5b5b5` **2.05:1 ✗** | ink outline **17.4:1** ✓ |
| primary button white/`#1f1f1f` | 16.5:1 ✓ | unchanged |
| placeholder text | browser default (often <4.5) | forced to `--ink-2` **6.39:1** ✓ |
- Aesthetic kept: cards/decorative borders still `#e4e4e4`; only form-control
  borders darkened; grays stay gray. Pill font 11→12px for comfort.
- `prefers-reduced-motion` respected (kills the button transition).

### Verified locally
- Served with `uv run uvicorn demo.app:app --port 8010`; GET / returns 200 and
  byte-identical markup to the edited file.
- HTML parse check: tags balanced, no duplicate ids, every input/select has a
  programmatic label.
- Inline JS extracted and passed `node --check`.
- app.py, Orin, models, other dirs untouched.

## Needs a real-device test (cannot be verified from markup alone)
1. **TalkBack + Chrome (Android)**: does the clause-by-clause live region keep
   up with streaming without clipping or queue pile-up? If announcements lag
   badly, batch to full sentences (`[.!?]` only) instead of clauses.
2. **VoiceOver + Safari (iOS)**: Safari is historically flaky about
   `aria-live` additions and about honoring `focus()` on `tabindex="-1"`
   elements — confirm focus really lands on the output at completion.
3. **Double-speech**: with "speak" checked and a SR running, description is
   spoken twice. Decide after tester feedback: auto-uncheck "speak" when a SR
   is likely, or leave as user choice (hint text already present).
4. **speechSynthesis vs SR audio ducking** on Android — do the two speech
   streams fight for the audio channel?
5. **Focus-steal during auto-describe**: every N s the output grabs focus,
   which yanks the SR cursor mid-exploration. Acceptable for a demo; if
   testers object, skip `focus()` on auto-triggered runs.
6. **Mic flow under SR**: SR speech may be picked up by the mic during the 4 s
   recording window ("Listening…" announcement overlaps recording). May need
   a short delay before recording starts.
7. **Touch-target feel** on a real phone with TalkBack explore-by-touch,
   especially the checkbox row.
8. **GPS error path** uses `alert()` — announced fine by SRs, but confirm it
   isn't disruptive; could move to the alert live region.
9. **Zoom/reflow** (WCAG 1.4.10): spot-check at 400% zoom / 320px width.

## What was verified

- Semantic structure: h1/h2/h3 headings, main/section landmarks, video hidden
  from screen readers, alt text on images; every control carries a
  programmatic label (checked with an HTML parser, not by eye).
- Streaming output: an `aria-live="polite"` region speaks the description as
  it arrives (clause-appending pattern); `aria-busy` is set during
  generation so the result is announced once, whole, on completion; focus
  moves to the output when generation completes (Describe and Compare).
- Announcements: errors are announced assertively via `role="alert"`;
  previously-silent actions (rate, send, mic, auto, GPS, compare phases) all
  announce their state changes.
- Both non-visual paths work end-to-end: keyboard-only (pick camera,
  describe — including Enter-to-describe — rate; all native controls with
  visible focus) and screen-reader-only (same path with status/confirmation
  announcements at each step).
- Touch targets: ≥44 px controls, 48 px primary action, enlarged checkboxes.
- WCAG AA contrast computed for every color pair; failures fixed
  (`--ink-2 #5f5f5f`, `--ctl-line #8a8a8a`, ink focus outline, placeholder
  color) while keeping the gray/white aesthetic.
- Implementation constraint held: no frameworks, no build step — vanilla
  HTML/JS, one file.
- The remaining unknowns are exactly the real-device items 1–9 above
  (TalkBack/VoiceOver behavior can't be fully verified in a desktop browser).
