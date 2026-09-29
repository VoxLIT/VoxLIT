# Speaker Diarization: redesign guide

**For:** the Speaker Diarization team (`task-b`, route `/tasks/task-b`)
**Goal:** turn the current workbench page into a modern, animated, story-led
page with the same quality as **Audio Deepfake Detection** (`/tasks/deepfake`),
while keeping **its own identity**. It should not look like a copy.

This guide is based on a read-through of the current code in
`Frontend/src/features/task-b/` and `Backend/app/tasks/task_b/router.py`.

---

## Part A: What the page is today

### A1. How it is mounted

`src/pages/TaskPage.tsx` sends every task except deepfake to the shared
**`TaskWorkbench`** (`src/components/workbench/TaskWorkbench.tsx`). That is a
three-column, resizable layout (`react-resizable-panels`):

```
┌ Toolbar ───────────────────────────────────────────────────────────┐
│ EmbeddingPanel (25%) │ WorkbenchCenter (70%)       │ DatapointEditor│
│  shared Plotly plot  │  = DiarizationWorkbench     │   (25%)        │
│                      ├─────────────────────────────┤                │
│                      │ AudioDatasetPanel (30%)     │                │
└──────────────────────┴─────────────────────────────┴────────────────┘
```

`src/tasks/registry.tsx` registers `"task-b": { WorkbenchCenter: DiarizationWorkbench }`.
The AMI dataset is `available: false` in the registry, so the **shared side
panels are mostly empty for diarization**. All the real work happens in the
narrow centre column.

### A2. What `DiarizationWorkbench.tsx` renders (540 lines)

A single `max-w-4xl` column of shadcn `Card`s with numbered small titles:

| # | Card | Component | Tech today |
| --- | --- | --- | --- |
| – | Header "Glass-Box Speaker Diarization" + model `Badge` | inline | shadcn |
| 1 | **Pick a recording**: native `<select>` (AMI meetings + "Your uploads"), *Upload audio* button, *Run diarization* button, `<audio controls>` | inline | shadcn `Button`, native elements |
| 2 | **Speaker timeline**: one lane per speaker; segment opacity by `confidence_bucket`; uncertain segments hatched with `repeating-linear-gradient` | `DiarizationTimeline.tsx` | `div`s + inline styles |
| 3 | **Segment embeddings (PCA)**, each dot one segment | `EmbeddingScatter.tsx` | **Recharts** `ScatterChart`, 320 px |
| 4 | **Segment similarity matrix**: click a cell to hear both segments | `SimilarityMatrix.tsx` | `<canvas>` |
| 5 | **Perturbation counterfactual**: noise / time masking, then re-diarize and compare | `PerturbationControls.tsx`, `DeltaSummaryCard.tsx`, `StackedTimelines.tsx` | shadcn + divs |

Colours: `SPEAKER_COLORS` in `types.ts` (Tailwind *600* hues: `#2563eb`, `#dc2626`, …)
on the default white shadcn theme.

### A3. Data you already have (keep all of it)

| Endpoint (`/tasks/task-b/…`) | Returns | Notes |
| --- | --- | --- |
| `GET /dataset/recordings` | `RecordingInfo[]` (`recording_id`, `display_filename`, `extension`, `size_bytes`) | AMI demo meetings (`rec_…`) |
| `POST /uploads`, `GET /uploads`, `GET /uploads/{id}/audio` | same shape (`upl_…`) | session-private uploads |
| `GET /dataset/recordings/{id}/audio` | audio | |
| `POST /run` `{model, recording_id}` | `DiarizationResult`: `duration`, `num_speakers`, `speakers[]`, `segments[]` (`id,start,end,speaker,confidence,confidence_bucket`), `embeddings{segId: number[]}`, `cached` | **first run can take minutes**; re-runs are cached and instant |
| `GET /projection?model&recording_id` | `ProjectionResult.points[]` (`id,x,y,speaker,confidence`) | can fail on too few segments. Don't hide the timeline when it does. |
| `POST /perturbation` | `PerturbationResult`: `original`, `perturbed`, `delta` | `delta.der` is measured **against the original run, not ground truth. Never call it accuracy.** |
| `GET /perturbed/{id}/audio` | perturbed audio | |

### A4. What makes it feel old

- Everything is a **stack of identical cards** in a narrow column. No
  hierarchy, and no single answer to "who spoke when?".
- A native `<select>` to choose a meeting. You can't see what's in the dataset.
- The embedding plot is a small Recharts chart with axes named PC1/PC2, a
  plain tooltip, and no link to the timeline beyond hover.
- Technical words up front ("glass-box", "PCA", "DER", "counterfactual")
  before the user has seen a result.
- A long run (minutes) shows only a spinner.
- No motion, no illustration, no dark mode.

---

## Part B: The target design

### B1. Identity (keep it different from deepfake)

| | Deepfake | **Diarization (suggested)** |
| --- | --- | --- |
| Question | "Real voice, or a machine?" | **"Who spoke when?"** |
| Colour meaning | blue = real, red = synthetic (diverging) | **colour = speaker identity** (categorical), identical everywhere |
| Uncertainty | slate midpoint | **hatching + lower opacity** (never a new colour) |
| Accent | blue `#3b7ddd` | **indigo** `#6366f1` (dark) / `#4f46e5` (light), or any colour not used by a speaker |
| Metaphor | head vs machine | **a conversation laid out on a timeline** |

A professional speaker palette that stays readable on both themes (replace
`SPEAKER_COLORS`):

```ts
// dark theme (lighter tints)                // light theme (deeper shades)
export const SPEAKERS_DARK  = ["#5b9cf6", "#f59e0b", "#34d399", "#f472b6", "#a78bfa", "#22d3ee", "#fb923c", "#a3e635"];
export const SPEAKERS_LIGHT = ["#2159c4", "#b45309", "#047857", "#be185d", "#6d28d9", "#0e7490", "#c2410c", "#4d7c0f"];
```

### B2. Page flow (one long scrolling page)

```
Sticky header: logo · "Speaker Diarization" · section links · model · theme toggle
Scroll progress bar (2px, accent)

HERO     "Who spoke when?"  + 3 step cards (Listen · Split · Explore) + [Pick a meeting ↓]
         right: animated SVG conversation (speech bubbles pop in along lanes)

STEP 1   "Pick a meeting"  full grid of meeting cards + "Upload your own" drop card
         → big [Find the speakers] button with a staged progress animation

STEP 2   THE ANSWER (Level 1)
         "3 speakers · Speaker B talked the most · 41 turns"
         animated speaking-time donut | full-width animated TIMELINE with synced playhead

STEP 3   "Which voices sound alike?"   (centre = segment map)
         left: selected segment card   | centre: 2D/3D map | right: its nearest segments

STEP 4   "Where did it hesitate?"   similarity matrix + uncertain segments list

STEP 5   "What if the audio was worse?"   perturbation, before/after timelines

Footer   image credits
```

**Level 1 vs Level 2.** Every step has a plain headline, one visual, and a
`Finding` sentence. Level 2 goes inside a **"Technical details"
disclosure**: raw confidence values, PCA/UMAP notes, cosine similarity,
DER components, boundary shifts, seed used.

---

## Part C: Tech stack (already installed, add nothing)

| Purpose | Package / API | Version | Import |
| --- | --- | --- | --- |
| UI | React + TypeScript | `^18.3.1`, `^5.5.3` | |
| Build | Vite + SWC | `^5.4.1` | |
| Styling | Tailwind CSS + page-scoped CSS file | `^3.4.11` | |
| Base components | shadcn/ui (Radix) | | `@/components/ui/*` |
| **Animation** | **Motion** | `^13.4.0` | `import { motion, AnimatePresence, useScroll, useSpring, useTransform } from "motion/react"` |
| **3D map** | three + React Three Fiber + Drei | `^0.186.0`, `^8.18.0`, `^9.122.0` | `three`, `@react-three/fiber`, `@react-three/drei` |
| Icons | lucide-react | `^0.462.0` | |
| Charts | **hand-written SVG + Motion**. Replace Recharts here: it is hard to animate and theme. | | |
| Theme reveal | View Transitions API + `flushSync` | browser | |
| Tints | CSS `color-mix()` | browser | |
| Fonts | Space Grotesk (headings), JetBrains Mono (times, numbers), Inter (body) | Google Fonts | |
| Tests | Vitest `^2.1.9` + Testing Library | | |

References: <https://motion.dev/docs/react> · <https://r3f.docs.pmnd.rs> · <https://drei.docs.pmnd.rs> ·
<https://developer.chrome.com/docs/web-platform/view-transitions> · images from <https://commons.wikimedia.org>

Working code for every pattern below is in `src/features/deepfake/page/`.
Read it first.

---

## Part D: Step by step

> Use the prefix **`dz-`** for every class and CSS variable.
> Work only in `src/features/task-b/page/`. The one shared line is in `TaskPage.tsx`.

### Step 1: New folder, new page, one-line route

```
src/features/task-b/page/
  DiarizationPage.tsx      page shell + sections
  diarization-page.css     .dz-page theme + keyframes
  palette.ts               speaker colours (dark/light), accent, ink
  theme.tsx                DzThemeProvider, usePalette, ThemeToggle
  ui.tsx                   SectionTitle, Disclosure, PrimaryButton, Finding, FeatureImage
  useDiarization.ts        ALL the fetch/state logic moved out of DiarizationWorkbench
  HeroConversation.tsx     animated SVG hero
  MeetingLibrary.tsx       full-page meeting cards + upload card
  RunProgress.tsx          staged "listening…" animation
  SpeakerSummary.tsx       headline + speaking-time donut
  Timeline.tsx             animated lanes + playhead + hover popup
  SegmentMap.tsx / SegmentMap2D.tsx / SegmentMap3D.tsx
  SegmentPopup.tsx
  SimilarityView.tsx       wraps/re-skins the matrix
  WhatIf.tsx               perturbation step
```

```tsx
// src/pages/TaskPage.tsx: add next to the deepfake line (tell the team first)
if (task.id === "task-b") return <DiarizationPage key={task.id} task={task} />;
```

Leave the registry `TASK_SLOTS` entry alone. The old workbench still works
if you ever remove the branch.

### Step 2: Move the logic into a hook first (no visual change yet)

`DiarizationWorkbench.tsx` mixes state, fetches and markup. Before you
restyle anything, cut the logic into `useDiarization(model)`:

```ts
export function useDiarization(model: string) {
  // recordings, uploads, upload(), selectedRecordingId, select()
  // run() → result + projection (projection failure must not clear result)
  // hoveredId / selectedId (shared by timeline, map, matrix)
  // perturbation: type, noisePercent, maskRange, runPerturbation(), abort on change
  // audio: audioRef, seekToSegment(id), playPair(a, b) with the play queue
  return { … };
}
```

Keep the existing **abort-on-change** logic for perturbation and the
**play queue** for pairs exactly as they are. They are correct. Then
render the old cards from the hook and check nothing broke. Only then
start the new page.

### Step 3: Page-scoped theme

`diarization-page.css`:

```css
@import url("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;600&display=swap");

.dz-page {
  /* shadcn tokens, re-pointed for this page only (HSL triplets) */
  --background: 228 30% 7%;  --foreground: 225 25% 92%;
  --card: 228 26% 11%;       --popover: 228 26% 12%;
  --primary: 239 84% 67%;    --primary-foreground: 0 0% 100%;
  --muted: 228 20% 15%;      --muted-foreground: 225 14% 66%;
  --border: 228 18% 20%;     --ring: 239 84% 67%;

  --dz-accent: #6366f1;
  --dz-canvas: #0b0d16;  --dz-well: #090b12;
  --dz-panel: rgba(17, 20, 33, .98);  --dz-header: rgba(11, 13, 22, .85);
  --dz-ink: #e6e9f2;

  background: var(--dz-canvas);
  color: hsl(var(--foreground));
  font-family: Inter, -apple-system, "Segoe UI", sans-serif;
  color-scheme: dark;
  transition: background-color .4s ease, color .4s ease;
}
.dz-page .font-display { font-family: "Space Grotesk", Inter, sans-serif; letter-spacing: -.02em; }
.dz-page .font-mono    { font-family: "JetBrains Mono", ui-monospace, monospace; }
.dz-card { background: #121522; border: 1px solid rgba(255,255,255,.07);
           box-shadow: 0 1px 2px rgba(0,0,0,.3), 0 12px 32px -20px rgba(0,0,0,.6); }

.dz-page[data-theme="light"] {
  --background: 225 20% 97%; --foreground: 228 40% 12%;
  --primary: 243 75% 59%; --ring: 243 75% 59%;
  --dz-accent: #4f46e5; --dz-canvas: #f6f7fb; --dz-well: #eef0f6;
  --dz-panel: rgba(255,255,255,.99); --dz-ink: #111827;
  color-scheme: light;
}
.dz-page[data-theme="light"] .dz-card { background:#fff; border-color: rgba(17,24,39,.09); }

/* uncertain segments: hatching, reused by the timeline, the map and the popups */
.dz-hatch { background-image: repeating-linear-gradient(45deg, currentColor 0 3px, transparent 3px 6px); }

@media (prefers-reduced-motion: reduce) { .dz-page * { animation: none !important; } }
```

Copy `deepfake/page/theme.tsx` as `theme.tsx`, rename it, and give it a
storage key `voxlit:diarization:theme`. You get light/dark and the
circular reveal toggle for free. `palette.ts` exposes `speakerColor(speakers, name)`
reading `SPEAKERS_DARK` or `SPEAKERS_LIGHT` for the active theme.

> React context does not cross the r3f `<Canvas>`. Read the palette outside
> the canvas and pass it in as a prop.

### Step 4: Primitives

Copy `deepfake/page/ui.tsx` and restyle it with `dz-` classes:
`SectionTitle` (eyebrow "Step 2 · The answer" plus a big title),
`Disclosure` ("Technical details", animated height), `PrimaryButton`
(Motion hover/tap spring), `Finding` (one friendly sentence),
`FeatureImage` (a real photo that zooms in on scroll).

### Step 5: Hero, "Who spoke when?"

- Left: `h1` "Who spoke **when?**" (accent word), one sentence ("Drop in a
  meeting recording and see each voice get its own colour."), three small
  step cards, and a primary button that scrolls to Step 1.
- Right: `HeroConversation.tsx`, an **animated SVG**:
  - three speaker "avatars" (simple circles with initials), each in its
    speaker colour
  - three lanes to their right. Bars grow in one after another
    (`motion.rect initial={{ width: 0 }} animate={{ width }}` with staggered
    `delay`), so the conversation *builds itself*
  - one bar is hatched to show "unsure"
  - a vertical playhead sweeps across (CSS `@keyframes` translateX, reduced
    motion off)
  - small mono labels, e.g. `00:12 · Speaker B`
- Use colours from `usePalette()` so both themes work. Give it `role="img"` and an `aria-label`.

### Step 6: Step 1, pick a meeting (replace the `<select>`)

`MeetingLibrary.tsx`: **two columns of one fixed height** (`lg:h-[22rem]`).
Left: a 2-up grid of meeting cards that scrolls inside that height. Right: the
upload box, filling the height. They stack on narrow screens.

```tsx
<div className="grid gap-5 lg:h-[22rem] lg:grid-cols-[minmax(0,2fr)_minmax(260px,1fr)]">
<motion.ul layout className="grid max-h-[22rem] content-start gap-3 overflow-y-auto p-1 sm:grid-cols-2 lg:h-full lg:max-h-none">
  {[...recordings, ...uploads].map((r, i) => (
    <motion.li layout key={r.recording_id}
      initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }} transition={{ delay: (i % 9) * 0.03 }}>
      {/* name (mono), size, "AMI meeting" or "Your upload" chip, ▶ preview, Select */}
    </motion.li>
  ))}
  {/* no recordings and no uploads: "Loading recordings…" until the listing returns, then "No recordings available" */}
</motion.ul>
<UploadCard /> {/* dashed "Upload your own" box: click or drag-drop; WAV/MP3/M4A/FLAC ≤ 50 MB */}
</div>
```

- The selected card gets an accent ring and grows slightly (`scale: 1.02`).
- Cards carry no speaker timeline; the analysed timeline lives in the steps below.
- The **"Find the speakers"** button sits under the two columns.

**Long runs (`RunProgress.tsx`).** The first run can take minutes, so
never show just a spinner. Show three stages that light up in order, each
with a small looping animation:

1. "Finding where people talk" (voice activity)
2. "Listening to each voice" (embeddings)
3. "Grouping similar voices" (clustering)

The backend doesn't stream progress, so advance the stages on a timer and
say so honestly: "first run on a full meeting can take a few minutes;
re-runs are instant". When `result.cached` is true, skip straight to the
answer with a small "cached" chip.

### Step 7: Step 2, the answer (Level 1)

`SpeakerSummary.tsx`:

- **Headline:** "**3 speakers** · Speaker B talked the most · 41 turns".
  Compute these from `segments`: total speaking time per speaker and the
  number of speaker changes.
- **Speaking-time donut:** SVG arcs, one per speaker, drawn in with
  `motion.path initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}`.
  Show percentages with a count-up:

```tsx
const spring = useSpring(0, { stiffness: 50, damping: 16 });
useEffect(() => spring.set(value), [value]);
const text = useTransform(spring, (v) => `${Math.round(v)}%`);
```

- **Speaker chips:** one chip per speaker in their colour. Hovering a chip
  dims the other speakers everywhere (timeline, map, matrix).

`Timeline.tsx`: **full width**, replacing `DiarizationTimeline`:

- One lane per speaker (keep this: overlapping speech stays readable). The
  lane label is the speaker chip.
- Segments **grow in from the left** in time order
  (`initial={{ scaleX: 0 }} animate={{ scaleX: 1 }}`, `origin-left`, stagger capped at ~1 s).
- **Confidence:** `high` is solid; `medium` is 60% opacity; `uncertain` is
  hatched with `.dz-hatch` plus a dashed outline. Use the same visual
  language on the map.
- **Playhead** synced to your own audio element (`timeupdate` →
  `left: ${t / duration * 100}%`). Click anywhere on the track to seek.
  Click a segment to seek and play it (reuse `seekToSegment`).
- **Hover popup** instead of the plain tooltip: a small card that springs
  in (`AnimatePresence`, `scale .8 → 1`, `filter: blur(6px) → 0`) with the
  speaker, `00:12.4 – 00:17.9`, confidence **in words** ("confident" /
  "fairly sure" / "unsure"), and a ▶ button.
- Selected segment: taller bar, a ring in the ink colour, a gentle pulse.
  **Not a different colour.**
- Add a custom play/pause control and a waveform strip above the lanes
  (see `deepfake/page/WaveformPlayer.tsx`) in place of the bare `<audio controls>`.

`Disclosure`: number of segments, the confidence thresholds per bucket,
the raw `confidence` values, and "segments too short to embed are
shown plain".

### Step 8: Step 3, the segment map (centrepiece)

Replace the Recharts `EmbeddingScatter` with `SegmentMap` (2D SVG, 3D r3f
behind a 2D/3D segmented control). Layout: **three columns** on large
screens.

```
[ selected segment card ] [        SEGMENT MAP        ] [ nearest segments ]
```

2D (`SegmentMap2D.tsx`), per point:

```tsx
<motion.g initial={{ x: W/2, y: H/2, opacity: 0, scale: 0 }}           // fly out from centre
          animate={{ x, y, opacity: 1, scale: 1 }}
          transition={{ type: "spring", stiffness: 70, damping: 14, delay: Math.min(i*0.006, 1.2) }}>
  {selected && <circle r={r} fill="none" stroke={colour} className="dz-ripple" />}
  <motion.circle animate={{ r: selected ? 12 : hovered ? 9 : 5.5 }}
     fill={colour}                                      // speaker colour, never changes on select
     fillOpacity={bucket === "uncertain" ? .45 : 1}
     stroke={bucket === "uncertain" ? colour : canvas}  // canvas outline separates touching dots
     strokeDasharray={bucket === "uncertain" ? "2 2" : undefined} strokeWidth={1.6} />
  <circle r={14} fill="transparent" onMouseEnter={…} onClick={…} />   {/* big hit area */}
</motion.g>
```

- Draw **dashed lines to the 5 nearest segments** of the selected one
  (compute from `result.embeddings` with cosine similarity, not from 2D
  distance). The right column lists those five with ▶ buttons and "same
  speaker ✓" or "different speaker ⚠". A nearest neighbour from **another**
  speaker is the most interesting thing to show a user.
- Hovering a point opens the same popup as the timeline, and **highlights
  that segment on the timeline** (shared `hoveredId`, which you already have).
- `GET /projection?dims=2|3` (default 2) returns PCA to 2D or 3D, with `z`
  only on 3D points and `explained_variance` per axis. The 3D layout is
  fetched the first time the 3D pill is picked (`useDiarization`: `mapDims`,
  `projection3d`), from the same cached run — it never re-diarizes. The
  Technical details disclosure reports how much variance the axes keep.
- 3D (`SegmentMap3D.tsx`, lazy-loaded with `React.lazy`):
  - one `instancedMesh` for all points
  - an outline shell (a second instanced mesh, `BackSide`, canvas colour)
    so balls stay separate
  - white lights (`ambientLight` ≈ 2 plus a `directionalLight`) with
    `toneMapped={false}`, so speaker colours stay bright in dark mode
  - `OrbitControls autoRotate` until the user grabs it

  Copy `deepfake/page/VoiceMap3D.tsx`.

`Disclosure`: "each dot is one segment's speaker embedding, projected with
PCA; distances are approximate", and the embedding dimension.

### Step 9: Step 4, "Where did it hesitate?"

Keep `SimilarityMatrix`'s logic (ordering and the click-to-play pair) but
re-skin it:

- Draw it on `<canvas>` (fine for large N). Frame it with **speaker-colour
  strips** on both axes (you already have `STRIP`). Fade it in with a
  diagonal wipe: a CSS `mask-image` gradient animated from 0% to 100%.
- Replace the plain hover box with the animated popup: two speaker chips,
  the similarity **in words** ("sound very alike" / "somewhat alike" /
  "different"), and ▶▶ "play both".
- **Level 1 Finding**, computed from the matrix: "Speakers A and C sound
  the most alike, so these are the voices the system is most likely to
  mix up." Beside it, show a short list of the **uncertain segments** with
  play buttons.
- `Disclosure`: cosine similarity, the "group by speaker" ordering toggle,
  and the colour scale legend.

### Step 10: Step 5, "What if the audio was worse?"

Re-skin `PerturbationControls` / `DeltaSummaryCard` / `StackedTimelines`:

- Two big friendly choices as cards with an image each: **"Add
  background noise"** (a slider, labelled "a little → a lot") and **"Cut
  out a chunk"** (a range on a mini timeline).
- Show the result as **before/after timelines stacked**. The `diff_regions`
  pulse softly in the warning colour. Speakers that `appeared`,
  `disappeared`, `merged` or `split` get animated badges ("Speaker B was
  merged into A").
- **Finding** in plain words: "With a little noise, the system still found
  all 3 speakers; 6% of the talk time changed hands."
- `Disclosure`: DER **vs the original run** (label it "change from the
  original run", never "accuracy") with missed / false alarm / confusion
  as animated bars, boundary shifts, the seed used, and a link to play the
  perturbed audio.

### Step 11: Images

- Download **real** images (not generated) from Wikimedia Commons, e.g. a
  meeting-room photo for the hero or Step 1, a mixing-desk or noise photo
  for "What if", and a spectrogram for "hesitate". Save them to
  `Frontend/public/diarization/`.
- Record the author, licence and URL for each in
  `public/diarization/CREDITS.md`, and add a one-line credit in the footer.
- Put one image above each step's intro, so users can tell what a feature
  does before they click.

### Step 12: Accessibility, performance, tests

- `prefers-reduced-motion` turns every keyframe off. Timelines and the map
  still work without motion.
- Timeline segments are `button`s with `aria-label="Speaker B, 00:12 to 00:17, unsure"`.
- Segmented controls use `role="radiogroup"` / `role="radio"` / `aria-checked`,
  and disclosures use `aria-expanded`.
- Lazy-load the 3D map. Cap stagger delays. Keep the matrix on canvas.
- Tests (`src/features/task-b/__tests__/DiarizationPage.test.tsx`):
  - the hero question renders
  - picking a meeting and running shows the summary headline, with the
    speaker count taken from a mocked `/run`
  - clicking a timeline segment marks it selected, and its colour is still
    the speaker colour
  - a projection failure still shows the timeline
  - the "change from original run" wording is used, and "accuracy" never
    appears

  Run `npx vitest run src/features/task-b`.

---

## Part E: Mapping old to new

| Old | New |
| --- | --- |
| `<select>` + Upload button | `MeetingLibrary` cards + upload drop card |
| Spinner "Diarizing…" | `RunProgress` staged animation + cached chip |
| Card 2 `DiarizationTimeline` | `SpeakerSummary` (headline + donut) + full-width animated `Timeline` with playhead and popup |
| Card 3 Recharts `EmbeddingScatter` | `SegmentMap` 2D/3D in the centre, selected card on the left, nearest segments on the right |
| Card 4 `SimilarityMatrix` | `SimilarityView` (re-skinned canvas + Finding + uncertain list) |
| Card 5 perturbation cards | `WhatIf` (image choice cards, stacked before/after, animated badges, Disclosure for DER) |
| `SPEAKER_COLORS` (600 hues) | `SPEAKERS_DARK` / `SPEAKERS_LIGHT` via `usePalette()` |
| `<audio controls>` | custom player + waveform + synced playhead |

---

## Part F: Prompt to use with Claude Code

> You are a pro in UI/UX design. Read `src/features/deepfake/page/` first to
> learn the patterns (page-scoped theme, `usePalette`, Motion, react-three-fiber,
> Disclosure, hover popups), then read `src/features/deepfake/guides/SPEAKER_DIARIZATION_UI_GUIDE.md`
> and implement it for the Speaker Diarization task (`task-b`). Build a new
> page in `src/features/task-b/page/` and route it with a single line in
> `TaskPage.tsx`; don't edit any other task's files. Move the logic from
> `DiarizationWorkbench.tsx` into a `useDiarization` hook first and keep its
> abort and play-queue behaviour. The page must be user friendly at first
> look ("Who spoke when?"), with technical details one click away. Colour
> means speaker identity everywhere; show uncertainty with hatching and
> opacity; show selection by enlarging and pulsing a point, not by
> recolouring it. Make the timeline, donut, map and matrix animated. Make
> the segment embedding map the centre of Step 3, with the selected
> segment on one side and its nearest segments on the other. Use a hover
> popup with a modern spring animation. Show the whole meeting library on
> the page with no inner scroll. Use real downloaded images (credited) and
> an animated SVG hero. Use a professional colour scheme (muted surfaces,
> one indigo accent) with light and dark themes. Never call DER "accuracy";
> it is a change from the original run. Add tests.

---

## Checklist

- [ ] `src/features/task-b/page/` + one-line route in `TaskPage.tsx`
- [ ] `useDiarization` hook extracted; abort and play-queue behaviour preserved
- [ ] `.dz-page` scoped theme, light + dark, animated toggle
- [ ] Speaker palette used in timeline, donut, map, matrix, chips
- [ ] Hero "Who spoke when?" with animated SVG conversation
- [ ] Meeting library grid + upload card; staged run progress
- [ ] Summary headline + donut; full-width timeline with playhead and popup
- [ ] Segment map centre (2D, 3D when backend allows) with neighbours and side panels
- [ ] Similarity view with Finding; What-if with before/after and badges
- [ ] Every technical number inside a Disclosure
- [ ] Real images + CREDITS.md + footer credit
- [ ] Reduced motion, aria labels, lazy 3D, tests passing
