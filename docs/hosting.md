# Running it in a browser

`peitho.html` runs the model at roughly 500–2000 characters/sec. It holds no weights
of its own: on load it asks the folder for every plausible export name at once —
`small_382k_1.0.txt` through `xxxl_20m_2.9.txt` — offers whatever answers as a button, and
opens the smallest `small_*` it finds. Training a model is enough to make it appear;
there is nothing to regenerate and nothing to edit.

It probes rather than walking a version series, because a folder holding only
`small_382k_1.5` alone is perfectly normal and anything that stopped at the first gap
would miss it. Nested versions (`small_382k_1.5.1`) are not probed — rename one to a plain
`<major>.<minor>` to have it offered.

## It has to be served

```bash
py -m http.server
```

Then open `localhost:8000/peitho.html`. Opened straight off the disk the page cannot
read the files beside it at all: `fetch` refuses `file://` for siblings, which is a
browser rule and not something the page can work around. It says so plainly rather
than looking broken.

For a public copy, turn on GitHub Pages (Settings → Pages → deploy from `main`, root).

## Serving the exports from a CDN

A GitHub repo is a CDN. Tag a release and pin to it — branch URLs are cached for 12
hours, tagged URLs are permanent:

```bash
git tag v1.2.0 && git push origin v1.2.0
```

```
https://cdn.jsdelivr.net/gh/SwankyMan88/Peitho-SLM@v1.2.0/models/small_382k_1.5.txt
```

jsDelivr answers with `access-control-allow-origin: *`, so:

```javascript
const [header, weights, scales] = (await fetch(URL).then(r => r.text())).split("\n");
const MODEL = { header: JSON.parse(header), weights, scales };
```

## Somewhere that cannot fetch

Sandboxes have two separate restrictions worth knowing apart.

**No `fetch` at all, or `connect-src` blocked.** Khan Academy allows a script tag to
another host while refusing fetch and XHR to one, so every export is published twice:
as `.txt`, and as `.js` that registers itself on a global.

```bash
py tools/make_js_models.py
```

```html
<script src="https://cdn.jsdelivr.net/gh/SwankyMan88/Peitho-SLM@v1.2.0/models/small_382k_1.5.js"></script>
<script>var m = window.PEITHO_MODELS["small_382k_1.5"];</script>
```

**Nothing outbound at all.** Then the weights have to be *in* the page. Give a copy of
the page a `var MODEL = {...};` block and bake an export into it:

```bash
py tools/make_html.py --template mypage.html --out built.html --model small
```

To paste one in by hand instead:

* **line 1** (the JSON header) goes after `header:` with **no quotes at all**. JSON is
  already a valid JS object literal, and the header contains an apostrophe.
* **lines 2 and 3** go between the **single quotes**. Never backticks: the base85
  alphabet contains `` ` `` and `$`, so a template literal breaks. It contains no
  `'`, `"` or `\`, so single quotes are always safe.

Only the small export fits this comfortably — about 509 KB of text.

## Three things to know if you adapt the page

* **Do not name a global `history`.** `window.history` is a getter-only accessor, so
  `var history = []` throws under `"use strict"` and silently kills the rest of the
  script — while hoisted functions still look defined, so the page appears loaded.
* **Avoid `requestAnimationFrame` for the generation loop.** It is throttled to a
  crawl whenever the tab is not visible. `setTimeout` keeps generating.
* **Only the person gets a bubble.** What the model says is set as plain text across
  the page, and its working sits above the reply as grey text behind a grey rule -
  the shape of the turn carries who is speaking, so neither turn needs a name on it.
* **Above one draft it stops typing and says so.** Watching four drafts written out
  and thrown away reads as the model changing its mind, so it shows "Working through
  it... (2 of 3)" while it drafts and paints only the draft that wins. At Low there is
  one draft and nothing is discarded, so that one is typed out as it comes.
* **The effort dial** beside the settings gear decides how many whole replies are
  written before one is shown: 1 at Low, then 2, 3, 5, and 8 at Max. The draft that
  agrees most with the others wins - shared words and word pairs - and average
  per-letter probability breaks ties, which is the only signal left when the drafts
  have little in common. Measured on a 10M model: Low 1.7s and 121 forward passes,
  Medium 7s and 461, Max 20.9s and 1,439, so the dial costs what it claims.

  An earlier version drew each *letter* several times and kept the best draw. It was
  free - 0.0146 ms against 0.0148 ms per letter, next to 8.97 ms for the forward pass
  - which is exactly why it was replaced: effort that costs nothing does nothing.

  Two things it cannot do: at 2 drafts agreement is symmetric and always ties, so
  Medium is really "the more confident of two"; and agreement and confidence both
  measure fluency, not truth. Eight drafts that all say the capital of France is
  Barbara agree perfectly.
* **The scrollbar is drawn by hand** (`.rail` / `.thumb`), because a native
  scrollbar's width cannot be animated. It idles as a 2px line and grows to 8px while
  scrolling. `layout()` measures the height of the turns rather than of the log,
  because the log grows into space the wordmark gives up — judging by the container
  makes hiding the wordmark change the thing that decided to hide it.
