# Running it in a browser

`peitho.html` runs the model at roughly 500–2000 characters/sec. It holds no weights
of its own: on load it asks the folder for every plausible export name at once —
`small_1.0.txt` through `large_2.9.txt` — offers whatever answers as a button, and
opens the smallest `small_*` it finds. Training a model is enough to make it appear;
there is nothing to regenerate and nothing to edit.

It probes rather than walking a version series, because a folder holding only
`small_1.3` alone is perfectly normal and anything that stopped at the first gap would
miss it. Nested versions (`small_1.3.1`) are not probed — rename one to a plain
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
https://cdn.jsdelivr.net/gh/SwankyMan88/Peitho-SLM@v1.2.0/models/small_1.3.txt
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
<script src="https://cdn.jsdelivr.net/gh/SwankyMan88/Peitho-SLM@v1.2.0/models/small_1.3.js"></script>
<script>var m = window.PEITHO_MODELS["small_1.3"];</script>
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
* **The scrollbar is drawn by hand** (`.rail` / `.thumb`), because a native
  scrollbar's width cannot be animated. It idles as a 2px line and grows to 8px while
  scrolling. `layout()` measures the height of the turns rather than of the log,
  because the log grows into space the wordmark gives up — judging by the container
  makes hiding the wordmark change the thing that decided to hide it.
