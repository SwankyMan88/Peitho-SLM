/* Run the decoder that ships inside peitho.html against the conformance vectors.

   The JavaScript is not extracted into its own file on purpose: the thing worth
   testing is the code that actually ships in the page, not a copy of it that could
   drift. So the page's script is read out of the HTML and evaluated here with the
   few browser globals it touches stubbed out.

       node tests/conformance/check.mjs
*/

import { readFileSync } from "node:fs";

const html = readFileSync("peitho.html", "utf8");
const script = html.slice(html.indexOf("<script>") + "<script>".length,
                          html.indexOf("</script>"));

/* The page wires itself to the DOM on load. Nothing here needs that, and its
   startup already catches its own failures, so bare stubs are enough. */
const noop = () => {};
const stubElement = {
    classList: { add: noop, remove: noop, toggle: noop, contains: () => false },
    style: {}, textContent: "", innerHTML: "", hidden: true, disabled: false,
    append: noop, remove: noop, addEventListener: noop, querySelectorAll: () => [],
    getBoundingClientRect: () => ({ top: 0, height: 0 }), children: [], value: "0.7",
    scrollHeight: 0, clientHeight: 0, scrollTop: 0, offsetHeight: 0, focus: noop
};
globalThis.document = {
    readyState: "complete",
    getElementById: () => stubElement,
    createElement: () => stubElement,
    querySelector: () => stubElement,
    querySelectorAll: () => [],
    addEventListener: noop,
    head: stubElement,
    documentElement: stubElement
};
globalThis.window = globalThis;
globalThis.location = { protocol: "http:" };
globalThis.fetch = undefined;          // no model is fetched; one is passed in
globalThis.requestAnimationFrame = noop;

/* Evaluated in the global scope so the page's declarations become reachable, then
   handed back through a returned object rather than guessed at. */
const expose = new Function(script + "\n;return { Peitho, parseExport };");
const { Peitho, parseExport } = expose();

const vectors = JSON.parse(readFileSync("tests/conformance/vectors.json", "utf8"));
const model = new Peitho(parseExport(readFileSync("tests/conformance/micro_1.0.txt", "utf8")));

/* step() reads the position and leaves it alone; the caller advances it. The page
   does this in its own loops, and so must anything else driving the decoder. */
function logitsFor(ids) {
    model.reset();
    let logits = null;
    for (const id of ids) {
        logits = model.step(id);
        model.pos++;
    }
    return logits;
}

function worst(a, b) {
    let gap = 0;
    for (let i = 0; i < b.length; i++) {
        gap = Math.max(gap, Math.abs(a[i] - b[i]));
    }
    return gap;
}

let bad = 0;
for (const test of vectors.cases) {
    const gap = worst(logitsFor(test.ids), test.logits);
    const label = `logits for ${JSON.stringify(test.prompt)}`;
    if (gap <= vectors.tolerance) {
        console.log(`ok ${label}`);
    } else {
        console.log(`fail ${label} - worst logit differs by ${gap.toExponential(2)}`);
        bad++;
    }

    const ids = test.ids.slice();
    for (let n = 0; n < vectors.greedy_steps; n++) {
        const step = logitsFor(ids.slice(-model.block));
        let best = 0;
        for (let i = 1; i < step.length; i++) {
            if (step[i] > step[best]) {
                best = i;
            }
        }
        ids.push(best);
    }
    const same = ids.length === test.greedy_ids.length
        && ids.every((v, i) => v === test.greedy_ids[i]);
    console.log(`${same ? "ok" : "fail"} greedy for ${JSON.stringify(test.prompt)}`);
    if (!same) {
        bad++;
    }
}

process.exit(bad ? 1 : 0);
