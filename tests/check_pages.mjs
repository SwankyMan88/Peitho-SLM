/* Exercise each page's reply renderer without a browser.

   paint() shipped in the Khan build calling itself instead of writing the body, so
   every reply without a think marker recursed until the stack blew - and the served
   page was fine, because the bug came from hand-porting between the two. Nothing
   caught it, because nothing ran that function outside a browser. This does.

       node tests/check_pages.mjs
*/

import { readFileSync, existsSync } from "node:fs";

const PAGES = ["peitho.html", "khan/peitho_cdn.html"];
const THINK = "◇";

/* A bubble is two elements the page writes into; that is all paint() touches. */
function stubBubble() {
    const make = () => ({
        textContent: "", className: "", lastChild: null,
        append(...kids) {
            this.children.push(...kids);
            this.lastChild = kids[kids.length - 1];
        },
        before() {},
        children: []
    });
    return { body: make(), div: make() };
}

let failed = 0;

function check(name, condition, detail) {
    if (condition) {
        console.log(`ok    ${name}`);
    } else {
        console.log(`FAIL  ${name}${detail ? ": " + detail : ""}`);
        failed++;
    }
}

for (const path of PAGES) {
    if (!existsSync(path)) {
        console.log(`skip  ${path} is not present`);
        continue;
    }
    const html = readFileSync(path, "utf8");
    const script = html.slice(html.indexOf("<script>") + "<script>".length,
                              html.indexOf("</script>"));

    const noop = () => {};
    const stub = {
        classList: { add: noop, remove: noop, toggle: noop, contains: () => false },
        style: {}, textContent: "", innerHTML: "", hidden: true, disabled: false,
        append: noop, remove: noop, addEventListener: noop, querySelectorAll: () => [],
        getBoundingClientRect: () => ({ top: 0, height: 0 }), children: [],
        value: "0.7", scrollHeight: 0, clientHeight: 0, scrollTop: 0, offsetHeight: 0
    };
    globalThis.document = {
        readyState: "complete",
        getElementById: () => stub,
        createElement: () => ({ ...stub, textContent: "", className: "", children: [],
                                append(...k) { this.children.push(...k);
                                               this.lastChild = k[k.length - 1]; } }),
        querySelector: () => stub, querySelectorAll: () => [],
        createTextNode: text => ({ textContent: text }),
        addEventListener: noop, head: stub, documentElement: stub
    };
    globalThis.window = globalThis;
    globalThis.addEventListener = noop;
    globalThis.location = { protocol: "http:" };
    globalThis.fetch = undefined;
    globalThis.requestAnimationFrame = noop;

    const { paint } = new Function(script + "\n;return { paint };")();

    // A reply with no marker: the branch that recursed.
    let bubble = stubBubble();
    let threw = null;
    try {
        paint(bubble, "Hey there. Go ahead.");
    } catch (e) {
        threw = e.message;
    }
    check(`${path}: a reply with no think marker renders`, threw === null, threw);
    check(`${path}: it lands in the body`,
          bubble.body.textContent === "Hey there. Go ahead.",
          JSON.stringify(bubble.body.textContent));

    // A reply with working: split at the marker, and the marker itself never shown.
    bubble = stubBubble();
    threw = null;
    try {
        paint(bubble, `Tens: 40 + 30 = 70.${THINK}That comes to 84.`);
    } catch (e) {
        threw = e.message;
    }
    check(`${path}: a reply with working renders`, threw === null, threw);
    check(`${path}: the reply is only what follows the marker`,
          bubble.body.textContent === "That comes to 84.",
          JSON.stringify(bubble.body.textContent));
    check(`${path}: the marker is not shown`,
          !bubble.body.textContent.includes(THINK));
}

process.exit(failed ? 1 : 0);
