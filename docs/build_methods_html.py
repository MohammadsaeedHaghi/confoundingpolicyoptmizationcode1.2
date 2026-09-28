#!/usr/bin/env python3
"""Build docs/methods.html from docs/METHODS.md (the .md is the single source; never edit the .html).

Each "## [Group] Label" section becomes one tab. Supported Markdown subset: ###/#### headings,
paragraphs, "- " and "1. " lists (single-line items), pipe tables (a table whose header is
"Fact | Value" renders as a facts grid), "> **Label.** ..." callouts (Finding / Caveat / Note / Check),
$$ display math blocks, $inline math$, `code`, **bold**, *italic*. Math is typeset in the browser by
MathJax 3 (SVG output: no font files needed).
Usage: python3 build_methods_html.py   (writes methods.html next to this file)
"""
import html
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC, OUT = HERE / "METHODS.md", HERE / "methods.html"


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def inline(t):
    maths, codes = [], []
    t = t.replace("\\*", "\x03")
    t = re.sub(r"`([^`]+)`", lambda m: (codes.append(m.group(1)), "\x01%d\x01" % (len(codes) - 1))[1], t)
    t = re.sub(r"\$([^$]+?)\$", lambda m: (maths.append(m.group(1)), "\x00%d\x00" % (len(maths) - 1))[1], t)
    t = html.escape(t, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    t = re.sub(r"\x01(\d+)\x01", lambda m: "<code>%s</code>" % html.escape(codes[int(m.group(1))], quote=False), t)
    t = re.sub(r"\x00(\d+)\x00", lambda m: "\\(%s\\)" % html.escape(maths[int(m.group(1))], quote=False), t)
    return t.replace("\x03", "*")


def cells(row):
    return [c.strip() for c in row.strip().strip("|").split("|")]


CALLOUT = {"finding": "Finding", "caveat": "Caveat", "note": "Note", "check": "Check"}


def render(lines, pfx):
    out, para, i = [], [], 0

    def flush():
        if para:
            out.append("<p>%s</p>" % inline(" ".join(para)))
            para.clear()

    while i < len(lines):
        ln = lines[i]; s = ln.strip()
        if not s:
            flush(); i += 1; continue
        if s == "$$":
            flush(); j = i + 1; body = []
            while lines[j].strip() != "$$":
                body.append(lines[j]); j += 1
            out.append('<div class="math-block">\\[%s\\]</div>' % html.escape("\n".join(body), quote=False))
            i = j + 1; continue
        if s.startswith("```"):
            flush(); j = i + 1; body = []
            while not lines[j].strip().startswith("```"):
                body.append(lines[j]); j += 1
            out.append("<pre><code>%s</code></pre>" % html.escape("\n".join(body)))
            i = j + 1; continue
        m = re.match(r"^(#{3,4}) (.*)$", s)
        if m:
            flush(); tag = "h3" if len(m.group(1)) == 3 else "h4"
            out.append('<%s id="%s-%s">%s</%s>' % (tag, pfx, slug(m.group(2)), inline(m.group(2)), tag))
            i += 1; continue
        if s.startswith("|"):
            flush(); rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i]); i += 1
            head, body = cells(rows[0]), [cells(r) for r in rows[2:]]
            if head == ["Fact", "Value"]:
                out.append('<dl class="facts">%s</dl>' % "".join(
                    "<dt>%s</dt><dd>%s</dd>" % (inline(k), inline(v)) for k, v in body))
            else:
                out.append('<div class="table-wrap"><table><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>' % (
                    "".join("<th>%s</th>" % inline(c) for c in head),
                    "".join("<tr>%s</tr>" % "".join("<td>%s</td>" % inline(c) for c in r) for r in body)))
            continue
        if s.startswith("> "):
            flush(); body = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                body.append(lines[i].strip()[1:].strip()); i += 1
            text = " ".join(body)
            m = re.match(r"^\*\*(\w+)\.\*\*\s*(.*)$", text)
            kind = m.group(1).lower() if m and m.group(1).lower() in CALLOUT else "note"
            text = m.group(2) if m else text
            out.append('<aside class="callout %s"><span class="callout-label">%s</span><p>%s</p></aside>'
                       % (kind, CALLOUT[kind], inline(text)))
            continue
        if re.match(r"^(- |\d+\. )", s):
            flush(); ordered = bool(re.match(r"^\d+\. ", s)); items = []
            while i < len(lines) and re.match(r"^(- |\d+\. )", lines[i].strip()):
                items.append(re.sub(r"^(- |\d+\. )", "", lines[i].strip())); i += 1
                while i < len(lines) and lines[i].startswith("   ") and lines[i].strip() and not re.match(r"^(- |\d+\. )", lines[i].strip()):
                    items[-1] += " " + lines[i].strip(); i += 1
            tag = "ol" if ordered else "ul"
            out.append("<%s>%s</%s>" % (tag, "".join("<li>%s</li>" % inline(t) for t in items), tag))
            continue
        para.append(s); i += 1
    flush()
    return "\n".join(out)


def main():
    lines = SRC.read_text().split("\n")
    title = lines[0].lstrip("# ").strip()
    secs, cur = [], None
    intro = []
    for ln in lines[1:]:
        m = re.match(r"^## \[([^\]]+)\] (.*)$", ln)
        if m:
            cur = {"group": m.group(1), "label": m.group(2).strip(), "lines": []}; secs.append(cur); continue
        (cur["lines"] if cur else intro).append(ln)

    groups = []
    for s in secs:
        if not groups or groups[-1][0] != s["group"]:
            groups.append((s["group"], []))
        groups[-1][1].append(s)
    nav = []
    for g, ss in groups:
        btns = "".join('<button type="button" role="tab" id="tab-%s" aria-controls="panel-%s" aria-selected="false" tabindex="-1">%s</button>'
                       % (slug(s["label"]), slug(s["label"]), html.escape(s["label"])) for s in ss)
        nav.append('<div class="tab-group"><span class="tab-group-label">%s</span><div class="tab-row">%s</div></div>' % (html.escape(g), btns))
    panels = "".join('<section class="panel" role="tabpanel" id="panel-%s" aria-labelledby="tab-%s" hidden><h2>%s</h2>%s</section>'
                     % (slug(s["label"]), slug(s["label"]), inline(s["label"]), render(s["lines"], slug(s["label"]))) for s in secs)
    page = TEMPLATE.replace("{{TITLE}}", html.escape(title)).replace("{{INTRO}}", render(intro, "intro")) \
                   .replace("{{NAV}}", "".join(nav)).replace("{{PANELS}}", panels)
    OUT.write_text(page)
    print("wrote %s (%d tabs, %d bytes)" % (OUT, len(secs), len(page)))


TEMPLATE = r"""<title>Code 1.2 Method Programs</title>
<meta name="description" content="The optimisation problem each code 1.2 method solves, derived from the code.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<style>
:root{
  --ground:#F3F4F0; --panel:#FFFFFF; --ink:#1D2320; --muted:#5E6862; --rule:#D8DCD4;
  --accent:#245C86; --accent-soft:#E4ECF3; --code:#7A4E12;
  --finding:#A8412A; --finding-bg:#F8EAE5; --caveat:#7A5A0C; --caveat-bg:#F6EFDA;
  --note:#4A5560; --note-bg:#ECEFF1; --check:#2E6E4A; --check-bg:#E5F1EA;
  --sans:"IBM Plex Sans Condensed","Arial Narrow",system-ui,sans-serif;
  --serif:"Source Serif 4",Georgia,"Times New Roman",serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --ground:#121614; --panel:#191E1B; --ink:#E4E8E3; --muted:#9AA59E; --rule:#2B322E;
  --accent:#86B6DB; --accent-soft:#1E2A34; --code:#D8A55C;
  --finding:#EB8F76; --finding-bg:#33211C; --caveat:#DCC07A; --caveat-bg:#2E2918;
  --note:#B3BCC4; --note-bg:#1F2428; --check:#88CBA3; --check-bg:#1B2A21;
}}
:root[data-theme="dark"]{
  --ground:#121614; --panel:#191E1B; --ink:#E4E8E3; --muted:#9AA59E; --rule:#2B322E;
  --accent:#86B6DB; --accent-soft:#1E2A34; --code:#D8A55C;
  --finding:#EB8F76; --finding-bg:#33211C; --caveat:#DCC07A; --caveat-bg:#2E2918;
  --note:#B3BCC4; --note-bg:#1F2428; --check:#88CBA3; --check-bg:#1B2A21;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{background:var(--ground);color:var(--ink);font-family:var(--serif);font-size:17px;line-height:1.6;margin:0;padding-inline:clamp(16px,4vw,40px)}
.wrap{max-width:980px;margin:0 auto}
header.top{padding-block:40px 20px}
.kicker{font-family:var(--mono);font-size:12.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0 0 10px}
h1{font-family:var(--sans);font-weight:600;font-size:clamp(28px,4.4vw,40px);line-height:1.12;margin:0 0 18px;text-wrap:balance;letter-spacing:-.005em}
header.top p{max-width:68ch;margin:0 0 12px}
nav.tabs{position:sticky;top:0;z-index:5;background:var(--ground);border-bottom:1px solid var(--rule);padding-block:10px;display:flex;flex-wrap:wrap;gap:6px 22px}
.tab-group{display:flex;flex-direction:column;gap:4px;min-width:0}
.tab-group-label{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.tab-row{display:flex;flex-wrap:wrap;gap:4px}
nav.tabs button{font-family:var(--sans);font-weight:500;font-size:15px;color:var(--ink);background:transparent;border:1px solid var(--rule);border-radius:3px;padding:4px 10px;cursor:pointer;line-height:1.3}
nav.tabs button:hover{border-color:var(--accent);color:var(--accent)}
nav.tabs button[aria-selected="true"]{background:var(--accent);border-color:var(--accent);color:var(--panel)}
nav.tabs button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
main{padding-block:28px 64px}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:4px;padding:clamp(18px,3.5vw,40px)}
.panel>h2{font-family:var(--sans);font-weight:600;font-size:30px;line-height:1.15;margin:0 0 18px;letter-spacing:-.005em}
.panel h3{font-family:var(--sans);font-weight:600;font-size:21px;margin:34px 0 10px;text-wrap:balance}
.panel h4{font-family:var(--sans);font-weight:600;font-size:17px;margin:24px 0 8px}
.panel p,.panel li{max-width:70ch}
.panel p{margin:0 0 14px}
.panel ul,.panel ol{margin:0 0 16px;padding-left:1.3em}
.panel li{margin:0 0 6px}
code{font-family:var(--mono);font-size:.84em;color:var(--code);background:transparent;overflow-wrap:anywhere}
pre{overflow-x:auto;background:var(--note-bg);padding:12px;border-radius:3px}
a{color:var(--accent)}
.math-block{overflow-x:auto;overflow-y:hidden;margin:6px 0 18px;padding:10px 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}
.math-block mjx-container{margin:0!important}
mjx-container{color:var(--ink)}
.table-wrap{overflow-x:auto;margin:4px 0 20px}
table{border-collapse:collapse;font-size:15px;line-height:1.45;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;vertical-align:top;padding:7px 10px;border-bottom:1px solid var(--rule)}
th{font-family:var(--sans);font-weight:600;font-size:14px;color:var(--muted);border-bottom:1.5px solid var(--ink);white-space:nowrap}
td:first-child{font-family:var(--sans);font-weight:500;white-space:nowrap}
dl.facts{display:grid;grid-template-columns:max-content 1fr;gap:6px 18px;margin:0 0 22px;padding:14px 16px;background:var(--accent-soft);border-radius:3px;font-size:15px}
dl.facts dt{font-family:var(--mono);font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);padding-top:3px}
dl.facts dd{margin:0;min-width:0}
.callout{margin:6px 0 20px;padding:12px 16px;border-radius:3px;border:1px solid transparent}
.callout p{margin:0}
.callout-label{display:inline-block;font-family:var(--mono);font-size:11px;letter-spacing:.09em;text-transform:uppercase;font-weight:500;margin-bottom:4px}
.callout.finding{background:var(--finding-bg);border-color:color-mix(in srgb,var(--finding) 35%,transparent)}
.callout.finding .callout-label{color:var(--finding)}
.callout.caveat{background:var(--caveat-bg);border-color:color-mix(in srgb,var(--caveat) 35%,transparent)}
.callout.caveat .callout-label{color:var(--caveat)}
.callout.note{background:var(--note-bg)}
.callout.note .callout-label{color:var(--note)}
.callout.check{background:var(--check-bg)}
.callout.check .callout-label{color:var(--check)}
footer{font-family:var(--mono);font-size:12px;color:var(--muted);padding-block:0 40px}
@media (max-width:560px){
  body{font-size:16px}
  nav.tabs{flex-wrap:nowrap;overflow-x:auto;gap:18px}
  .tab-row{flex-wrap:nowrap}
  nav.tabs button{white-space:nowrap}
  dl.facts{grid-template-columns:1fr}
  dl.facts dd{margin-bottom:6px}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important;scroll-behavior:auto!important}}
</style>
<div class="wrap">
<header class="top">
<p class="kicker">code 1.2 · methods/ · read against the source</p>
<h1>{{TITLE}}</h1>
{{INTRO}}
</header>
<nav class="tabs" role="tablist" aria-label="Methods">{{NAV}}</nav>
<main>{{PANELS}}</main>
<footer>Source of truth: code 1.2/docs/METHODS.md · this page is generated by docs/build_methods_html.py</footer>
</div>
<script>
window.MathJax = {
  tex: { inlineMath: [["\\(", "\\)"]], displayMath: [["\\[", "\\]"]] },
  svg: { fontCache: "global" },
  startup: { typeset: false, ready() { MathJax.startup.defaultReady(); MathJax.startup.promise.then(() => typeset(current())); } }
};
const done = new WeakSet();
function current() { return document.querySelector(".panel:not([hidden])"); }
function typeset(panel) {
  if (!panel || done.has(panel) || !window.MathJax || !MathJax.typesetPromise) return;
  done.add(panel);
  MathJax.typesetPromise([panel]).catch(() => done.delete(panel));
}
const tabs = [...document.querySelectorAll('nav.tabs [role="tab"]')];
function show(id, focus) {
  const btn = document.getElementById("tab-" + id) || tabs[0];
  tabs.forEach(t => {
    const on = t === btn;
    t.setAttribute("aria-selected", on); t.tabIndex = on ? 0 : -1;
    document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
  });
  if (focus) btn.focus();
  typeset(current());
  const want = "#" + btn.id.slice(4);
  if (location.hash !== want) history.replaceState(null, "", want);
}
tabs.forEach((t, i) => {
  t.addEventListener("click", () => { show(t.id.slice(4)); window.scrollTo({ top: document.querySelector("nav.tabs").offsetTop, behavior: "auto" }); });
  t.addEventListener("keydown", e => {
    const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (d) { e.preventDefault(); show(tabs[(i + d + tabs.length) % tabs.length].id.slice(4), true); }
  });
});
show((location.hash || "").slice(1));
window.addEventListener("hashchange", () => show(location.hash.slice(1)));
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg-full.min.js"></script>
"""

if __name__ == "__main__":
    main()
