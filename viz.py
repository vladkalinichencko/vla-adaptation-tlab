"""runs/results.jsonl -> one self-contained interactive page.

The deliverable is the cost curve, and a curve of five points is easy to over-read.
So the page keeps both readings on it: the aggregated curve the task asks for, and
every individual evaluated cell behind it, with the number of episodes each rests on.
At 20 episodes one success is 0.05, and that has to stay visible.

    python viz.py                      # -> runs/report.html
"""

import argparse
import json
import pathlib

import cost_curve
import rollouts

TEMPLATE = """<title>VLA — кривая цены адаптации</title>
<style>
:root { --bg:#fff; --fg:#111; --mut:#666; --line:#ddd; }
@media (prefers-color-scheme: dark) { :root:not([data-theme=light]) {
  --bg:#14161a; --fg:#e8e8e8; --mut:#9aa0a6; --line:#2c3038; } }
:root[data-theme=dark] { --bg:#14161a; --fg:#e8e8e8; --mut:#9aa0a6; --line:#2c3038; }
body { background:var(--bg); color:var(--fg); font:14px/1.5 -apple-system,system-ui,sans-serif;
       margin:0 auto; max-width:1000px; padding:24px; }
h1 { font-size:20px; margin:0 0 4px; } h2 { font-size:15px; margin:26px 0 8px; font-weight:600; }
p.note { color:var(--mut); margin:2px 0 12px; }
table { border-collapse:collapse; font-size:13px; } td,th { padding:3px 12px 3px 0; text-align:right; }
th:first-child,td:first-child { text-align:left; }
th { border-bottom:1px solid var(--line); font-weight:600; }
.card { border:1px solid var(--line); border-radius:6px; padding:8px 10px; display:inline-block; }
.card b { font-size:12px; } svg { display:block; margin-top:4px; }
.legend { display:flex; gap:14px; flex-wrap:wrap; font-size:12px; color:var(--mut); margin:6px 0; }
.legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; }
.ax { stroke:var(--line); } .tick { fill:var(--mut); font-size:10px; }
</style>
<h1>VLA: сколько демо стоит новая задача</h1>
<p class="note">Success на целевых задачах libero_goal в зависимости от числа демо.
Метод хорош ровно настолько, насколько он двигает эту кривую влево.</p>
<div id="app"></div>
<script>
const DATA = __DATA__;
const PAL = ["#2b6cb0","#c05621","#2f855a","#805ad5","#b83280","#4a5568","#b7791f","#2c7a7b"];
const SVG = new Set(["svg","g","path","line","text","rect","circle"]);
function el(tag, attrs, kids) {
  const n = document.createElementNS(SVG.has(tag) ? "http://www.w3.org/2000/svg"
    : "http://www.w3.org/1999/xhtml", tag);
  for (const k in (attrs||{})) n.setAttribute(k, attrs[k]);
  for (const c of (kids||[])) n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  return n;
}
const fmt = v => v == null ? "—" : (+v.toFixed(3)).toString();
const app = document.getElementById("app");

// --- кривая
{
  const methods = Object.keys(DATA.curve);
  const w = 620, h = 340, pad = 46;
  const budgets = [...new Set(methods.flatMap(m => Object.keys(DATA.curve[m]).map(Number)))].sort((a,b)=>a-b);
  const X = v => pad + (budgets.indexOf(v)) / Math.max(budgets.length-1,1) * (w-pad-14);
  const Y = v => h-26 - v * (h-40);
  const g = el("svg",{width:w,height:h});
  g.appendChild(el("line",{x1:pad,y1:h-26,x2:w-14,y2:h-26,class:"ax"}));
  g.appendChild(el("line",{x1:pad,y1:12,x2:pad,y2:h-26,class:"ax"}));
  for (const v of [0,0.25,0.5,0.75,1]) {
    g.appendChild(el("line",{x1:pad,y1:Y(v),x2:w-14,y2:Y(v),class:"ax"}));
    g.appendChild(el("text",{x:6,y:Y(v)+4,class:"tick"},[fmt(v)]));
  }
  for (const b of budgets)
    g.appendChild(el("text",{x:X(b),y:h-8,class:"tick","text-anchor":"middle"},[String(b)+" демо"]));
  methods.forEach((m,i) => {
    const ns = Object.keys(DATA.curve[m]).map(Number).sort((a,b)=>a-b);
    g.appendChild(el("path",{fill:"none",stroke:PAL[i%PAL.length],"stroke-width":2,
      d: ns.map((n,j)=>(j?"L":"M")+X(n)+" "+Y(DATA.curve[m][n][0])).join(" ")}));
    for (const n of ns)
      g.appendChild(el("circle",{cx:X(n),cy:Y(DATA.curve[m][n][0]),r:4,fill:PAL[i%PAL.length]}));
  });
  app.appendChild(el("h2",{},["Кривая цены"]));
  app.appendChild(el("div",{class:"legend"}, methods.map((m,i)=>
    el("span",{},[el("i",{style:"background:"+PAL[i%PAL.length]}), m]))));
  app.appendChild(el("div",{class:"card"},[g]));
}

// --- кадры провалов
if (DATA.failures.length) {
  app.appendChild(el("h2",{},["Где ломается траектория"]));
  app.appendChild(el("p",{class:"note"},["Кадры равномерно по эпизоду, слева направо. "
    + "Строка — один эпизод; подпись — прогон и исход."]));
  const t = el("table",{});
  for (const f of DATA.failures) {
    const tr = el("tr",{});
    tr.appendChild(el("td",{},[f.run + " · эп." + f.episode + " · " + (f.ok ? "успех" : "ПРОВАЛ")]));
    for (const src of f.frames)
      tr.appendChild(el("td",{},[el("img",{src: src, style:"width:150px;border-radius:4px"})]));
    t.appendChild(tr);
  }
  app.appendChild(t);
}

// --- все ячейки
{
  app.appendChild(el("h2",{},["Каждая измеренная ячейка"]));
  app.appendChild(el("p",{class:"note"},["При 20 эпизодах один успех — это 0.05. "
    + "Разница меньше 0.1 между двумя ячейками ничего не значит."]));
  const t = el("table",{},[el("tr",{},["метод","задача","демо","эпизодов","success","политика"]
    .map(x=>el("th",{},[x])))]);
  for (const r of DATA.rows)
    t.appendChild(el("tr",{},[r.method, r.task, String(r.n_demos), String(r.n_episodes),
      fmt(r.success), (r.instruction ? "инструкция: " + r.instruction : r.policy)]
      .map(v=>el("td",{},[v]))));
  app.appendChild(t);
}
</script>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="?", default="runs/results.jsonl")
    p.add_argument("--out", default="runs/report.html")
    p.add_argument("--only", nargs="+", default=None, help="какие методы показывать на кривой")
    p.add_argument("--fail-runs", nargs="+", default=["fixed_n25"],
                   help="из каких оценок брать кадры провалов")
    p.add_argument("--n-fails", type=int, default=3)
    args = p.parse_args()

    rows = [json.loads(x) for x in pathlib.Path(args.results).read_text().splitlines() if x.strip()]
    curve = cost_curve.load(args.results)
    if args.only:
        curve = {m: v for m, v in curve.items() if m in args.only}
    picked = [e for e in rollouts.episodes() if e[0] in args.fail_runs]
    failures = []
    for run, _, episode, ok, path in ([e for e in picked if not e[3]][: args.n_fails]
                                      + [e for e in picked if e[3]][:1]):
        images, _ = rollouts.frames(path, 4)
        failures.append({"run": run, "episode": episode, "ok": ok,
                         "frames": [rollouts.as_data_uri(f) for f in images]})

    data = {"curve": {m: {str(n): v for n, v in d.items()} for m, d in curve.items()},
            "rows": rows, "failures": failures}
    pathlib.Path(args.out).write_text(TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False)))
    print(f"{len(curve)} методов, {len(rows)} ячеек, {len(failures)} роллаутов -> {args.out}")


if __name__ == "__main__":
    main()
