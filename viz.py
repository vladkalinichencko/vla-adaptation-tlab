"""Build the self-contained VLA experiment observer in runs/report.html."""

import json
import math
import pathlib
import re
from collections import defaultdict


LABELS = {
    "default": "Naive fine-tune, old proxy",
    "fixed": "Corrected fine-tune, old proxy",
    "seen_pretrain": "Seen pretrain",
    "zero_shot": "Zero-shot",
    "wrong_instruction": "Wrong instruction",
    "naive_finetune": "Naive fine-tune",
    "longer_finetune": "Fine-tune, twice as long",
    "full_finetune": "Full fine-tune",
    "mix_seen": "Target mixed with seen",
    "lora_r32": "LoRA r=32",
    "chunk_10": "Action chunk 10",
    "image_augmentations": "Image augmentations",
    "latent_transition": "Latent transition pretrain",
    "latent_seen_decoder": "Decoder trained on seen actions",
    "latent_seen_actions": "Latent transition with seen actions",
    "latent_video_only": "Latent transition without seen actions",
}


def readable(name):
    name = re.sub(r"^preliminary_", "", name)
    name = re.sub(r"_t\d+(?:_n\d+)?(?:_s\d+)?$", "", name)
    return LABELS.get(name, name.replace("_", " ").title())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def finite(value):
    return value if not isinstance(value, float) or math.isfinite(value) else None


def proxy_curves(rows):
    grouped = defaultdict(list)
    for row in rows:
        if "eval" not in row:
            grouped[(row["method"], row["n_demos"])].append(row["success"])
    curves = defaultdict(list)
    for (method, demos), values in grouped.items():
        curves[method].append({"demos": demos, "success": sum(values) / len(values)})
    return [
        {"name": readable(method), "points": sorted(points, key=lambda x: x["demos"])}
        for method, points in curves.items() if len(points) > 1
    ]


def load_data():
    runs = pathlib.Path("runs")
    rows = read_jsonl(runs / "results.jsonl")
    training = []
    for path in sorted(runs.glob("preliminary_*/metrics.jsonl")):
        values = read_jsonl(path)
        kept = [row for i, row in enumerate(values) if i == 0 or i == len(values) - 1 or (i + 1) % 10 == 0]
        training.append({"name": readable(path.parent.name), "rows": [
            {key: finite(value) for key, value in row.items()} for row in kept
        ]})

    diagnostics = runs / "diagnostics"
    actions = [{
        "name": readable(path.stem.removesuffix("_actions")),
        "samples": json.loads(path.read_text()),
    } for path in sorted(diagnostics.glob("preliminary_*_actions.json"))]
    transitions = []
    for path in sorted(diagnostics.glob("preliminary_*_transitions.json")):
        row = json.loads(path.read_text())
        row["name"] = readable(path.stem.removesuffix("_transitions"))
        transitions.append(row)
    cells = [{**row, "phase": "Preliminary" if "eval" in row else "Old proxy",
              "label": readable(row["method"])} for row in rows]
    return {"curves": proxy_curves(rows), "training": training, "actions": actions,
            "transitions": transitions, "cells": cells}


TEMPLATE = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VLA experiment observer</title>
<style>
:root{--ink:#18202b;--muted:#667085;--line:#dfe4eb;--blue:#2563eb;--orange:#d97706;--teal:#0f766e;--violet:#7c3aed}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1120px;margin:auto;padding:34px 24px 64px}h1{font-size:25px;letter-spacing:-.025em;margin:0 0 3px}h2{font-size:17px;letter-spacing:-.01em;margin:0 0 4px}p{margin:0}.intro,.note,.meta{color:var(--muted)}.intro{font-size:15px;margin-bottom:28px}
.grid{display:grid;grid-template-columns:1fr 1fr;column-gap:42px}.panel{padding:28px 0;margin:0;border-top:1px solid var(--line)}.wide{grid-column:1/-1}
.controls{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0 10px}select,input{border:0;border-bottom:1px solid #aeb7c3;border-radius:0;background:#fff;color:var(--ink);padding:6px 2px;font:inherit}select:focus,input:focus{outline:0;border-color:var(--ink)}input{min-width:240px}
.chart{width:100%;height:auto;display:block}.axis{stroke:#cfd6df}.gridline{stroke:#e9edf2}.tick{fill:var(--muted);font-size:11px}.label{font-size:12px;font-weight:650}.hit{fill:transparent;cursor:crosshair}.swatch{display:inline-block;width:24px;border-top:3px solid;margin:0 6px 3px 14px}.swatch:first-child{margin-left:0}.dash{border-top-style:dashed}
.tooltip{position:fixed;display:none;pointer-events:none;background:#111827;color:#fff;border-radius:4px;padding:7px 9px;font-size:12px;line-height:1.4;z-index:9;max-width:300px}
.heat{display:grid;gap:2px;margin-top:12px;overflow:auto}.heat-row{display:grid;grid-template-columns:72px repeat(50,minmax(9px,1fr));gap:2px;min-width:760px;align-items:center}.heat-cell{height:23px;border-radius:2px}.heat-name{color:var(--muted);font-size:11px}
details summary{cursor:pointer;font-weight:650}table{border-collapse:collapse;width:100%;font-size:12px}.table-wrap{max-height:420px;overflow:auto;margin-top:10px}th,td{padding:7px 9px;border-bottom:1px solid #edf0f4;text-align:left;white-space:nowrap}th{position:sticky;top:0;background:#f8fafc;z-index:1}td.num{text-align:right;font-variant-numeric:tabular-nums}
@media(max-width:760px){.grid{grid-template-columns:1fr}.panel{padding:24px 0}main{padding:24px 16px}.wide{grid-column:auto}}
</style><main>
<h1>VLA experiment observer</h1><p class="intro">Training behavior, predicted action chunks, and measured adaptation results.</p>
<section class="panel"><h2>Cost curve</h2><p class="note">Only old proxy methods measured at multiple demonstration budgets are connected. Preliminary one-budget runs stay out of this comparison.</p><div id="cost"></div></section>
<div class="grid"><section class="panel"><h2>Training dynamics</h2><p class="note">Logged every ten optimizer steps; hover for the exact value.</p><div class="controls"><select id="train-run"></select><select id="train-metric"></select></div><div id="training"></div></section>
<section class="panel"><h2>Action chunks</h2><p class="note">Target and policy prediction for one of the seven action coordinates across the full chunk.</p><div class="controls"><select id="action-run"></select><select id="action-sample"></select><select id="action-dim"></select></div><p class="meta" id="action-meta"></p><div id="actions"></div></section>
<section class="panel wide"><h2>Latent transition</h2><p class="note">Cosine similarity between predicted and observed visual-latent change, by chunk step and camera.</p><div id="latent"></div></section></div>
<section class="panel"><details><summary>All measured cells (<span id="cell-count"></span>)</summary><div class="controls"><select id="phase"><option>Preliminary</option><option>Old proxy</option><option>All</option></select><input id="search" placeholder="Filter method or task"></div><div class="table-wrap"><table><thead><tr><th>Phase</th><th>Method</th><th>Task</th><th>Demos</th><th>Episodes</th><th>Success</th></tr></thead><tbody id="cells"></tbody></table></div></details></section>
</main><div class="tooltip" id="tip"></div><script>
const DATA=__DATA__, NS="http://www.w3.org/2000/svg", tip=document.querySelector("#tip");
const $=s=>document.querySelector(s), E=(tag,a={})=>{const n=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(a))n.setAttribute(k,v);return n};
const fmt=v=>v==null?"—":Number(v).toLocaleString(undefined,{maximumFractionDigits:4});
const METRICS={loss:"Total loss",grad_norm:"Gradient norm",lr:"Learning rate",next_lr:"Next learning rate",seconds:"Seconds per step",samples_per_second:"Samples per second",losses_after_forward:"Action loss",losses_after_in_ep_bound:"Loss after episode mask",losses_after_rm_padding:"Loss after padding mask"};
function hover(node,html){node.onmousemove=e=>{tip.innerHTML=html();tip.style.display="block";tip.style.left=Math.min(e.clientX+12,innerWidth-310)+"px";tip.style.top=(e.clientY+12)+"px"};node.onmouseleave=()=>tip.style.display="none"}
function lineChart(target,series,{yMin=null,yMax=null,xName="Step"}={}){const W=850,H=310,L=55,R=116,T=18,B=38,all=series.flatMap(s=>s.points),xs=all.map(p=>p.x),ys=all.map(p=>p.y).filter(Number.isFinite);target.replaceChildren();if(!all.length||!ys.length){target.textContent="No data";return}const x0=Math.min(...xs),x1=Math.max(...xs),lo=yMin??Math.min(...ys),hi=yMax??Math.max(...ys),span=Math.max(hi-lo,1e-9),X=x=>L+(x-x0)/Math.max(x1-x0,1)*(W-L-R),Y=y=>H-B-(y-lo)/span*(H-T-B),svg=E("svg",{viewBox:`0 0 ${W} ${H}`,class:"chart"});
for(let i=0;i<=4;i++){const y=T+i*(H-T-B)/4,v=hi-i*span/4;svg.append(E("line",{x1:L,y1:y,x2:W-R,y2:y,class:"gridline"}));const t=E("text",{x:4,y:y+4,class:"tick"});t.textContent=fmt(v);svg.append(t)}
svg.append(E("line",{x1:L,y1:H-B,x2:W-R,y2:H-B,class:"axis"}));const ticks=[...new Set(xs)].sort((a,b)=>a-b),stride=Math.max(1,Math.ceil(ticks.length/8));ticks.filter((_,i)=>i===0||i===ticks.length-1||i%stride===0).forEach(x=>{const t=E("text",{x:X(x),y:H-12,class:"tick","text-anchor":"middle"});t.textContent=x;svg.append(t)});
series.forEach((s,si)=>{const path=E("path",{d:s.points.map((p,i)=>`${i?"L":"M"}${X(p.x)} ${Y(p.y)}`).join(" "),fill:"none",stroke:s.color,"stroke-width":2.5,"stroke-dasharray":s.dash?"8 5":""});svg.append(path);s.points.forEach(p=>{const c=E(si&&s.square?"rect":"circle",si&&s.square?{x:X(p.x)-4,y:Y(p.y)-4,width:8,height:8,fill:s.color}:{cx:X(p.x),cy:Y(p.y),r:4,fill:s.color});hover(c,()=>`<b>${s.name}</b><br>${xName}: ${p.x}<br>Value: ${fmt(p.y)}${p.detail||""}`);svg.append(c)});if(s.direct){const t=E("text",{x:W-8,y:Y(s.points.at(-1).y)+(si?16:-8),class:"label",fill:s.color,"text-anchor":"end"});t.textContent=s.name;svg.append(t)}});target.append(svg)}
lineChart($("#cost"),DATA.curves.map((s,i)=>({...s,color:i?"#d97706":"#2563eb",dash:i>0,square:i>0,direct:true,points:s.points.map(p=>({x:p.demos,y:p.success}))})),{yMin:0,yMax:1,xName:"Demos"});
const trainRun=$("#train-run"),trainMetric=$("#train-metric");DATA.training.forEach((r,i)=>trainRun.add(new Option(r.name,i)));
function training(){const run=DATA.training[+trainRun.value],keys=Object.keys(run?.rows[0]||{}).filter(k=>k!=="step"&&run.rows.some(r=>Number.isFinite(r[k]))),old=trainMetric.value;trainMetric.replaceChildren(...keys.map(k=>new Option(METRICS[k]||k,k)));if(keys.includes(old))trainMetric.value=old;const key=trainMetric.value;lineChart($("#training"),[{name:METRICS[key]||key,color:"#7c3aed",points:run.rows.filter(r=>Number.isFinite(r[key])).map(r=>({x:r.step,y:r[key]}))}])}trainRun.onchange=training;trainMetric.onchange=training;training();
const actionRun=$("#action-run"),actionSample=$("#action-sample"),actionDim=$("#action-dim");DATA.actions.forEach((r,i)=>actionRun.add(new Option(r.name,i)));for(let i=0;i<7;i++)actionDim.add(new Option(`Action ${i}`,i));
function actionSamples(){const samples=DATA.actions[+actionRun.value]?.samples||[];actionSample.replaceChildren(...samples.map((s,i)=>new Option(`Episode ${s.episode}, frame ${s.frame}`,i)));actions()}
function actions(){const s=DATA.actions[+actionRun.value]?.samples[+actionSample.value];if(!s)return;const d=+actionDim.value,target=s.target.map((a,i)=>({x:i,y:a[d]})),pred=s.predicted.map((a,i)=>({x:i,y:a[d]})),mae=target.reduce((v,p,i)=>v+Math.abs(p.y-pred[i].y),0)/target.length;target.forEach((p,i)=>p.detail=`<br>Prediction: ${fmt(pred[i].y)}<br>Absolute error: ${fmt(Math.abs(p.y-pred[i].y))}`);pred.forEach((p,i)=>p.detail=`<br>Target: ${fmt(target[i].y)}<br>Absolute error: ${fmt(Math.abs(p.y-target[i].y))}`);$("#action-meta").textContent=`${s.instruction} · mean absolute error ${fmt(mae)}`;lineChart($("#actions"),[{name:"Target",color:"#2563eb",points:target},{name:"Prediction",color:"#d97706",dash:true,square:true,points:pred}])}actionRun.onchange=actionSamples;actionSample.onchange=actions;actionDim.onchange=actions;actionSamples();
function latent(){const root=$("#latent"),d=DATA.transitions[0];if(!d){root.textContent="No latent-transition diagnostics";return}const flat=d.cosine_by_step_and_view.flat(),mean=flat.reduce((a,b)=>a+b,0)/flat.length;root.innerHTML=`<p class="meta">${d.name} · episode ${d.episode}, frame ${d.frame} · mean cosine ${fmt(mean)}</p>`;const heat=document.createElement("div");heat.className="heat";for(let v=0;v<d.cosine_by_step_and_view[0].length;v++){const row=document.createElement("div");row.className="heat-row";const name=document.createElement("span");name.className="heat-name";name.textContent=`Camera ${v+1}`;row.append(name);d.cosine_by_step_and_view.forEach((step,i)=>{const value=step[v],cell=document.createElement("span");cell.className="heat-cell";cell.style.background=`hsl(${12+Math.max(0,Math.min(1,(value+1)/2))*198} 68% 48%)`;hover(cell,()=>`<b>Camera ${v+1}, step ${i}</b><br>Cosine: ${fmt(value)}<br>Target norm: ${fmt(d.target_norm_by_step_and_view[i][v])}<br>Prediction norm: ${fmt(d.predicted_norm_by_step_and_view[i][v])}`);row.append(cell)});heat.append(row)}root.append(heat)}latent();
function cells(){const phase=$("#phase").value,q=$("#search").value.toLowerCase(),rows=DATA.cells.filter(r=>(phase==="All"||r.phase===phase)&&`${r.label} ${r.task}`.toLowerCase().includes(q));$("#cell-count").textContent=rows.length;$("#cells").replaceChildren(...rows.map(r=>{const tr=document.createElement("tr");[r.phase,r.label,r.task,r.n_demos,r.n_episodes,fmt(r.success)].forEach((v,i)=>{const td=document.createElement("td");td.textContent=v;if(i>2)td.className="num";tr.append(td)});return tr}))}$("#phase").onchange=cells;$("#search").oninput=cells;cells();
</script></html>'''


def main():
    data = load_data()
    output = pathlib.Path("runs/report.html")
    output.write_text(TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False, allow_nan=False)))
    print(f"{len(data['curves'])} curves, {len(data['training'])} training runs, "
          f"{len(data['actions'])} action diagnostics, {len(data['cells'])} cells -> {output}")


if __name__ == "__main__":
    main()
