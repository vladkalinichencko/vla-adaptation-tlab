"""Build the self-contained VLA Mac screening observer."""

import base64
import json
import math
import pathlib
import re
from collections import defaultdict


LABELS = {
    "seen_pretrain": "Seen pretrain",
    "zero_shot": "Zero-shot",
    "wrong_instruction": "Wrong instruction",
    "naive_finetune": "Naive fine-tune",
    "longer_finetune": "Longer fine-tune",
    "full_finetune": "Full fine-tune",
    "mix_seen": "Mix seen",
    "lora_r32": "LoRA r=32",
    "chunk_10": "Chunk 10",
    "image_augmentations": "Image augmentations",
    "latent_transition": "Latent transition pretrain",
    "latent_seen_decoder": "Latent decoder on seen actions",
    "latent_seen_actions": "Latent with seen actions",
    "latent_video_only": "Latent without seen actions",
    "lapo_representation": "LAPO representation",
    "lapo_policy": "LAPO latent policy",
    "lapo_seen_decoder": "LAPO decoder on seen actions",
    "lapo_seen_actions": "LAPO with seen actions",
    "lapo_video_only": "LAPO without seen actions",
    "default": "Naive fine-tune",
    "fixed": "Corrected fine-tune",
}


def run_id(name):
    name = re.sub(r"^preliminary_", "", name)
    return re.sub(r"_t\d+(?:_n\d+)?(?:_s\d+)?$", "", name)


def label(name):
    return LABELS.get(run_id(name), run_id(name).replace("_", " ").title())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def finite(value):
    return value if not isinstance(value, float) or math.isfinite(value) else None


def mean_difference(samples, left, right):
    values = [
        abs(a - b)
        for sample in samples
        for left_step, right_step in zip(sample[left], sample[right])
        for a, b in zip(left_step, right_step)
    ]
    return sum(values) / len(values)


def load_data():
    runs = pathlib.Path("runs")
    rows = read_jsonl(runs / "results.jsonl")
    training = []
    for path in sorted(runs.glob("preliminary_*/metrics.jsonl")):
        raw = read_jsonl(path)
        run = json.loads((path.parent / "run.json").read_text())
        kept = [row for index, row in enumerate(raw)
                if index == 0 or index == len(raw) - 1 or (index + 1) % 10 == 0]
        start = raw[0]["loss"]
        training.append({
            "id": run_id(path.parent.name),
            "name": label(path.parent.name),
            "steps": run["training"]["steps"],
            "trainable_parameters": run["training"]["trainable_parameters"],
            "rows": [{"step": row["step"], "loss": finite(row["loss"]),
                      "relative_loss": finite(row["loss"] / start)} for row in kept],
        })

    actions = []
    for path in sorted((runs / "diagnostics").glob("preliminary_*_actions.json")):
        samples = json.loads(path.read_text())
        identifier = run_id(path.stem.removesuffix("_actions"))
        actions.append({
            "id": identifier,
            "name": label(identifier),
            "samples": samples,
            "mae": mean_difference(samples, "target", "predicted"),
        })

    reference = next(action for action in actions if action["id"] == "naive_finetune")
    for action in actions:
        if action["id"] == "seen_pretrain":
            continue
        for sample, expected in zip(action["samples"], reference["samples"]):
            if (sample["episode"], sample["frame"]) != (expected["episode"], expected["frame"]):
                raise ValueError(f"Action snapshot is not comparable: {action['name']}")
            if sample["target"] != expected["target"][:len(sample["target"])]:
                raise ValueError(f"Action target differs: {action['name']}")

    transitions = []
    for path in sorted((runs / "diagnostics").glob("preliminary_*_transitions.json")):
        row = json.loads(path.read_text())
        row.update(id=run_id(path.stem.removesuffix("_transitions")),
                   name=label(path.stem.removesuffix("_transitions")))
        transitions.append(row)

    controls = []
    for action in actions:
        samples = action["samples"]
        if samples and "zero_latent_actions" in samples[0]:
            controls.append({
                "id": action["id"],
                "name": action["name"],
                "values": [
                    {"name": "zero latent", "value": mean_difference(samples, "predicted", "zero_latent_actions")},
                    {"name": "reversed step order", "value": mean_difference(samples, "predicted", "reversed_latent_actions")},
                    {"name": "observed transition", "value": mean_difference(samples, "predicted", "true_latent_actions")},
                ],
            })

    proxy = defaultdict(list)
    for row in rows:
        if "eval" not in row:
            proxy[(row["method"], row["n_demos"])].append(row["success"])
    proxy_points = defaultdict(list)
    for (method, demos), values in proxy.items():
        proxy_points[method].append({"demos": demos, "success": sum(values) / len(values)})

    preliminary = [row for row in rows if "eval" in row]
    augmentation = None
    augmentation_path = runs / "diagnostics/augmentations/metadata.json"
    if augmentation_path.is_file():
        augmentation = json.loads(augmentation_path.read_text())
        for sample in augmentation["samples"]:
            image = augmentation_path.parent / sample["image"]
            sample["src"] = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()
    tiny = None
    tiny_path = runs / "latent_tiny_overfit"
    if (tiny_path / "run.json").is_file():
        tiny = {
            "summary": json.loads((tiny_path / "run.json").read_text()),
            "rows": read_jsonl(tiny_path / "metrics.jsonl"),
        }
    lapo = None
    lapo_path = runs / "lapo_pipeline_tiny_overfit"
    if (lapo_path / "run.json").is_file():
        lapo = {
            "summary": json.loads((lapo_path / "run.json").read_text()),
            "representation": read_jsonl(lapo_path / "metrics.jsonl"),
            "policy": read_jsonl(lapo_path / "policy_metrics.jsonl"),
        }
    return {
        "training": training,
        "actions": actions,
        "transitions": transitions,
        "controls": controls,
        "tiny": tiny,
        "lapo": lapo,
        "augmentation": augmentation,
        "proxy": [{"id": method, "name": label(method),
                   "points": sorted(points, key=lambda point: point["demos"])}
                  for method, points in proxy_points.items()],
        "rollouts": {
            "episodes": sum(row["n_episodes"] for row in preliminary),
            "successes": sum(round(row["success"] * row["n_episodes"]) for row in preliminary),
            "methods": len(preliminary),
        },
    }


TEMPLATE = r'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VLA Mac screening</title><style>
:root{--fg:#15202b;--mut:#5f6b76;--line:#d6dce2;--blue:#1d4ed8;--orange:#c2410c;--grey:#a9b1b9;--paper:#f7f8fa}
*{box-sizing:border-box}body{font:14px/1.42 system-ui;margin:0;color:var(--fg);background:#fff}
header{position:sticky;top:0;z-index:3;background:#fffffff2;backdrop-filter:blur(10px);border-bottom:1px solid var(--line);padding:10px max(16px,calc((100vw - 1460px)/2));display:grid;grid-template-columns:auto 1fr auto;gap:10px 18px;align-items:center}
h1{font-size:16px;margin:0;white-space:nowrap}.readout{display:flex;justify-content:flex-end;gap:18px;font-variant-numeric:tabular-nums}.readout span{white-space:nowrap}.readout b{display:block;font-size:16px}
main{max-width:1460px;margin:auto;padding:18px 16px 42px;display:grid;grid-template-columns:1fr 1fr;gap:24px 28px}.wide{grid-column:1/-1}section{border-top:1px solid var(--line);padding-top:12px;min-width:0}
h2{font-size:15px;margin:0 0 4px}.explain,.caption{color:var(--mut);margin:0 0 10px;max-width:105ch}.caption{margin-top:7px}.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:14px 18px}.metric-title{font-weight:600;margin-bottom:3px}
svg{display:block;width:100%;height:auto;background:var(--paper)}.axis{stroke:#9ca6af;stroke-width:1}.grid{stroke:#e5e8eb;stroke-width:1}.line{fill:none;stroke-width:2;opacity:.9}.target{fill:none;stroke:#111827;stroke-width:2.5}.dim{stroke:var(--grey);opacity:.17;stroke-width:1}.selected{stroke:var(--blue);opacity:1;stroke-width:3}.naive{stroke:var(--orange);stroke-dasharray:6 4;opacity:1}
.legend,.sample-controls{display:flex;flex-wrap:wrap;gap:5px}.legend{grid-column:1/-1}.legend button,.sample-controls button{font:inherit;border:1px solid var(--line);background:#fff;color:var(--mut);padding:4px 8px;border-radius:5px;cursor:pointer}.legend button.active,.sample-controls button.active{border-color:var(--blue);color:var(--blue);background:#eef4ff}.swatch{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}.sample-controls{margin:8px 0 12px}
.images{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px}.images figure{margin:0}.images img{display:block;width:100%;height:auto}.images figcaption{margin-top:5px;color:var(--mut)}
.tooltip{position:fixed;display:none;pointer-events:none;background:#111827;color:#fff;padding:6px 8px;border-radius:4px;font-size:12px;z-index:5}.provenance{color:var(--mut);margin-top:10px}.provenance a{color:var(--blue)}
@media(max-width:1000px){header{grid-template-columns:1fr}.readout{justify-content:flex-start}main{grid-template-columns:1fr}.wide{grid-column:auto}.images{grid-template-columns:repeat(3,1fr)}}@media(max-width:620px){.readout{display:grid;grid-template-columns:1fr 1fr}.images{grid-template-columns:1fr 1fr}}
</style><header><h1>VLA Mac screening</h1><div class="readout" id="readout"></div><div class="legend" id="method-filter"></div></header><main>
<section class="wide"><h2>Old proxy results</h2><p class="explain">Every measured method is visible. Lines exist only when the same method was evaluated at several demonstration budgets. These runs used the old proxy dataset.</p><svg id="proxy" viewBox="0 0 1180 390"></svg></section>
<section class="wide"><h2>Training dynamics</h2><p class="explain">Every run starts at 1.0. The shared logarithmic scale shows relative optimization progress despite different loss definitions. Episode-mask and padding-mask losses are omitted because they are identical in every recorded step.</p><svg id="training" viewBox="0 0 1180 390"></svg></section>
<section class="wide"><h2>Action error on fixed target-demo frames</h2><p class="explain">Mean absolute error over three frames, every chunk step, and seven action coordinates. Lower means the checkpoint fits these demonstration frames better; this is not held-out rollout quality.</p><svg id="action-error" viewBox="0 0 1180 500"></svg></section>
<section class="wide"><h2>Fine-tuning cost versus target-demo fit</h2><p class="explain">Only the four directly comparable fine-tuning choices are shown. Left trains fewer parameters; lower reconstructs the recorded actions better. The x-axis is logarithmic.</p><svg id="finetune-cost" viewBox="0 0 1180 420"></svg><p class="caption">Naive and longer train the same action expert; longer only doubles the steps. Full also trains the VLM and vision encoder. LoRA trains adapters while the base weights stay frozen.</p></section>
<section class="wide" id="augmentation-section"><h2>Image augmentations used in training</h2><p class="explain">Each frame receives three of six transforms: brightness and contrast 0.8 to 1.2, saturation 0.5 to 1.5, hue -0.05 to 0.05, sharpness 0.5 to 1.5, or affine rotation ±5° with translation up to 5%.</p><div class="images" id="augmentations"></div></section>
<section class="wide"><h2>Predicted action chunks</h2><p class="explain">Seven action coordinates are shown together on the same [-1, 1] scale. Black is the demonstrated target. Chunk 10 ends after step 9 by design.</p><div class="sample-controls" id="sample-controls"></div><div class="plots" id="action-plots"></div></section>
<section class="wide"><h2>Rejected raw visual-residual predictor</h2><p class="explain" id="cosine-note">This first implementation predicted the full visual-token residual directly. It remains here as a failed experiment.</p><div class="plots"><div><div class="metric-title">Preliminary run by chunk step</div><svg id="cosine" viewBox="0 0 580 320"></svg></div><div id="tiny-box"><div class="metric-title">Tiny-set overfit by optimizer step</div><svg id="tiny-overfit" viewBox="0 0 580 320"></svg><p class="caption" id="tiny-note"></p></div></div></section>
<section class="wide" id="lapo-section"><h2>Continuous LAPO tiny-set gate</h2><p class="explain">The inverse model compresses each adjacent visual-token pair into 32 values. The forward model must reconstruct the token change from the current frame and this latent.</p><div class="plots"><div><div class="metric-title">Forward reconstruction</div><svg id="lapo-reconstruction" viewBox="0 0 580 320"></svg></div><div><div class="metric-title">Representation and policy alignment</div><svg id="lapo-cosine" viewBox="0 0 580 320"></svg></div></div><p class="caption" id="lapo-note"></p></section>
<section class="wide"><h2>Rejected latent action controls</h2><p class="explain">These controls belong to the rejected raw-residual implementation and are kept only as historical evidence.</p><div class="plots" id="controls"></div></section>
<section class="wide"><details class="provenance"><summary>Raw artifacts</summary><p><a href="results.jsonl">results.jsonl</a> contains rollout cells. Training metrics and action snapshots remain in their run directories.</p></details></section>
</main><div class="tooltip" id="tip"></div><script>
const DATA=__DATA__,S="http://www.w3.org/2000/svg",BLUE="#1d4ed8",ORANGE="#c2410c",GREY="#a9b1b9",tip=document.querySelector("#tip");
const node=(tag,a={},text="")=>{const n=document.createElementNS(S,tag);for(const[k,v]of Object.entries(a))n.setAttribute(k,v);if(text!=="")n.textContent=text;return n};
const palette=["#2563eb","#d97706","#0f766e","#7c3aed","#be123c","#0369a1","#4d7c0f","#a21caf","#92400e","#475569","#0891b2","#c026d3","#4338ca"];
const allMethods=[...new Map(DATA.training.concat(DATA.actions).map(x=>[x.id,{id:x.id,name:x.name}])).values()],methods=DATA.actions.filter(x=>x.id!=="seen_pretrain").map(x=>({id:x.id,name:x.name})),colors=Object.fromEntries(allMethods.map((m,i)=>[m.id,palette[i%palette.length]]));
let selected=null,sampleIndex=0;
function showTip(event,html){tip.innerHTML=html;tip.style.display="block";tip.style.left=Math.min(event.clientX+12,innerWidth-270)+"px";tip.style.top=event.clientY+12+"px"}function hideTip(){tip.style.display="none"}
function style(id){if(!selected)return{stroke:colors[id]||GREY,cls:"line"};if(id===selected)return{stroke:BLUE,cls:"line selected"};if(id==="naive_finetune")return{stroke:ORANGE,cls:"line naive"};return{stroke:GREY,cls:"line dim"}}
function plotFrame(svg,{x=[0,1],y=[0,1],xTicks,yTicks,xLabel,yLabel,logY=false}){svg.replaceChildren();const W=+svg.viewBox.baseVal.width,H=+svg.viewBox.baseVal.height,p={l:58,r:18,t:18,b:46},tx=v=>p.l+(v-x[0])/(x[1]-x[0])*(W-p.l-p.r),transform=v=>logY?Math.log10(v):v,yy=y.map(transform),ty=v=>H-p.b-(transform(v)-yy[0])/(yy[1]-yy[0])*(H-p.t-p.b);for(const v of yTicks){const py=ty(v);svg.append(node("line",{x1:p.l,y1:py,x2:W-p.r,y2:py,class:"grid"}),node("text",{x:p.l-8,y:py+4,"text-anchor":"end",fill:"#5f6b76","font-size":11},String(v)))}for(const v of xTicks){const px=tx(v);svg.append(node("text",{x:px,y:H-14,"text-anchor":"middle",fill:"#5f6b76","font-size":11},String(v)))}svg.append(node("line",{x1:p.l,y1:p.t,x2:p.l,y2:H-p.b,class:"axis"}),node("line",{x1:p.l,y1:H-p.b,x2:W-p.r,y2:H-p.b,class:"axis"}));svg.append(node("text",{x:W-p.r,y:H-3,"text-anchor":"end",fill:"#5f6b76","font-size":12},xLabel),node("text",{x:p.l+5,y:p.t+13,fill:"#5f6b76","font-size":12},yLabel));return{X:tx,Y:ty,W,H,p}}
function path(svg,rows,X,Y,id,xKey,yKey,name){const s=style(id),p=node("path",{d:rows.map((r,i)=>(i?"L":"M")+X(r[xKey])+","+Y(r[yKey])).join(" "),class:s.cls,stroke:s.stroke});p.style.cursor="pointer";p.onmouseenter=e=>showTip(e,`<b>${name}</b>`);p.onmousemove=e=>showTip(e,`<b>${name}</b>`);p.onmouseleave=hideTip;p.onclick=()=>select(id);svg.append(p)}
function point(svg,x,y,color,html,r=4){const c=node("circle",{cx:x,cy:y,r,fill:color});c.onmousemove=e=>showTip(e,html);c.onmouseleave=hideTip;svg.append(c);return c}
function select(id){selected=selected===id?null:id;render()}
function renderLegend(){const root=document.querySelector("#method-filter");root.replaceChildren();for(const m of methods){const b=document.createElement("button"),s=document.createElement("i");s.className="swatch";s.style.background=selected?(m.id===selected?BLUE:m.id==="naive_finetune"?ORANGE:GREY):colors[m.id];b.append(s,m.name);b.className=m.id===selected?"active":"";b.onclick=()=>select(m.id);root.append(b)}}
function renderProxy(){const svg=document.querySelector("#proxy"),{X,Y}=plotFrame(svg,{x:[0,25],y:[0,1],xTicks:[0,5,10,15,20,25],yTicks:[0,.25,.5,.75,1],xLabel:"demonstrations",yLabel:"rollout success"});DATA.proxy.forEach((m,i)=>{const color=palette[i%palette.length];if(m.points.length>1)svg.append(node("path",{d:m.points.map((p,j)=>(j?"L":"M")+X(p.demos)+","+Y(p.success)).join(" "),class:"line",stroke:color}));m.points.forEach(p=>point(svg,X(p.demos),Y(p.success),color,`<b>${m.name}</b><br>${p.demos} demos<br>success ${p.success.toFixed(2)}`))})}
function renderTraining(){const svg=document.querySelector("#training"),{X,Y}=plotFrame(svg,{x:[0,210],y:[.1,4],xTicks:[0,50,100,150,200],yTicks:[.1,.25,.5,1,2,4],xLabel:"optimizer step",yLabel:"loss / step-1 loss",logY:true});svg.append(node("line",{x1:X(0),y1:Y(1),x2:X(210),y2:Y(1),stroke:"#6b7280","stroke-dasharray":"3 4"}));for(const run of DATA.training)path(svg,run.rows,X,Y,run.id,"step","relative_loss",run.name)}
function targetActions(){return DATA.actions.find(x=>x.id==="naive_finetune")||DATA.actions.find(x=>x.id!=="seen_pretrain")}
function comparableActions(){return DATA.actions.filter(x=>x.id!=="seen_pretrain")}
function renderActionError(){const rows=comparableActions().slice().sort((a,b)=>a.mae-b.mae),svg=document.querySelector("#action-error"),W=1180,H=500,p={l:220,r:55,t:18,b:38},max=.35,X=v=>p.l+v/max*(W-p.l-p.r),gap=(H-p.t-p.b)/rows.length;svg.replaceChildren();for(let i=0;i<=7;i++){const v=i*.05,x=X(v);svg.append(node("line",{x1:x,y1:p.t,x2:x,y2:H-p.b,class:"grid"}),node("text",{x,y:H-14,"text-anchor":"middle",fill:"#5f6b76","font-size":11},v.toFixed(2)))}rows.forEach((r,i)=>{const y=p.t+gap*(i+.5),s=style(r.id);svg.append(node("text",{x:p.l-12,y:y+4,"text-anchor":"end",fill:s.stroke,"font-size":12},r.name),node("line",{x1:p.l,y1:y,x2:X(r.mae),y2:y,stroke:"#cfd5dc"}));const c=point(svg,X(r.mae),y,s.stroke,`<b>${r.name}</b><br>training-demo action MAE ${r.mae.toFixed(4)}`,5);c.onclick=()=>select(r.id)});svg.append(node("text",{x:W-p.r,y:H-3,"text-anchor":"end",fill:"#5f6b76","font-size":12},"mean absolute error →"))}
function renderFinetuneCost(){const ids=new Set(["naive_finetune","longer_finetune","full_finetune","lora_r32"]),rows=DATA.training.filter(x=>ids.has(x.id)).map(run=>({...run,mae:DATA.actions.find(x=>x.id===run.id).mae})),svg=document.querySelector("#finetune-cost"),W=1180,H=420,p={l:70,r:30,t:24,b:52},lo=1e6,hi=5e8,X=v=>p.l+(Math.log10(v)-Math.log10(lo))/(Math.log10(hi)-Math.log10(lo))*(W-p.l-p.r),Y=v=>H-p.b-(v-.12)/.1*(H-p.t-p.b);svg.replaceChildren();for(const v of [1e6,1e7,1e8,5e8]){const x=X(v);svg.append(node("line",{x1:x,y1:p.t,x2:x,y2:H-p.b,class:"grid"}),node("text",{x,y:H-18,"text-anchor":"middle",fill:"#5f6b76","font-size":11},v>=1e8?`${v/1e6}M`:v>=1e6?`${v/1e6}M`:v))}for(const v of [.12,.14,.16,.18,.20,.22]){const y=Y(v);svg.append(node("line",{x1:p.l,y1:y,x2:W-p.r,y2:y,class:"grid"}),node("text",{x:p.l-9,y:y+4,"text-anchor":"end",fill:"#5f6b76","font-size":11},v.toFixed(2)))}rows.forEach(run=>{const s=style(run.id),x=X(run.trainable_parameters),y=Y(run.mae),right=x>W-190;point(svg,x,y,s.stroke,`<b>${run.name}</b><br>${(run.trainable_parameters/1e6).toFixed(2)}M trainable parameters<br>${run.steps} steps<br>action MAE ${run.mae.toFixed(4)}`,6);const t=node("text",{x:x+(right?-9:9),y:y-8,"text-anchor":right?"end":"start",fill:s.stroke,"font-size":12,"font-weight":600,"paint-order":"stroke",stroke:"#f7f8fa","stroke-width":4},run.name);svg.append(t)});svg.append(node("text",{x:W-p.r,y:H-3,"text-anchor":"end",fill:"#5f6b76","font-size":12},"trainable parameters, log scale →"),node("text",{x:p.l+5,y:p.t+13,fill:"#5f6b76","font-size":12},"action MAE, lower is better"))}
function renderAugmentations(){const root=document.querySelector("#augmentations"),section=document.querySelector("#augmentation-section"),data=DATA.augmentation;if(!data){section.style.display="none";return}root.replaceChildren();for(const sample of data.samples){const figure=document.createElement("figure"),image=document.createElement("img"),caption=document.createElement("figcaption");image.src=sample.src;image.alt=sample.name;caption.textContent=sample.transforms.length?sample.transforms.join(" + "):"original";figure.append(image,caption);root.append(figure)}}
function renderSampleControls(){const root=document.querySelector("#sample-controls"),samples=targetActions().samples;root.replaceChildren();samples.forEach((s,i)=>{const b=document.createElement("button");b.textContent=`frame ${s.frame}`;b.className=i===sampleIndex?"active":"";b.onclick=()=>{sampleIndex=i;renderActions();renderSampleControls()};root.append(b)})}
function renderActions(){const root=document.querySelector("#action-plots"),base=targetActions().samples[sampleIndex],runs=comparableActions();root.replaceChildren();for(let dim=0;dim<7;dim++){const box=document.createElement("div"),title=document.createElement("div"),svg=node("svg",{viewBox:"0 0 340 205"});title.className="metric-title";title.textContent=`action[${dim}]`;box.append(title,svg);const {X,Y}=plotFrame(svg,{x:[0,49],y:[-1.05,1.05],xTicks:[0,10,20,30,40,49],yTicks:[-1,0,1],xLabel:"chunk step",yLabel:"value"}),target=base.target.map((a,i)=>({step:i,value:a[dim]}));svg.append(node("path",{d:target.map((r,i)=>(i?"L":"M")+X(r.step)+","+Y(r.value)).join(" "),class:"target"}));for(const run of runs){const sample=run.samples[sampleIndex],rows=sample.predicted.map((a,i)=>({step:i,value:a[dim]}));path(svg,rows,X,Y,run.id,"step","value",run.name);rows.forEach(r=>{const hit=node("circle",{cx:X(r.step),cy:Y(r.value),r:4,fill:"transparent"});hit.onmousemove=e=>showTip(e,`<b>${run.name}</b><br>action[${dim}], step ${r.step}<br>prediction ${r.value.toFixed(4)}<br>target ${target[r.step]?.value.toFixed(4)??"not recorded"}`);hit.onmouseleave=hideTip;svg.append(hit)})}root.append(box)}}
function renderCosine(){const svg=document.querySelector("#cosine"),d=DATA.transitions[0];if(!d)return;const values=d.cosine_by_step_and_view.flat(),mean=values.reduce((a,b)=>a+b,0)/values.length;document.querySelector("#cosine-note").textContent=`The rejected predictor reached mean cosine ${mean.toFixed(5)} against the observed visual-token change. A value near zero means the directions are unrelated.`;const {X,Y}=plotFrame(svg,{x:[0,49],y:[-1,1],xTicks:[0,10,20,30,40,49],yTicks:[-1,-.5,0,.5,1],xLabel:"chunk step",yLabel:"cosine"});for(let camera=0;camera<2;camera++){const rows=d.cosine_by_step_and_view.map((v,i)=>({step:i,value:v[camera]})),color=camera?ORANGE:BLUE;svg.append(node("path",{d:rows.map((r,i)=>(i?"L":"M")+X(r.step)+","+Y(r.value)).join(" "),class:"line",stroke:color,"stroke-dasharray":camera?"6 4":""}));rows.forEach(r=>point(svg,X(r.step),Y(r.value),color,`<b>${d.name}, camera ${camera+1}</b><br>step ${r.step}<br>cosine ${r.value.toFixed(5)}`,2.5))}}
function renderTiny(){const box=document.querySelector("#tiny-box"),d=DATA.tiny;if(!d){box.style.display="none";return}const svg=document.querySelector("#tiny-overfit"),rows=[{step:0,cosine:d.summary.initial_transition.cosine},...d.rows],{X,Y}=plotFrame(svg,{x:[0,d.summary.steps_per_phase],y:[-.1,1],xTicks:[0,250,500,750,1000],yTicks:[0,.25,.5,.75,1],xLabel:"optimizer step",yLabel:"cosine"});svg.append(node("path",{d:rows.map((r,i)=>(i?"L":"M")+X(r.step)+","+Y(r.cosine)).join(" "),class:"line",stroke:BLUE}));rows.forEach(r=>point(svg,X(r.step),Y(r.cosine),BLUE,`<b>Tiny-set overfit</b><br>step ${r.step}<br>cosine ${r.cosine.toFixed(4)}`,2.5));const s=d.summary;document.querySelector("#tiny-note").textContent=`Decoder MAE with observed latent: ${s.true_latent_action_mae.toFixed(3)}. With predicted latent: ${s.predicted_latent_action_mae.toFixed(3)}. With zero latent: ${s.zero_latent_action_mae.toFixed(3)}. The predictor did not memorize three fixed windows.`}
function renderLapo(){const section=document.querySelector("#lapo-section"),d=DATA.lapo;if(!d){section.style.display="none";return}const lossSvg=document.querySelector("#lapo-reconstruction"),lossRows=[{step:0,loss:d.summary.initial.loss,zero_loss:d.summary.initial.zero_loss,shuffled_loss:d.summary.initial.shuffled_loss},...d.representation],frame=plotFrame(lossSvg,{x:[0,d.summary.steps],y:[0,4],xTicks:[0,250,500,750,1000],yTicks:[0,1,2,3,4],xLabel:"optimizer step",yLabel:"masked MSE"}),series=[{key:"loss",name:"true z",color:BLUE},{key:"shuffled_loss",name:"shuffled z",color:ORANGE},{key:"zero_loss",name:"zero z",color:"#64748b"}];for(const item of series){lossSvg.append(node("path",{d:lossRows.map((r,i)=>(i?"L":"M")+frame.X(r.step)+","+frame.Y(r[item.key])).join(" "),class:"line",stroke:item.color}));for(const r of lossRows)point(lossSvg,frame.X(r.step),frame.Y(r[item.key]),item.color,`<b>${item.name}</b><br>step ${r.step}<br>MSE ${r[item.key].toFixed(4)}`,2)}const cosineSvg=document.querySelector("#lapo-cosine"),rep=[{step:0,cosine:d.summary.initial.cosine},...d.representation],policy=d.policy,aligned=plotFrame(cosineSvg,{x:[0,d.summary.steps],y:[0,1],xTicks:[0,250,500,750,1000],yTicks:[0,.25,.5,.75,1],xLabel:"optimizer step",yLabel:"cosine"});for(const item of [{rows:rep,name:"forward delta",color:BLUE},{rows:policy,name:"predicted latent",color:ORANGE}]){cosineSvg.append(node("path",{d:item.rows.map((r,i)=>(i?"L":"M")+aligned.X(r.step)+","+aligned.Y(r.cosine)).join(" "),class:"line",stroke:item.color}));for(const r of item.rows)point(cosineSvg,aligned.X(r.step),aligned.Y(r.cosine),item.color,`<b>${item.name}</b><br>step ${r.step}<br>cosine ${r.cosine.toFixed(5)}`,2)}const a=d.summary.action_mae;document.querySelector("#lapo-note").textContent=`Final forward MSE: true z ${d.summary.final.loss.toFixed(3)}, shuffled z ${d.summary.final.shuffled_loss.toFixed(3)}, zero z ${d.summary.final.zero_loss.toFixed(3)}. Action MAE: true z ${a.true_latent.toFixed(3)}, predicted z ${a.predicted_latent.toFixed(3)}, zero z ${a.zero_latent.toFixed(3)}.`}
function renderControls(){const root=document.querySelector("#controls"),specs=[{name:"Remove transition",key:"zero latent",note:"Nonzero means the latent changes action values."},{name:"Reverse 50 steps",key:"reversed step order",note:"Near zero means the decoder ignores step order."},{name:"Use observed transition",key:"observed transition",note:"Nonzero means predicted and observed transitions decode differently."}];root.replaceChildren();for(const spec of specs){const box=document.createElement("div"),title=document.createElement("div"),svg=node("svg",{viewBox:"0 0 340 205"}),caption=document.createElement("p");title.className="metric-title";title.textContent=spec.name;caption.className="caption";caption.textContent=spec.note;box.append(title,svg,caption);const rows=DATA.controls.map(run=>({id:run.id,name:run.name.replace("Latent with ","With ").replace("Latent without ","Without "),value:run.values.find(x=>x.name===spec.key).value})),W=340,H=205,p={l:105,r:20,t:18,b:38},X=v=>p.l+v/.06*(W-p.l-p.r),gap=(H-p.t-p.b)/rows.length;for(let i=0;i<=3;i++){const v=i*.02,x=X(v);svg.append(node("line",{x1:x,y1:p.t,x2:x,y2:H-p.b,class:"grid"}),node("text",{x,y:H-14,"text-anchor":"middle",fill:"#5f6b76","font-size":10},v.toFixed(2)))}rows.forEach((r,i)=>{const y=p.t+gap*(i+.5),s=style(r.id);svg.append(node("text",{x:p.l-8,y:y+4,"text-anchor":"end",fill:s.stroke,"font-size":11},r.name),node("line",{x1:p.l,y1:y,x2:X(r.value),y2:y,stroke:"#cfd5dc"}));point(svg,X(r.value),y,s.stroke,`<b>${r.name}: ${spec.name}</b><br>mean action change ${r.value.toFixed(6)}`,4)});root.append(box)}}
function render(){renderLegend();renderTraining();renderActionError();renderFinetuneCost();renderActions();renderCosine();renderTiny();renderLapo();renderControls()}const r=DATA.rollouts;document.querySelector("#readout").innerHTML=`<span>target rollout success<b>${r.successes}/${r.episodes}</b></span><span>evaluated variants<b>${r.methods}</b></span><span>training runs<b>${DATA.training.length}</b></span>`;renderProxy();renderAugmentations();renderSampleControls();render();
</script>'''


def main():
    data = load_data()
    output = pathlib.Path("runs/report.html")
    output.write_text(TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False, allow_nan=False)))
    print(f"{len(data['training'])} training runs, {len(data['actions'])} action snapshots, "
          f"{len(data['proxy'])} proxy methods -> {output}")


if __name__ == "__main__":
    main()
