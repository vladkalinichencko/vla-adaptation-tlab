"""Build the final A100 SmolVLA report from downloaded run artifacts."""

import json
import re
from pathlib import Path

RUNS = Path("runs")

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def keep_curve(rows: list[dict], every: int = 10) -> list[dict]:
    return [row for i, row in enumerate(rows) if i == 0 or i == len(rows) - 1 or (i + 1) % every == 0]

def mean_difference(samples: list[dict], left: str, right: str) -> float:
    values = []
    for sample in samples:
        for left_step, right_step in zip(sample[left], sample[right], strict=True):
            values.extend(abs(a - b) for a, b in zip(left_step, right_step, strict=True))
    return sum(values) / len(values)

def action_run(root: Path, name: str) -> dict:
    samples = json.loads((root / f"actions_final_{name}.json").read_text())
    return {
        "id": name,
        "name": {
            "mix_seen_t0_n5_s0": "Mix seen",
            "lora_r32_t0_n5_s0": "LoRA r=32",
            "lapo_video_only_t0_n5_s0": "Continuous LAPO",
            "zero_shot_t0": "Zero-shot",
            "wrong_instruction_t0": "Wrong instruction",
        }[name],
        "samples": samples,
        "mae": mean_difference(samples, "target", "predicted"),
    }

def rollout(root: Path, name: str, wanted_success: bool | None = None) -> dict:
    data = json.loads((root / f"eval_final_{name}.json").read_text())
    metrics = data["per_task"][0]["metrics"]
    successes = metrics["successes"]
    index = 0 if wanted_success is None else next(i for i, value in enumerate(successes) if value == wanted_success)
    video = next((root / f"videos_final_{name}").glob(f"*/eval_episode_{index}.mp4"), None)
    return {
        "name": name.replace("_", " "),
        "success": successes[index],
        "video": str(video.relative_to(Path("."))) if video else None,
    }

def load_data() -> dict:
    summary_path = sorted((RUNS / "final").glob("summary.json"))[-1]
    root = summary_path.parent
    summary = json.loads(summary_path.read_text())

    seen = keep_curve(read_jsonl(root / "metrics_final_seen_pretrain.jsonl"))
    adaptation = []
    pattern = re.compile(r"metrics_final_(mix_seen|lora_r32)_t(\d+)_n(\d+)_s(\d+)\.jsonl")
    for path in sorted(root.glob("metrics_final_*.jsonl")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        method, task, demos, seed = match.groups()
        adaptation.append({
            "method": method,
            "task": int(task),
            "demos": int(demos),
            "seed": int(seed),
            "rows": keep_curve(read_jsonl(path)),
        })

    actions = [action_run(root, name) for name in (
        "mix_seen_t0_n5_s0",
        "lora_r32_t0_n5_s0",
        "lapo_video_only_t0_n5_s0",
        "zero_shot_t0",
        "wrong_instruction_t0",
    )]
    reference = actions[0]["samples"]
    for run in actions[1:]:
        assert [(x["episode"], x["frame"]) for x in run["samples"]] == [
            (x["episode"], x["frame"]) for x in reference
        ]
        assert [x["target"] for x in run["samples"]] == [x["target"] for x in reference]

    language = []
    lapo_controls = []
    for task in range(3):
        zero = json.loads((root / f"actions_final_zero_shot_t{task}.json").read_text())
        wrong = json.loads((root / f"actions_final_wrong_instruction_t{task}.json").read_text())
        language.append({"task": task, "change": mean_difference(
            [{"zero": a["predicted"], "wrong": b["predicted"]} for a, b in zip(zero, wrong, strict=True)],
            "zero", "wrong",
        )})
        samples = json.loads((root / f"actions_final_lapo_video_only_t{task}_n5_s0.json").read_text())
        lapo_controls.append({
            "task": task,
            "predicted_mae": mean_difference(samples, "target", "predicted"),
            "true_mae": mean_difference(samples, "target", "true_latent_actions"),
            "zero_mae": mean_difference(samples, "target", "zero_latent_actions"),
        })

    lapo = {
        "representation": keep_curve(read_jsonl(root / "metrics_final_lapo_representation.jsonl")),
        "policy": keep_curve(read_jsonl(root / "metrics_final_lapo_policy.jsonl")),
        "adaptation": [
            {"task": task, "rows": keep_curve(read_jsonl(
                root / f"metrics_final_lapo_video_only_t{task}_n5_s0.jsonl"
            ))}
            for task in range(3)
        ],
        "controls": lapo_controls,
    }
    videos = [
        rollout(root, "zero_shot_t0"),
        rollout(root, "wrong_instruction_t0"),
        rollout(root, "lapo_video_only_t0_n5_s0"),
        rollout(root, "mix_seen_t0_n25_s0", True),
        rollout(root, "lora_r32_t0_n25_s0", True),
    ]
    return {
        "summary": summary,
        "seen": seen,
        "adaptation": adaptation,
        "actions": actions,
        "language": language,
        "lapo": lapo,
        "videos": videos,
    }

TEMPLATE = r'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SmolVLA final A100 runs</title><style>
:root{--fg:#15202b;--mut:#5f6b76;--line:#d6dce2;--blue:#1d4ed8;--orange:#c2410c;--purple:#7c3aed;--paper:#f7f8fa}
*{box-sizing:border-box}body{font:14px/1.42 system-ui;margin:0;color:var(--fg)}header{position:sticky;top:0;z-index:3;background:#fffffff2;backdrop-filter:blur(10px);border-bottom:1px solid var(--line);padding:10px max(16px,calc((100vw - 1400px)/2));display:flex;gap:24px;align-items:center}h1{font-size:16px;margin:0}.readout{display:flex;gap:20px}.readout b{display:block;font-size:16px}main{max-width:1400px;margin:auto;padding:18px 16px 42px;display:grid;grid-template-columns:1fr 1fr;gap:24px 28px}.wide{grid-column:1/-1}section{border-top:1px solid var(--line);padding-top:12px;min-width:0}h2{font-size:15px;margin:0 0 4px}.note{color:var(--mut);margin:0 0 10px}.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}.metric{font-weight:600;margin-bottom:3px}svg{display:block;width:100%;height:auto;background:var(--paper)}.axis{stroke:#9ca6af}.grid{stroke:#e5e8eb}.line{fill:none;stroke-width:1.5;opacity:.55}.target{fill:none;stroke:#111827;stroke-width:2.5}.buttons{display:flex;gap:5px;margin:8px 0 12px}.buttons button{font:inherit;border:1px solid var(--line);background:#fff;padding:4px 8px;border-radius:5px}.buttons button.active{border-color:var(--blue);color:var(--blue)}.videos{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}.videos figure{margin:0}.videos video{width:100%;display:block;background:#000}.videos figcaption{margin-top:5px}.tooltip{position:fixed;display:none;pointer-events:none;background:#111827;color:#fff;padding:6px 8px;border-radius:4px;font-size:12px;z-index:5}@media(max-width:850px){main{grid-template-columns:1fr}.wide{grid-column:auto}header{align-items:flex-start;flex-direction:column}.readout{flex-wrap:wrap}}
</style><header><h1>SmolVLA, final A100 runs</h1><div class="readout" id="readout"></div></header><main>
<section class="wide"><h2>Cost curve</h2><p class="note">Three target tasks, two training seeds and 20 rollout episodes per cell.</p><div class="plots" id="cost"></div></section>
<section><h2>Seen pretrain</h2><p class="note">The actual 5000-step A100 loss.</p><svg id="seen" viewBox="0 0 680 340"></svg></section>
<section><h2>Target adaptation</h2><p class="note">All 36 A100 Mix and LoRA runs. Curves are raw action loss, not normalized proxy values.</p><svg id="adaptation" viewBox="0 0 680 340"></svg></section>
<section class="wide"><h2>Action fit on fixed target frames</h2><p class="note">Task 0, five demonstrations, seed 0. Lower is better. This is demo fit, not rollout success.</p><svg id="action-error" viewBox="0 0 1180 320"></svg></section>
<section class="wide"><h2>Predicted action chunks</h2><p class="note">The black line is the demonstrated 50-step action chunk. Every coloured line comes from the final A100 checkpoint.</p><div class="buttons" id="frames"></div><div class="plots" id="actions"></div></section>
<section><h2>Language control</h2><p class="note">Mean action change on the same observation after replacing the instruction. Rollout success was zero for both instructions.</p><svg id="language" viewBox="0 0 680 340"></svg></section>
<section><h2>Continuous LAPO action controls</h2><p class="note">Action MAE using predicted, observed and zero latent on the same target frames.</p><svg id="lapo-controls" viewBox="0 0 680 340"></svg></section>
<section class="wide"><h2>Continuous LAPO training</h2><p class="note">All three phases from the final A100 run. The representation stage improved only weakly, while the latent policy fitted that representation.</p><div class="plots"><div><div class="metric">Representation</div><svg id="lapo-representation" viewBox="0 0 430 300"></svg></div><div><div class="metric">Latent policy</div><svg id="lapo-policy" viewBox="0 0 430 300"></svg></div><div><div class="metric">Target decoder, five demos</div><svg id="lapo-adaptation" viewBox="0 0 430 300"></svg></div></div></section>
<section class="wide"><h2>Final A100 rollouts</h2><p class="note">These videos are from the same checkpoints and eval JSON used above.</p><div class="videos" id="videos"></div></section>
</main><div class="tooltip" id="tip"></div><script>
const D=__DATA__,S="http://www.w3.org/2000/svg",C={mix_seen:"#1d4ed8",lora_r32:"#c2410c",lapo:"#7c3aed",zero:"#64748b",wrong:"#0f766e"},tip=document.querySelector("#tip");
const n=(tag,a={},text="")=>{const e=document.createElementNS(S,tag);for(const[k,v]of Object.entries(a))e.setAttribute(k,v);e.textContent=text;return e};
function frame(svg,xmax,ymax,xlabel,ylabel,xticks=5,yticks=4){svg.replaceChildren();const W=svg.viewBox.baseVal.width,H=svg.viewBox.baseVal.height,p={l:78,r:26,t:18,b:45},X=v=>p.l+v/xmax*(W-p.l-p.r),Y=v=>H-p.b-v/ymax*(H-p.t-p.b),ydiv=Math.max(yticks,1);for(let i=0;i<=yticks;i++){const v=ymax*i/ydiv,y=Y(v);svg.append(n("line",{x1:p.l,y1:y,x2:W-p.r,y2:y,class:"grid"}),n("text",{x:p.l-8,y:y+4,"text-anchor":"end",fill:"#5f6b76","font-size":11},v.toFixed(ymax<1?2:1)))}for(let i=0;i<=xticks;i++){const v=xmax*i/xticks,x=X(v);svg.append(n("text",{x,y:H-14,"text-anchor":"middle",fill:"#5f6b76","font-size":11},v.toFixed(xmax<1?3:xmax<10?1:0)))}svg.append(n("line",{x1:p.l,y1:p.t,x2:p.l,y2:H-p.b,class:"axis"}),n("line",{x1:p.l,y1:H-p.b,x2:W-p.r,y2:H-p.b,class:"axis"}),n("text",{x:W-p.r,y:H-3,"text-anchor":"end",fill:"#5f6b76","font-size":12},xlabel),n("text",{x:p.l+5,y:p.t+13,fill:"#5f6b76","font-size":12},ylabel));return{X,Y,W,H,p}}
function line(svg,rows,f,x,y,color,name,width=1.5){const path=n("path",{d:rows.map((r,i)=>(i?"L":"M")+f.X(r[x])+","+f.Y(r[y])).join(" "),class:"line",stroke:color,"stroke-width":width});path.onmousemove=e=>{tip.style.display="block";tip.style.left=e.clientX+12+"px";tip.style.top=e.clientY+12+"px";tip.textContent=name};path.onmouseleave=()=>tip.style.display="none";svg.append(path)}
function legend(svg,items){const W=svg.viewBox.baseVal.width;let x=62,y=14;for(const[color,label,shape]of items){if(shape==="dot"){svg.append(n("circle",{cx:x+5,cy:y-4,r:4,fill:color}))}else{svg.append(n("line",{x1:x,y1:y-4,x2:x+14,y2:y-4,stroke:color,"stroke-width":2.5,"stroke-linecap":"round"}))}const text=n("text",{x:x+20,y,fill:"#5f6b76","font-size":11},label);svg.append(text);x+=26+label.length*5.6;if(x>W-70){x=62;y+=14}}}
function bars(svg,rows,key,max,color){const f=frame(svg,max,rows.length,"mean absolute error","",4,0),gap=(f.H-f.p.t-f.p.b)/rows.length;rows.forEach((r,i)=>{const y=f.p.t+gap*(i+.5);svg.append(n("text",{x:f.p.l-8,y:y+4,"text-anchor":"end",fill:"#334155","font-size":11},r.name),n("line",{x1:f.p.l,y1:y,x2:f.X(r[key]),y2:y,stroke:"#cfd5dc","stroke-width":2}),n("circle",{cx:f.X(r[key]),cy:y,r:5,fill:color(r,i)}),n("text",{x:f.X(r[key])+8,y:y+4,fill:"#334155","font-size":11},r[key].toFixed(3)))})}
function renderCost(){const root=document.querySelector("#cost"),s=D.summary,panels=[["All tasks",null],["Task 0",0],["Task 1",1],["Task 2",2]];for(const[title,task]of panels){const box=document.createElement("div"),h=document.createElement("div"),svg=n("svg",{viewBox:"0 0 330 250"});h.className="metric";h.textContent=title;box.append(h,svg);const f=frame(svg,25,1,"demonstrations","success",5,4);for(const[id,name]of [["mix_seen","Mix seen"],["lora_r32","LoRA r=32"]]){const values=task===null?s.mean_success[id]:s.task_mean_success[id][task],rows=[5,10,25].map(d=>({d,v:values[d]}));line(svg,rows,f,"d","v",C[id],name,2.5);rows.forEach(r=>svg.append(n("circle",{cx:f.X(r.d),cy:f.Y(r.v),r:4,fill:C[id]})))}const keys=[[C.mix_seen,"Mix seen","line"],[C.lora_r32,"LoRA r=32","line"]];if(task===null){svg.append(n("circle",{cx:f.X(0),cy:f.Y(0),r:4,fill:C.zero}),n("circle",{cx:f.X(5),cy:f.Y(0),r:4,fill:C.lapo}));keys.push([C.zero,"Zero-shot","dot"],[C.lapo,"LAPO, 5 demos","dot"])}legend(svg,keys);root.append(box)}}
function renderTraining(){let svg=document.querySelector("#seen"),f=frame(svg,5000,3.5,"optimizer step","loss",5,7);line(svg,D.seen,f,"step","loss",C.mix_seen,"Seen pretrain",2);svg=document.querySelector("#adaptation");f=frame(svg,1250,2,"optimizer step","action loss",5,4);for(const r of D.adaptation)line(svg,r.rows,f,"step","loss",C[r.method],`${r.method}, task ${r.task}, ${r.demos} demos, seed ${r.seed}`)}
function renderActionError(){bars(document.querySelector("#action-error"),D.actions.map(x=>({name:x.name,mae:x.mae})),"mae",.4,(r,i)=>[C.mix_seen,C.lora_r32,C.lapo,C.zero,C.wrong][i])}
let selectedFrame=0;function renderActions(){const root=document.querySelector("#actions"),buttons=document.querySelector("#frames");root.replaceChildren();buttons.replaceChildren();D.actions[0].samples.forEach((s,i)=>{const b=document.createElement("button");b.textContent=`frame ${s.frame}`;b.className=i===selectedFrame?"active":"";b.onclick=()=>{selectedFrame=i;renderActions()};buttons.append(b)});for(let dim=0;dim<7;dim++){const box=document.createElement("div"),h=document.createElement("div"),svg=n("svg",{viewBox:"0 0 330 210"});h.className="metric";h.textContent=`action[${dim}]`;box.append(h,svg);const f=frame(svg,49,2.1,"chunk step","value + 1.05",5,4),shift=rows=>rows.map((v,i)=>({step:i,value:v[dim]+1.05})),target=shift(D.actions[0].samples[selectedFrame].target);svg.append(n("path",{d:target.map((r,i)=>(i?"L":"M")+f.X(r.step)+","+f.Y(r.value)).join(" "),class:"target"}));D.actions.forEach((run,i)=>line(svg,shift(run.samples[selectedFrame].predicted),f,"step","value",[C.mix_seen,C.lora_r32,C.lapo,C.zero,C.wrong][i],run.name));root.append(box)}}
function renderLanguage(){bars(document.querySelector("#language"),D.language.map(x=>({name:`task ${x.task}`,value:x.change})),"value",.1,()=>C.wrong)}
function renderLapo(){bars(document.querySelector("#lapo-controls"),D.lapo.controls.flatMap(x=>[["predicted",x.predicted_mae,C.lapo],["observed",x.true_mae,C.mix_seen],["zero",x.zero_mae,C.zero]].map(v=>({name:`task ${x.task}, ${v[0]} z`,value:v[1],color:v[2]}))),"value",.4,r=>r.color);let svg=document.querySelector("#lapo-representation"),f=frame(svg,500,1.2,"optimizer step","masked MSE",5,4);line(svg,D.lapo.representation,f,"step","representation_loss",C.lapo,"with z",2);line(svg,D.lapo.representation,f,"step","zero_latent_loss",C.zero,"zero z",2);svg=document.querySelector("#lapo-policy");f=frame(svg,500,1,"optimizer step","cosine",5,4);line(svg,D.lapo.policy,f,"step","latent_policy_cosine",C.lapo,"predicted vs target latent",2);svg=document.querySelector("#lapo-adaptation");f=frame(svg,100,2.5,"optimizer step","action loss",5,5);D.lapo.adaptation.forEach((x,i)=>line(svg,x.rows,f,"step","action_loss",[C.mix_seen,C.lora_r32,C.lapo][i],`task ${x.task}`,2))}
function renderVideos(){const root=document.querySelector("#videos");for(const item of D.videos){const figure=document.createElement("figure"),caption=document.createElement("figcaption");if(item.video){const video=document.createElement("video");video.src=item.video;video.controls=true;video.preload="metadata";figure.append(video)}else{const missing=document.createElement("div");missing.textContent="video not recorded";missing.style.cssText="aspect-ratio:16/9;display:grid;place-items:center;background:#f7f8fa;color:#5f6b76";figure.append(missing)}caption.textContent=`${item.name}, ${item.success?"success":"failure"}`;figure.append(caption);root.append(figure)}}
const s=D.summary;document.querySelector("#readout").innerHTML=`<span>A100 eval episodes<b>${s.matrix.eval_episodes}</b></span><span>successful episodes<b>${s.matrix.successful_episodes}</b></span><span>training metric files<b>${D.adaptation.length+6}</b></span>`;renderCost();renderTraining();renderActionError();renderActions();renderLanguage();renderLapo();renderVideos();
</script>'''

def main() -> None:
    output = Path("report_page.html")
    output.write_text(TEMPLATE.replace("__DATA__", json.dumps(load_data(), ensure_ascii=False)))
    print(output)

if __name__ == "__main__":
    main()
