"""Execute a reproducible detector regression run."""
from __future__ import annotations
import argparse, json, os, statistics, time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import cv2
from hth.geometry.common import document_mask, resize_for_analysis, scale_bbox, valid_bbox
from .adapters.grabcut import detect as grabcut_detect
from .io import create_run_directory, environment_info, utc_now, write_json
from .metrics import bbox_iou, edge_errors
from .parameter_space import canonical_parameters, parameter_set_id, exhaustive_parameter_sets
from .reports import ranking_key, write_rankings, write_raw_results
from .strategies.cartesian import generate as cartesian_generate
from .strategies.binary_refine import search as binary_search
from .progress import ProgressReporter

DETECTORS={"grabcut":grabcut_detect}


def repository_root(path: Path) -> Path:
    candidate = path.resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / ".git").exists():
            return directory
    return Path.cwd()


def print_environment_banner(*, environment: dict[str, Any], detector: str, golden_set: Path, source_commit: str | None = None) -> None:
    """Print the execution environment once before a long regression begins."""
    print("Detector Regression Environment")
    print("=" * 31)
    print(f"Execution Target : {environment.get('execution_target') or '--'}")
    print(f"Runner           : {environment.get('runner_name') or '--'} ({environment.get('runner_environment') or '--'})")
    print(f"Machine          : {environment.get('machine_name') or '--'}")
    print(f"CPU              : {environment.get('cpu_model') or '--'}")
    print(f"Logical CPUs     : {environment.get('logical_cpu_count') or '--'}")
    memory = environment.get('memory_gib')
    print(f"Memory           : {memory:.2f} GiB" if isinstance(memory, (int, float)) else "Memory           : --")
    print(f"OS / Architecture: {environment.get('platform') or '--'} / {environment.get('runner_arch') or '--'}")
    print(f"Python           : {environment.get('python_version') or '--'}")
    print(f"OpenCV           : {environment.get('opencv_version') or '--'}")
    print(f"NumPy            : {environment.get('numpy_version') or '--'}")
    print(f"Pipeline Commit  : {environment.get('pipeline_commit') or '--'}")
    print(f"Source Commit    : {source_commit or '--'}")
    print(f"Golden Set       : {golden_set}")
    print(f"Detector         : {detector}")
    print()

def parse_args(argv: list[str] | None=None) -> argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument("--detector-config",type=Path,required=True)
    p.add_argument("--golden-set",type=Path,required=True)
    p.add_argument("--image-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True,help="Regression root; a detector/run-* directory is created below it.")
    p.add_argument("--strategy",choices=("exhaustive","binary-refine"),default="exhaustive")
    p.add_argument("--max-dimension",type=int,default=1800)
    p.add_argument("--limit",type=int,default=None)
    p.add_argument("--top",type=int,default=20)
    p.add_argument("--run-id",default=None)
    return p.parse_args(argv)

def find_image(root:Path, ordinal:int)->Path:
    for suffix in (".png",".jpg",".jpeg",".tif",".tiff",".webp"):
        for p in (root/"raw"/f"fs_{ordinal:04d}{suffix}",root/f"fs_{ordinal:04d}{suffix}"):
            if p.exists(): return p
    raise FileNotFoundError(f"No image found for Golden Set ordinal {ordinal}")

def load_pages(path:Path,root:Path,maximum:int)->list[dict[str,Any]]:
    data=json.loads(path.read_text(encoding="utf-8")); pages=[]
    for page in data.get("pages",[]):
        approved=page.get("physical_document_bbox")
        if not valid_bbox(approved): continue
        ordinal=int(page["global_ordinal"]); image_path=find_image(root,ordinal)
        original=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
        if original is None: raise RuntimeError(f"Could not read image: {image_path}")
        oh,ow=original.shape[:2]; image,scale=resize_for_analysis(original,maximum); mask,diag=document_mask(image)
        pages.append({"global_ordinal":ordinal,"label":page.get("label",f"page_{ordinal}"),"layout_type":page.get("layout_type","other"),"approved_bbox":[int(v) for v in approved],"image_path":str(image_path),"image":image,"mask":mask,"mask_diagnostics":diag,"scale":scale,"original_width":ow,"original_height":oh})
    if not pages: raise ValueError("Golden Set contains no approved pages with valid bounding boxes")
    return pages

def evaluate_set(detector:Any, parameters:dict[str,Any], pages:list[dict[str,Any]])->dict[str,Any]:
    page_results=[]; started=time.perf_counter()
    for page in pages:
        page_started=time.perf_counter()
        try:
            candidate=detector(image_bgr=page["image"],mask=page["mask"],parameters=parameters)
            elapsed=(time.perf_counter()-page_started)*1000
            if candidate.bbox is None:
                page_results.append({"global_ordinal":page["global_ordinal"],"label":page["label"],"layout_type":page["layout_type"],"status":candidate.status if candidate.status!="ok" else "no_candidate","iou":0.0,"edge_error_mean_px":None,"edge_error_maximum_px":None,"elapsed_ms":round(elapsed,3),"candidate":asdict(candidate)})
                continue
            predicted=scale_bbox(candidate.bbox,1.0/page["scale"],page["original_width"],page["original_height"])
            approved=page["approved_bbox"]; errors=edge_errors(predicted,approved)
            page_results.append({"global_ordinal":page["global_ordinal"],"label":page["label"],"layout_type":page["layout_type"],"status":"ok","approved_bbox":approved,"predicted_bbox":predicted,"iou":round(bbox_iou(predicted,approved),8),"edge_errors":errors,"edge_error_mean_px":round(float(errors["mean"]),3),"edge_error_maximum_px":int(errors["maximum"]),"elapsed_ms":round(elapsed,3),"candidate":asdict(candidate)})
        except Exception as exc:
            elapsed=(time.perf_counter()-page_started)*1000
            page_results.append({"global_ordinal":page["global_ordinal"],"label":page["label"],"layout_type":page["layout_type"],"status":"error","iou":0.0,"edge_error_mean_px":None,"edge_error_maximum_px":None,"elapsed_ms":round(elapsed,3),"error":{"type":type(exc).__name__,"message":str(exc)}})
    successful=[r for r in page_results if r["status"]=="ok"]; ious=[float(r["iou"]) for r in page_results]; edges=[float(r["edge_error_mean_px"]) for r in successful]; elapsed=[float(r["elapsed_ms"]) for r in page_results]
    return {"parameter_set_id":parameter_set_id(parameters),"parameters":parameters,"summary":{"page_count":len(page_results),"success_count":len(successful),"failure_count":len(page_results)-len(successful),"mean_iou":round(sum(ious)/len(ious),8),"minimum_iou":round(min(ious),8),"stddev_iou":round(statistics.pstdev(ious),8),"mean_edge_error_px":round(sum(edges)/len(edges),3) if edges else None,"elapsed_ms_total":round(sum(elapsed),3),"wall_ms":round((time.perf_counter()-started)*1000,3)},"pages":page_results}

def run(args:argparse.Namespace)->Path:
    config=json.loads(args.detector_config.read_text(encoding="utf-8")); name=str(config["detector"])
    if name not in DETECTORS: raise SystemExit(f"Unsupported detector: {name}")
    run_id,run_dir=create_run_directory(args.output,name,args.run_id); started=utc_now(); wall=time.perf_counter()
    environment=environment_info(repository_root(args.detector_config))
    source_commit=os.environ.get("HTH_SOURCE_COMMIT")
    write_json(run_dir/"parameters.json",{"schema_version":"0.2","detector":name,"strategy":args.strategy,"detector_config":str(args.detector_config),"golden_set":str(args.golden_set),"image_root":str(args.image_root),"max_dimension":args.max_dimension,"limit":args.limit,"configuration":config})
    manifest={"schema_version":"0.1","run_id":run_id,"detector":name,"strategy":args.strategy,"status":"running","started_at_utc":started,"outputs":[]}
    write_json(run_dir/"manifest.json",manifest)
    try:
        pages=load_pages(args.golden_set,args.image_root,args.max_dimension); detector=DETECTORS[name]
        if not callable(detector):
            raise TypeError(
                f"Detector registry entry {name!r} is not callable: "
                f"{type(detector).__name__}"
            )
        profiles={canonical_parameters(p):n for n,p in config.get("profiles",{}).items()}
        baseline_parameters=config.get("profiles",{}).get("baseline")
        if not isinstance(baseline_parameters,dict):
            raise ValueError("Detector configuration must define profiles.baseline")
        baseline_key=canonical_parameters(baseline_parameters)

        exhaustive_candidates=[
            parameters for parameters in cartesian_generate(config)
            if canonical_parameters(parameters) != baseline_key
        ]
        if args.limit is not None:
            exhaustive_candidates=exhaustive_candidates[:args.limit]
        estimated_total=(
            len(exhaustive_candidates)
            if args.strategy=="exhaustive"
            else max(0,len(exhaustive_parameter_sets(config))-1)
        )

        print_environment_banner(environment=environment,detector=name,golden_set=args.golden_set,source_commit=source_commit)
        progress=ProgressReporter(total=estimated_total,interval_seconds=60.0)
        progress.start()

        progress.begin_evaluation("baseline")
        baseline_result=evaluate_set(detector,dict(baseline_parameters),pages)
        progress.observe_baseline(baseline_result)

        def evaluate(parameters:dict[str,Any])->dict[str,Any]:
            canonical=canonical_parameters(parameters)
            if canonical == baseline_key:
                return baseline_result
            profile=profiles.get(canonical)
            profile_name=profile or f"ps:{parameter_set_id(parameters)[:8]}"
            progress.begin_evaluation(profile_name)
            result=evaluate_set(detector,parameters,pages)
            progress.observe(result,profile)
            return result
        if args.strategy=="exhaustive":
            results=[baseline_result,*[evaluate(p) for p in exhaustive_candidates]]
        else:
            results=binary_search(config,evaluate,ranking_key)
            if not any(canonical_parameters(r["parameters"]) == baseline_key for r in results):
                results.insert(0,baseline_result)
        progress_snapshot=progress.finish()
        for r in results: r["profile"]=profiles.get(canonical_parameters(r["parameters"])); r["run_id"]=run_id
        ranked=sorted(results,key=ranking_key)
        for rank,r in enumerate(ranked,1): r["rank"]=rank
        baseline=next((r for r in ranked if r.get("profile")=="baseline"),None)
        raw=run_dir/"raw"/"results.csv"; rankings=run_dir/"reports"/"rankings.csv"; top=run_dir/"reports"/"top20.csv"
        write_raw_results(raw,ranked); write_rankings(rankings,ranked); write_rankings(top,ranked[:max(0,args.top)])
        summary={"schema_version":"0.5","run_id":run_id,"detector":name,"strategy":args.strategy,"page_ordinals":[p["global_ordinal"] for p in pages],"parameter_set_count":len(ranked),"page_evaluation_count":len(ranked)*len(pages),"winner":ranked[0],"baseline":baseline,"runner":environment,"source_commit":source_commit,"progress":{"estimated_parameter_sets":progress_snapshot.total,"completed_parameter_sets":progress_snapshot.completed,"average_eval_rate":progress_snapshot.eval_rate,"failures":progress_snapshot.failures,"best_mean_iou":progress_snapshot.best_mean_iou,"best_worst_page_iou":progress_snapshot.minimum_page_iou,"last_improvement_elapsed_seconds":progress_snapshot.last_improvement_elapsed_seconds,"time_since_last_improvement_seconds":progress_snapshot.last_improvement_seconds}}
        write_json(run_dir/"reports"/"summary.json",summary)
        finished=utc_now(); info={"schema_version":"0.2","run_id":run_id,"detector":name,"strategy":args.strategy,"status":"complete","started_at_utc":started,"finished_at_utc":finished,"elapsed_seconds":round(time.perf_counter()-wall,3),"golden_set":str(args.golden_set),"detector_config":str(args.detector_config),"source_commit":source_commit,**environment}
        write_json(run_dir/"RUN-INFO.json",info)
        manifest.update({"status":"complete","finished_at_utc":finished,"outputs":["RUN-INFO.json","parameters.json","raw/results.csv","reports/summary.json","reports/rankings.csv","reports/top20.csv"]}); write_json(run_dir/"manifest.json",manifest)
        # Convenience report at detector root, refreshed on every completed run.
        write_rankings(run_dir.parent/f"{name}-regression-results.csv",ranked)
        winner_summary=ranked[0]["summary"]
        baseline_summary=baseline["summary"] if baseline else None
        winner_profile=ranked[0].get("profile") or "ps:"+ranked[0]["parameter_set_id"][:8]
        progress.announce("Regression complete", winner_profile)
        elapsed_seconds=time.perf_counter()-wall
        print("Regression Summary")
        print(f"Run: {run_id}")
        print(f"Elapsed: {elapsed_seconds:.1f}s")
        print(f"Average Eval Rate: {(len(ranked)/elapsed_seconds if elapsed_seconds else 0.0):.4f}/s")
        print(f"Parameter sets evaluated: {len(ranked)}")
        print(f"Page evaluations: {len(ranked)*len(pages)}")
        print(f"Successful parameter sets: {sum(1 for r in ranked if int(r['summary'].get('failure_count',0) or 0)==0)}")
        print(f"Failed page evaluations: {progress_snapshot.failures}")
        print(f"Winner: {winner_profile}")
        print(f"Average Page IoU: {winner_summary['mean_iou']:.4f}")
        print(f"Minimum Page IoU: {winner_summary['minimum_iou']:.4f}")
        print(f"Std Dev: {winner_summary['stddev_iou']:.4f}")
        if progress_snapshot.last_improvement_elapsed_seconds is not None:
            print(f"Last improvement at: {progress_snapshot.last_improvement_elapsed_seconds:.1f}s")
        if baseline_summary:
            print(f"Baseline Average Page IoU: {baseline_summary['mean_iou']:.4f}")
            print(f"Average Page IoU improvement: {winner_summary['mean_iou']-baseline_summary['mean_iou']:+.4f}")
            print(f"Minimum Page IoU improvement: {winner_summary['minimum_iou']-baseline_summary['minimum_iou']:+.4f}")
        print(json.dumps({"run_id":run_id,"run_directory":str(run_dir),"winner":ranked[0],"baseline":baseline},indent=2))
        return run_dir
    except Exception as exc:
        manifest.update({"status":"failed","finished_at_utc":utc_now(),"error":{"type":type(exc).__name__,"message":str(exc)}}); write_json(run_dir/"manifest.json",manifest)
        raise

def main(argv:list[str]|None=None)->int:
    run(parse_args(argv)); return 0
