import os
import sys
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

# TODO Add comment parameter to save to test result file

_F1_PROMPT = (
    "You are a strict grader. Compare the EXPECTED and ANSWER. "
    "Identify the essential factual points. Compute precision and recall over those points, "
    "then F1 = 2*precision*recall/(precision+recall). "
    "Return ONLY the F1 as a number between 0 and 1 with up to 3 decimals, no text.\n\n"
    "EXPECTED:\n{expected}\n\n"
    "ANSWER:\n{answer}"
)


def _env_bool(name: str, default=None):
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _format_scope(scope) -> str:
    if not scope:
        return "-"
    if isinstance(scope, (list, tuple)):
        return ",".join(str(s) for s in scope)
    return str(scope)


def _summarize_events(events) -> list[str]:
    logs: list[str] = []
    for ev in events or []:
        kind = ev.get("event")
        if kind == "gate":
            decision = ev.get("decision") or "pass"
            scope = _format_scope(ev.get("normalized_scope") or ev.get("scope_candidates"))
            anchors = len(ev.get("anchors") or [])
            logs.append(f"gate={decision} scope={scope} anchors={anchors}")
        elif kind == "retrieval":
            label = ev.get("label") or "primary"
            scope = _format_scope(ev.get("scope"))
            pool = ev.get("pool_top_k")
            hits = ev.get("hybrid_retrieved")
            final = ev.get("final")
            mmr = ev.get("mmr_applied")
            logs.append(
                f"retrieval[{label}] scope={scope} pool={pool} hits={hits} final={final} mmr={mmr}"
            )
        elif kind == "gate_fail_open":
            scope = _format_scope(ev.get("scope"))
            logs.append(f"fail_open scope={scope} reason={ev.get('reason')}")
        elif kind == "answer":
            source = ev.get("source")
            fallback = ev.get("fallback")
            retrieved = ev.get("retrieved")
            logs.append(f"answer source={source} fallback={fallback} retrieved={retrieved}")
    return logs


def mark_answers(
    answer_key_path: str,
    max_workers: int = None,
    show_progress: bool = False,
) -> dict:
    # Load env and timezone
    load_dotenv()
    timezone = os.getenv("TZ", "America/New_York")
    os.environ["TZ"] = timezone
    time.tzset()

    save_event_trace = _env_bool("SAVE_EVENT_TRACE", False)

    # Resolve project root
    project_root = Path(__file__).resolve().parents[1]

    # Locate latest answers file under AB Testing/Output Answers
    answers_dir = project_root / "AB Testing" / "Output Answers"
    answer_files = sorted(
        [p for p in answers_dir.glob("answers_*.json") if p.is_file()]
    )
    latest_answers_path = answer_files[-1]

    # Resolve answer key path (absolute or relative)
    key_path = Path(answer_key_path)
    if not key_path.is_absolute():
        candidate = project_root / key_path
        if candidate.exists():
            key_path = candidate
        else:
            # If just a filename, also try AB Testing/
            if len(key_path.parts) == 1:
                candidate = project_root / "AB Testing" / key_path.name
                key_path = candidate
            else:
                key_path = project_root / key_path

    # Read JSON files
    with open(key_path, "r", encoding="utf-8") as f:
        key_items = json.load(f)
    with open(latest_answers_path, "r", encoding="utf-8") as f:
        answer_data = json.load(f)
    
    # Handle both new format (with config) and old format (without config)
    if isinstance(answer_data, dict) and "answers" in answer_data:
        config = answer_data.get("config", {})
        answer_items = answer_data["answers"]
    else:
        # Old format - just an array of answers
        answer_items = answer_data
        config = {}

    # Pair by index
    n = min(len(key_items), len(answer_items))
    paired_questions = []
    for i in range(n):
        key_obj = key_items[i]
        ans_obj = answer_items[i]
        question_text = ans_obj.get("question") or key_obj.get("question") or ""
        expected = (key_obj.get("answer") or "").strip()
        answer = (ans_obj.get("answer") or "").strip()
        events = ans_obj.get("events", []) if save_event_trace else []
        paired_questions.append(
            {
                "question": question_text,
                "expected": expected,
                "answer": answer,
                "events": events,
            }
        )

    # GPT-evaluated F1 per pair (optionally parallel)
    # Force model to gpt-5-mini regardless of env model selection
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    def _grade(expected: str, answer: str) -> float:
        prompt = _F1_PROMPT.format(expected=expected, answer=answer)
        resp = llm.invoke(prompt)
        return float(str(resp.content).strip())

    if max_workers is None:
        max_workers = min(16, (os.cpu_count() or 4) * 2)

    f1_scores = [0.0] * n

    progress_bar = None
    if show_progress and n:
        bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        try:
            term_width = shutil.get_terminal_size().columns
        except Exception:
            term_width = 100
        progress_bar = tqdm(
            total=n,
            desc=f"Marking answers (workers={max_workers})",
            ncols=term_width,
            bar_format=bar_format,
            leave=False,
            disable=(n == 0),
        )

    if n:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_idx = {
                ex.submit(_grade, item["expected"], item["answer"]): idx
                for idx, item in enumerate(paired_questions)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                f1_scores[idx] = future.result()
                if progress_bar is not None:
                    progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()
    avg_f1 = sum(f1_scores) / n if n else 0.0
    avg_f1_percent = avg_f1 * 100.0

    # Get config values (from answers file or environment)
    run_config = {
        "top_k": config.get("top_k") or (int(os.getenv("top_k")) if os.getenv("top_k") else None),
        "MMR_k": config.get("MMR_k") or (int(os.getenv("MMR_k")) if os.getenv("MMR_k") else None),
        "MMR_LAMBDA": config.get("MMR_LAMBDA") or (float(os.getenv("MMR_LAMBDA")) if os.getenv("MMR_LAMBDA") else None),
        "ALWAYS_KEEP_TOP": config.get("ALWAYS_KEEP_TOP") or (int(os.getenv("ALWAYS_KEEP_TOP")) if os.getenv("ALWAYS_KEEP_TOP") else None),
        "neighbor_window": config.get("neighbor_window"),
        # Card-first gate toggles (prefer value from answers file config; else from env; else None)
        "CARD_FIRST_ENABLED": (
            config.get("CARD_FIRST_ENABLED")
            if "CARD_FIRST_ENABLED" in config
            else (
                (os.getenv("CARD_FIRST_ENABLED").strip().lower() in ("1", "true", "yes", "on"))
                if os.getenv("CARD_FIRST_ENABLED")
                else None
            )
        ),
        "ALLOW_DIRECT_ANSWER": (
            config.get("ALLOW_DIRECT_ANSWER")
            if "ALLOW_DIRECT_ANSWER" in config
            else (
                (os.getenv("ALLOW_DIRECT_ANSWER").strip().lower() in ("1", "true", "yes", "on"))
                if os.getenv("ALLOW_DIRECT_ANSWER")
                else None
            )
        ),
    }
    
    # Extract timestamp from answers filename (e.g., "answers_20251101_153304.json" -> "20251101_153304")
    answers_stem = latest_answers_path.stem
    timestamp = answers_stem.replace("answers_", "")
    
    # Results object
    now = datetime.now()
    results = {
        "saved_at": now.strftime("%Y-%m-%dT%H:%M:%S"),
        "expected_file": str(key_path),
        "answers_file": str(latest_answers_path),
        "config": run_config,
        "overall": {
            "evaluated": float(n),
            "average_f1": avg_f1,
            "average_f1_percent": avg_f1_percent,
        },
        "questions": [],
    }

    for i, item in enumerate(paired_questions):
        question_entry = {
            "question": item["question"],
            "expected": item["expected"],
            "answer": item["answer"],
            "f1": f1_scores[i],
        }
        if save_event_trace:
            events = item.get("events", [])
            question_entry["events"] = events
            question_entry["logs"] = _summarize_events(events)
        results["questions"].append(question_entry)

    # Save to AB Testing/Test Results
    out_dir = project_root / "AB Testing" / "Test Results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"test_results_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Console summary
    def _log(message: str = "") -> None:
        if show_progress:
            tqdm.write(message)
        else:
            print(message)

    _log()
    _log("=" * 60)
    _log("TEST RESULTS")
    _log("=" * 60)
    _log()
    _log("Configuration:")
    _log(f"  top_k: {run_config['top_k']}")
    _log(f"  MMR_k: {run_config['MMR_k']}")
    _log(f"  MMR_LAMBDA: {run_config['MMR_LAMBDA']}")
    _log(f"  ALWAYS_KEEP_TOP: {run_config['ALWAYS_KEEP_TOP']}")
    _log(f"  neighbor_window: {run_config['neighbor_window']}")
    _log()
    _log("Results:")
    _log(f"  Average F1: {avg_f1_percent:.1f}% ({avg_f1:.3f})")
    _log()
    _log("Card-first gate:")
    _log(f"  CARD_FIRST_ENABLED: {run_config['CARD_FIRST_ENABLED']}")
    _log(f"  ALLOW_DIRECT_ANSWER: {run_config['ALLOW_DIRECT_ANSWER']}")
    _log("=" * 60)
    _log()

    return results

