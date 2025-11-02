import os
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor


_F1_PROMPT = (
    "You are a strict grader. Compare the EXPECTED and ANSWER. "
    "Identify the essential factual points. Compute precision and recall over those points, "
    "then F1 = 2*precision*recall/(precision+recall). "
    "Return ONLY the F1 as a number between 0 and 1 with up to 3 decimals, no text.\n\n"
    "EXPECTED:\n{expected}\n\n"
    "ANSWER:\n{answer}"
)


def mark_answers(answer_key_path: str, max_workers: int = None) -> dict:
    # Load env and timezone
    load_dotenv()
    timezone = os.getenv("TZ", "America/New_York")
    os.environ["TZ"] = timezone
    time.tzset()

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
        paired_questions.append((question_text, expected, answer))

    # GPT-evaluated F1 per pair (optionally parallel)
    # Force model to gpt-5-mini regardless of env model selection
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    def _grade(expected: str, answer: str) -> float:
        prompt = _F1_PROMPT.format(expected=expected, answer=answer)
        resp = llm.invoke(prompt)
        return float(str(resp.content).strip())

    if max_workers is None:
        max_workers = min(16, (os.cpu_count() or 4) * 2)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_grade, e, a) for _, e, a in paired_questions]
        f1_scores = [f.result() for f in futures]
    avg_f1 = sum(f1_scores) / n if n else 0.0
    avg_f1_percent = avg_f1 * 100.0

    # Get config values (from answers file or environment)
    run_config = {
        "top_k": config.get("top_k") or (int(os.getenv("top_k")) if os.getenv("top_k") else None),
        "MMR_k": config.get("MMR_k") or (int(os.getenv("MMR_k")) if os.getenv("MMR_k") else None),
        "MMR_LAMBDA": config.get("MMR_LAMBDA") or (float(os.getenv("MMR_LAMBDA")) if os.getenv("MMR_LAMBDA") else None),
        "ALWAYS_KEEP_TOP": config.get("ALWAYS_KEEP_TOP") or (int(os.getenv("ALWAYS_KEEP_TOP")) if os.getenv("ALWAYS_KEEP_TOP") else None),
        "neighbor_window": config.get("neighbor_window"),
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
        "questions": [
            {
                "question": q,
                "expected": e,
                "answer": a,
                "f1": f1_scores[i],
            }
            for i, (q, e, a) in enumerate(paired_questions)
        ],
    }

    # Save to AB Testing/Test Results
    out_dir = project_root / "AB Testing" / "Test Results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"test_results_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Console summary
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print("\nConfiguration:")
    print(f"  top_k: {run_config['top_k']}")
    print(f"  MMR_k: {run_config['MMR_k']}")
    print(f"  MMR_LAMBDA: {run_config['MMR_LAMBDA']}")
    print(f"  ALWAYS_KEEP_TOP: {run_config['ALWAYS_KEEP_TOP']}")
    print(f"  neighbor_window: {run_config['neighbor_window']}")
    print("\nResults:")
    print(f"  Average F1: {avg_f1_percent:.1f}% ({avg_f1:.3f})")
    print("="*60 + "\n")

    return results

