import os
import json
import time
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv

from RAG.GraphRAG import query_graph_rag

# Load environment variables from .env at repo root
load_dotenv()

# Get timezone from .env with default fallback
timezone = os.getenv('TZ')
os.environ["TZ"] = timezone
time.tzset()


def _normalize_answer_text(answer: str, question: str) -> str:
    if not isinstance(answer, str):
        return answer
    s = answer.strip()
    q = (question or "").lower()
    # currency forms → K.D. N
    s = re.sub(r"\bKWD\b", "K.D.", s, flags=re.IGNORECASE)
    s = re.sub(r"\bK\.?D\.?\b", "K.D.", s, flags=re.IGNORECASE)
    s = re.sub(r"\bKD\b", "K.D.", s, flags=re.IGNORECASE)
    # normalize 'K.D. N' regardless of order
    s = re.sub(r"(?i)\b(?:KWD|KD|K\.?D\.?)\s*(\d+(?:\.\d+)?)\b", r"K.D. \1", s)
    s = re.sub(r"(?i)\b(\d+(?:\.\d+)?)\s*(?:KWD|KD|K\.?D\.?)\b", r"K.D. \1", s)
    # collapse double dots e.g., K.D.. -> K.D.
    s = s.replace("K.D..", "K.D.")
    # if numeric only and fee/bond context → prefix K.D.
    if re.fullmatch(r"\d+(?:\.\d+)?", s) and any(w in q for w in ["fee", "document", "bond", "guarantee", "price", "cost"]):
        s = f"K.D. {s}"

    # days normalization when question asks duration/validity
    if re.fullmatch(r"\d+", s) and any(w in q for w in ["valid", "validity", "period", "days", "how long"]):
        s = f"{s} days"
    s = re.sub(r"\b(\d+)\s*day\b", r"\1 days", s, flags=re.IGNORECASE)

    # date normalization to YYYY-MM-DD for common patterns
    def _to_iso_date(txt: str) -> str:
        t = txt.strip()
        # strip trailing parenthetical
        t = re.sub(r"\s*\([^)]*\)\s*$", "", t)
        # YYYY-MM-DD or YYYY/MM/DD or YYYY.MM.DD → YYYY-MM-DD
        m = re.fullmatch(r"(\d{4})[-/.](\d{2})[-/.](\d{2})", t)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        # DD-MM-YYYY or DD/MM/YYYY
        m = re.fullmatch(r"(\d{2})[-/](\d{2})[-/](\d{4})", t)
        if m:
            return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
        # Month DD, YYYY
        m = re.fullmatch(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s*(\d{4})", t, flags=re.IGNORECASE)
        if m:
            month_map = {m_name: f"{i:02d}" for i, m_name in enumerate(["January","February","March","April","May","June","July","August","September","October","November","December"], start=1)}
            mm = month_map.get(m.group(1).capitalize())
            dd = f"{int(m.group(2)):02d}"
            return f"{m.group(3)}-{mm}-{dd}"
        # DD Month YYYY
        m = re.fullmatch(r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})", t, flags=re.IGNORECASE)
        if m:
            month_map = {m_name: f"{i:02d}" for i, m_name in enumerate(["January","February","March","April","May","June","July","August","September","October","November","December"], start=1)}
            mm = month_map.get(m.group(2).capitalize())
            dd = f"{int(m.group(1)):02d}"
            return f"{m.group(3)}-{mm}-{dd}"
        return txt

    if any(w in q for w in ["date", "closing", "deadline"]):
        s = _to_iso_date(s)
    return s

def batch_query_graph_rag(
    input_question_file: str,
    max_workers: int = 1,
    top_k: int = None,
    neighbor_window: int = 1
):
    """
    Process multiple queries in parallel using threading.
    
    Args:
        input_question_file: Path to JSON file containing questions
        max_workers: Number of worker threads (defaults to min(16, cpu_count * 2) if not provided)
        top_k: Optional top_k parameter for query_graph_rag (uses default if not provided)
        neighbor_window: Adjacent-chunk window per hit (0 disables; 1 includes i±1)
    
    Returns:
        List of dictionaries with "question" and "answer" fields
    """
    # Get project root (parent of RAG directory)
    project_root = Path(__file__).resolve().parent.parent
    
    # Resolve input spec which may include '@file (start-end)' shorthand
    file_spec = (input_question_file or "").strip()
    range_start = None
    range_end = None
    shorthand_match = re.match(r"^\s*@\s*(?P<file>[^()]+?)\s*\(\s*(?P<start>\d+)\s*[-:]\s*(?P<end>\d+)\s*\)\s*$", file_spec)
    if shorthand_match:
        file_part = shorthand_match.group("file").strip()
        try:
            range_start = int(shorthand_match.group("start"))
            range_end = int(shorthand_match.group("end"))
        except Exception:
            range_start = None
            range_end = None
        input_path = Path(file_part)
    else:
        input_path = Path(file_spec)

    # Map relative paths: try project root first; if just a filename, also try 'AB Testing/'
    if not input_path.is_absolute():
        candidate = project_root / input_path
        if candidate.exists():
            input_path = candidate
        else:
            # If only a filename was provided, look under AB Testing
            if len(input_path.parts) == 1:
                candidate = project_root / "AB Testing" / input_path.name
                if candidate.exists():
                    input_path = candidate
                else:
                    input_path = project_root / input_path
            else:
                input_path = candidate
    
    with open(input_path, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    
    # Extract questions from JSON objects
    questions = []
    for item in questions_data:
        if isinstance(item, dict) and 'question' in item:
            questions.append(item['question'])
        elif isinstance(item, str):
            questions.append(item)
    
    # Apply optional 1-based inclusive slice if provided via shorthand
    if questions and range_start is not None and range_end is not None:
        # Normalize to 1-based bounds within [1, len]
        total = len(questions)
        start_1b = max(1, min(range_start, total))
        end_1b = max(1, min(range_end, total))
        if end_1b < start_1b:
            start_1b, end_1b = end_1b, start_1b
        # Convert to 0-based half-open slice
        start_0b = start_1b - 1
        end_exclusive = end_1b
        questions = questions[start_0b:end_exclusive]

    if not questions:
        raise ValueError("No questions found in input file")
    
    # Set default max_workers if not provided
    if max_workers is None:
        max_workers = min(16, (os.cpu_count() or 4) * 2)
    
    # Process queries in parallel
    results = []
    
    def process_query(question):
        try:
            if top_k is not None:
                answer = query_graph_rag(question, top_k=top_k, neighbor_window=neighbor_window)
            else:
                answer = query_graph_rag(question, neighbor_window=neighbor_window)
            return {"question": question, "answer": answer}
        except Exception as e:
            return {"question": question, "answer": f"Error: {str(e)}"}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all queries
        future_to_question = {
            executor.submit(process_query, question): question 
            for question in questions
        }
        
        # Collect results as they complete, maintaining order
        question_to_result = {}
        for future in as_completed(future_to_question):
            result = future.result()
            question_to_result[result["question"]] = result
        
        # Reconstruct results in original order
        results = [question_to_result[q] for q in questions]
    
    # Normalize answers before saving
    for r in results:
        r["answer"] = _normalize_answer_text(r.get("answer"), r.get("question"))

    # Generate timestamp with timezone
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    # Collect configuration values
    config = {
        "top_k": top_k if top_k is not None else int(os.getenv("top_k")),
        "MMR_k": int(os.getenv("MMR_k")),
        "MMR_LAMBDA": float(os.getenv("MMR_LAMBDA")),
        "ALWAYS_KEEP_TOP": int(os.getenv("ALWAYS_KEEP_TOP")),
        "neighbor_window": neighbor_window,
        # Card-first gate toggles (booleans if set, else None)
        "CARD_FIRST_ENABLED": (lambda v: (v.strip().lower() in ("1", "true", "yes", "on")) if v is not None else None)(os.getenv("CARD_FIRST_ENABLED")),
        "ALLOW_DIRECT_ANSWER": (lambda v: (v.strip().lower() in ("1", "true", "yes", "on")) if v is not None else None)(os.getenv("ALLOW_DIRECT_ANSWER")),
    }
    
    # Save results to output file
    output_dir = project_root / "AB Testing" / "Output Answers"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"answers_{timestamp}.json"
    
    output_data = {
        "config": config,
        "answers": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(results)} queries with {max_workers} workers")
    print(f"Results saved to: {output_file}")
    
    return results

