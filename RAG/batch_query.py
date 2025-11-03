import os
import json
import time
import sys
import shutil
import warnings
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dotenv import load_dotenv
from tqdm.auto import tqdm

from RAG.GraphRAG import query_graph_rag
from RAG.normalize import normalize_answer_text

# Load environment variables from .env at repo root
load_dotenv()

# Get timezone from .env with default fallback
timezone = os.getenv('TZ')
os.environ["TZ"] = timezone
time.tzset()

# Suppress noisy DeprecationWarnings from the Neo4j GraphRAG retriever to keep tqdm output clean
warnings.filterwarnings(
    "ignore",
    message="The default returned 'id' field in the search results will be removed.",
    category=DeprecationWarning,
    module="neo4j_graphrag.retrievers.vector",
)

def batch_query_graph_rag(
    input_question_file: str,
    max_workers: int = 1,
    top_k: int = None,
    neighbor_window: int = 1,
    show_progress: bool = False,
):
    """
    Process multiple queries in parallel using threading.
    
    Args:
        input_question_file: Path to JSON file containing questions
        max_workers: Number of worker threads (defaults to min(16, cpu_count * 2) if not provided)
        top_k: Optional top_k parameter for query_graph_rag (uses default if not provided)
        neighbor_window: Adjacent-chunk window per hit (0 disables; 1 includes i±1)
        show_progress: If True, display a tqdm progress bar as queries complete
    
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
    total_questions = len(questions)

    progress_bar = None
    progress_params = None
    if show_progress:
        bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        try:
            term_width = shutil.get_terminal_size().columns
        except Exception:
            term_width = 100
        progress_params = dict(
            total=total_questions,
            desc=f"Batch queries (workers={max_workers})",
            ncols=term_width,
            bar_format=bar_format,
            leave=False,
        )
    
    def process_query(item):
        idx, question = item
        try:
            if top_k is not None:
                answer = query_graph_rag(
                    question,
                    top_k=top_k,
                    neighbor_window=neighbor_window,
                    reference_seq_offset=idx,
                )
            else:
                answer = query_graph_rag(
                    question,
                    neighbor_window=neighbor_window,
                    reference_seq_offset=idx,
                )
            return {"index": idx, "question": question, "answer": answer}
        except Exception as e:
            return {"index": idx, "question": question, "answer": f"Error: {str(e)}"}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all queries
        future_to_question = {
            executor.submit(process_query, item): item
            for item in enumerate(questions)
        }

        # Collect results as they complete, maintaining order
        index_to_result = {}
        for future in as_completed(future_to_question):
            result = future.result()
            index_to_result[result["index"]] = result
            if progress_params is not None:
                if progress_bar is None:
                    progress_bar = tqdm(initial=1, **progress_params)
                else:
                    progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()

    # Reconstruct results in original order
    results = [
        {
            "question": index_to_result[i]["question"],
            "answer": index_to_result[i]["answer"],
        }
        for i in range(len(questions))
    ]
    
    # Normalize answers before saving
    for r in results:
        r["answer"] = normalize_answer_text(r.get("answer"), r.get("question"))

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
    
    def _log(message: str) -> None:
        if show_progress:
            tqdm.write(message, file=sys.stderr)
        else:
            print(message)

    _log(f"Processed {len(results)} queries with {max_workers} workers")
    _log(f"Results saved to: {output_file}")
    
    return results

