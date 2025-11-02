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


def batch_query_graph_rag(
    input_question_file: str,
    max_workers: int = None,
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
    
    # Generate timestamp with timezone
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    # Save results to output file
    output_dir = project_root / "AB Testing" / "Output Answers"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"answers_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(results)} queries with {max_workers} workers")
    print(f"Results saved to: {output_file}")
    
    return results

