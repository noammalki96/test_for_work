#!/usr/bin/env python3
import argparse
import json
import os
import sys

# Dynamic path resolution to find local modules (call_llm.py, basic_prompts.py, etc.)
submission_directory = os.path.dirname(os.path.abspath(__file__))
repository_source_directory = os.path.abspath(os.path.join(submission_directory, "..", "..", "src"))
sys.path.insert(0, submission_directory)
if os.path.isdir(repository_source_directory):
    sys.path.insert(0, repository_source_directory)

from call_llm import APPROVED_MODELS
# Assuming you saved our optimized pipeline functions into a file named knowledge_extractor.py
from extractor import extract_final_graph

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

def main():
    argument_parser = argparse.ArgumentParser(description="Extract an optimized DAG Knowledge Graph from a LaTeX chapter.")
    argument_parser.add_argument("input", help="Path to the input .tex file")
    argument_parser.add_argument("-o", "--output", required=True, help="Path to the output .json file")
    argument_parser.add_argument("--model", default=DEFAULT_MODEL, choices=APPROVED_MODELS, help="LLM model to use")
    argument_parser.add_argument("--no-llm", action="store_true", help="Run deterministic parsing only (skip all LLM API calls)")
    parsed_arguments = argument_parser.parse_args()

    # Determine execution mode
    llm_is_enabled = not parsed_arguments.no_llm and os.environ.get("BDP_FORCE_NO_LLM", "0") != "1"
    
    print(f"Annotating: {parsed_arguments.input}")
    print(f"Using model: {parsed_arguments.model}" if llm_is_enabled else "No LLM used - Deterministic Extraction Only")

    # Read the raw LaTeX file
    try:
        with open(parsed_arguments.input, "r", encoding="utf-8", errors="replace") as f:
            latex_content = f.read()
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    # Fire the optimized Two-Pass Architecture Pipeline
    extracted_knowledge_graph = extract_final_graph(
        latex_text=latex_content,
        model_identifier=parsed_arguments.model,
        use_llm=llm_is_enabled
    )

    # Safe atomic write operation to prevent corrupted JSON outputs on crash
    output_directory = os.path.dirname(os.path.abspath(parsed_arguments.output))
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
        
    temporary_output_path = parsed_arguments.output + ".tmp"
    with open(temporary_output_path, "w", encoding="utf-8") as output_file:
        json.dump(extracted_knowledge_graph, output_file, ensure_ascii=False, indent=2)
        
    os.replace(temporary_output_path, parsed_arguments.output)
    
    # Calculate final graph metrics
    node_count = len(extracted_knowledge_graph.get('nodes', []))
    edge_count = len(extracted_knowledge_graph.get('edges', []))
    
    print(f"Extracted {node_count} nodes and {edge_count} edges")
    print(f"Output successfully written to: {parsed_arguments.output}")

if __name__ == "__main__":
    main()