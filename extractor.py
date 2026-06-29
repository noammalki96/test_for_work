#!/usr/bin/env python3
import argparse
import json
import re
import sys
import os
import concurrent.futures
from collections import Counter, defaultdict

# Go UP one directory level to find the root project folder
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, projec_root)

# Now Python can see these files in the root folder!
from call_llm import call_llm, APPROVED_MODELS
from basic_prompts import NODE_PROMPT_TEMPLATE, EDGE_PROMPT_TEMPLATE

# User-provided mapping for environment normalization
ENV_MAP = {
    "definition": "definition",
    "defn": "definition",
    "assumption": "definition",
    "assumptions": "definition",
    "hyp": "definition",
    "condition": "definition",
    "conditions": "definition",
    "notation": "definition",
    "problem": "definition",
    "setup": "definition",
    "axiom": "definition",
    "theorem": "theorem",
    "thm": "theorem",
    "proposition": "theorem",
    "prop": "theorem",
    "claim": "theorem",
    "conjecture": "theorem",
    "result": "theorem",
    "fact": "theorem",
    "observation": "theorem",
    "obs": "theorem",
    "lemma": "lemma",
    "lem": "lemma",
    "corollary": "corollary",
    "cor": "corollary",
    "proof": "proof",
    "algorithm": "algorithm",
    "procedure": "theorem",
    "pseudocode": "algorithm"
}

# Environments to strictly exclude from extraction
EXCLUDE_ENVS = {
    "equation", "equation*", "align", "align*", "gather", "gather*",
    "multline", "multline*", "figure", "figure*", "table", "table*",
    "thebibliography", "bibliography", "tikzpicture", "example", "exercise"
}

TARGET_TYPES = {"theorem", "lemma", "corollary", "proof", "definition", "algorithm"}

# Regex for implicit definitions in plain text
DEF_KEYWORDS = re.compile(
    r'\b(called|defined|define|defines|let|denote|denoted|denotes)\b',
    re.IGNORECASE
)


def extract_fragments(latex_text):
    """
    Parses LaTeX text using a stack to handle nested environments.
    Extracts targeted blocks and scans plain text for implicit definitions.
    """
    # 1. Strip comments to prevent false positive parsing
    latex_text = re.sub(r'(?<!\\)%.*$', '', latex_text, flags=re.MULTILINE)

    fragments = []
    plain_text_chunks = []

    # Regex to catch \begin{...} and \end{...}
    env_regex = re.compile(r'\\(begin|end)\s*\{([^}]+)\}')

    stack = []
    current_pos = 0

    # 2. Extract explicit environments via stack parsing
    for match in env_regex.finditer(latex_text):
        tag_type = match.group(1)  # 'begin' or 'end'
        env_name = match.group(2).strip()

        if tag_type == 'begin':
            if not stack:
                # Transitioning from plain text into a top-level environment
                plain_text_chunks.append(latex_text[current_pos:match.start()])
                current_pos = match.start()
            stack.append(env_name)

        else:  # 'end'
            if stack and stack[-1] == env_name:
                stack.pop()
                if not stack:
                    # Transitioning out of a top-level environment
                    env_start = current_pos
                    env_end = match.end()
                    env_content = latex_text[env_start:env_end]

                    mapped_type = ENV_MAP.get(env_name)

                    # If it's a target, save it
                    if mapped_type in TARGET_TYPES:
                        fragments.append({
                            "type": mapped_type,
                            "original_env": env_name,
                            "content": env_content.strip()
                        })
                    # If it's NOT a target and NOT explicitly excluded,
                    # we treat its inner content as plain text to be scanned for definitions.
                    elif env_name not in EXCLUDE_ENVS:
                        plain_text_chunks.append(env_content)

                    current_pos = env_end

    # Append any remaining plain text at the end of the document
    if current_pos < len(latex_text) and not stack:
        plain_text_chunks.append(latex_text[current_pos:])

    # 3. Process plain text chunks for implicit definitions
    for chunk in plain_text_chunks:
        # Split by double newline to evaluate paragraph by paragraph
        paragraphs = re.split(r'\n\s*\n', chunk)
        for para in paragraphs:
            para_clean = para.strip()
            if not para_clean:
                continue

            if DEF_KEYWORDS.search(para_clean):
                fragments.append({
                    "type": "definition",
                    "original_env": "implicit",
                    "content": para_clean
                })

    return fragments

def extract_nodes_from_batches(fragment_batches, model_identifier):
    """
    Map Phase: Concurrently sends fragment batches to the LLM.
    """
    all_extracted_nodes = []

    def process_batch(batch_text):
        prompt = NODE_PROMPT_TEMPLATE.replace("{fragments_batch}", batch_text)
        # Using the standard wrapper we built earlier
        raw_response = call_LLM_model(prompt, model_identifier, max_tokens=4096)
        # Using the Llama-hardened parser we built earlier
        parsed_json = extract_json_from_response(raw_response)

        # Ensure it's a list
        if isinstance(parsed_json, dict) and "nodes" in parsed_json:
            return parsed_json["nodes"]
        elif isinstance(parsed_json, list):
            return parsed_json
        return []

    # Concurrency: Fire all LLM calls simultaneously
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Map the batches to the processing function
        results = list(executor.map(process_batch, fragment_batches))

    # Flatten the list of lists
    for result_list in results:
        all_extracted_nodes.extend(result_list)

    return all_extracted_nodes


def deduplicate_nodes(nodes):
    """
    Reduce Phase: Removes exact duplicates and merges nodes with the same ID.
    """
    unique_nodes = {}
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue

        # If we haven't seen this ID, add it
        if node_id not in unique_nodes:
            unique_nodes[node_id] = node
        else:
            # If we have, prioritize the one with more fields populated
            existing_len = len(str(unique_nodes[node_id]))
            new_len = len(str(node))
            if new_len > existing_len:
                unique_nodes[node_id] = node

    return list(unique_nodes.values())

def extract_json_from_response(response):
    """Extract JSON object from LLM response (handles markdown blocks, thinking tags)."""
    # Strip <think>...</think> tags (Qwen3 reasoning), including unclosed tags
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    response = re.sub(r'<think>.*', '', response, flags=re.DOTALL).strip()

    # Try to find JSON in markdown code block
    match = re.search(r'```(?:json)?\s*\n?(.*?)```', response, re.DOTALL | re.IGNORECASE)
    if match:
        response = match.group(1)

    # Try to find JSON object directly
    match = re.search(r'(\{.*\}|\[.*\])', response, re.DOTALL)
    if match:
        json_str = match.group(1)

        # Attempt strict parsing first
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # 4. Llama-3.1 Fallback: Strip trailing commas and try again
            json_str_cleaned = re.sub(r',\s*([\]}])', r'\1', json_str)
            try:
                return json.loads(json_str_cleaned)
            except json.JSONDecodeError:
                pass

    # 5. Last resort: try parsing the raw stripped response
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None


def call_LLM_model(prompt, model_identifier, max_tokens=4096, temperature=0.0):
    """
    Standardized wrapper to call the LLM for both Node and Edge extraction phases.
    Enforces a strict JSON-only output format.
    """
    system_instruction = (
        "You are a precise mathematical knowledge extraction system. "
        "Output ONLY valid JSON. No explanations, no reasoning, no thinking."
    )

    try:
        # Calls your underlying base API function
        response = call_llm(
            prompt=prompt,
            model=model_identifier,
            system_prompt=system_instruction,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response

    except Exception as api_error:
        print(f"  [!] LLM API Call failed: {api_error}")
        # Returning an empty JSON block ensures downstream parsers don't crash
        # on 'NoneType' errors if the API drops the connection.
        return "{}"


def generate_identifier_indexes(node_list):
    """
    Builds O(1) lookup dictionaries for fast edge resolution.
    """
    index_by_key = {node["key"]: node for node in node_list if node.get("key")}
    index_by_label = {
        node["label"]: node
        for node in node_list
        if node.get("label")
    }
    index_by_final_id = {node["final_id"]: node for node in node_list if node.get("final_id")}

    return index_by_key, index_by_label, index_by_final_id


def register_valid_dependency(edge_collection, source_id, target_id, allowed_ids):
    """
    Safely inserts a directed edge into the set after validating node existence and preventing self-loops.
    """
    if not source_id or not target_id or source_id == target_id:
        return
    if source_id not in allowed_ids or target_id not in allowed_ids:
        return

    edge_collection.add((source_id, target_id))


def _is_node_name_cited(origin_node, destination_text):
    """
    Detects if the semantic name of an earlier node is explicitly mentioned in the text of a later node.
    """
    origin_name = (origin_node.get("name") or "").strip().lower()
    if not origin_name or origin_name in {"proof", "theorem", "lemma", "definition", "corollary", "algorithm"}:
        return False

    import re
    name_tokens = set(re.findall(r"[a-z][a-z0-9]*", origin_name))

    # Require at least 2 tokens unless it's a known highly-specific keyword
    if len(name_tokens) < 2 and not re.search(r"\b(?:vc|svd|shap|shapley|sauer|gibbs|lloyd|ward|perceptron)\b",
                                              origin_name):
        return False

    destination_lower = destination_text.lower()

    # 1. Check for exact substring match
    if origin_name in destination_lower:
        return True

    # 2. Check for token subset match (for longer names that might be slightly modified)
    destination_tokens = set(re.findall(r"[a-z][a-z0-9]*", destination_lower))
    return len(name_tokens) >= 3 and name_tokens.issubset(destination_tokens)


def assemble_structural_dependencies(graph_nodes):
    """
    Orchestrates the discovery of dependencies using LaTeX labels, textual citations,
    and document adjacency.
    """
    import re

    # Use the identifier indexer function we renamed previously
    key_index, label_index, final_id_index = generate_identifier_indexes(graph_nodes)
    valid_id_set = set(final_id_index)

    # Index nodes by structural ID (e.g., 'T5.1')
    structural_id_map = {
        node.get("structural_id"): node
        for node in graph_nodes
        if node.get("structural_id")
    }

    type_to_prefix = {
        "theorem": "T", "lemma": "L", "corollary": "C", "proposition": "T",
        "claim": "T", "conjecture": "T", "definition": "D", "algorithm": "A",
    }

    found_edges = set()
    latest_seen_by_type = {}
    last_referenced_node = None

    for i, node in enumerate(graph_nodes):
        current_node_id = node["final_id"]
        node_text = node.get("content", "") or node.get("statement", "")
        text_lower = node_text.lower()

        # 1. Handle Proof-to-Result links
        if node["type"] == "proof" and node.get("proves_final_id"):
            register_valid_dependency(found_edges, current_node_id, node["proves_final_id"], valid_id_set)

        # 2. Extract LaTeX label references
        latex_refs = re.findall(r"\\(?:ref|eqref|autoref|pageref|cref|Cref)\*?\{([^{}]+)\}", node_text)
        for ref_label in latex_refs[:8]:
            dependency_node = label_index.get(ref_label) or key_index.get(ref_label)
            if dependency_node:
                register_valid_dependency(found_edges, dependency_node["final_id"], current_node_id, valid_id_set)

        # 3. Detect textual structural citations (e.g., "Lemma 4.2")
        text_citation_count = 0
        for ref_word, ref_num in re.findall(
                r"\b(Theorem|Lemma|Corollary|Proposition|Claim|Conjecture|Definition|Algorithm)s?\s*~?\s*"
                r"([0-9]+(?:\.[0-9]+){0,3})\b", node_text, flags=re.IGNORECASE):

            struct_id = type_to_prefix[ref_word.lower()] + ref_num
            referenced_node = structural_id_map.get(struct_id)

            if referenced_node and referenced_node["final_id"] != current_node_id:
                if referenced_node.get("source_order", 0) <= node.get("source_order", 0) or node["type"] == "proof":
                    register_valid_dependency(found_edges, referenced_node["final_id"], current_node_id, valid_id_set)
                    text_citation_count += 1
            if text_citation_count >= 4:
                break

        # 4. Resolve relative references (e.g., "above theorem")
        for ref_type in ("lemma", "theorem", "corollary", "definition", "algorithm"):
            if re.search(r"\b(?:previous|preceding|above|last)\s+" + ref_type + r"\b", text_lower):
                referenced_node = latest_seen_by_type.get(ref_type)
                if referenced_node:
                    register_valid_dependency(found_edges, referenced_node["final_id"], current_node_id, valid_id_set)

        # 5. Detect named references (e.g., "by Perceptron lemma")
        if re.search(r"\b(?:by|using|uses|from|apply|applying|follows\s+from|according\s+to)\b", text_lower):
            named_citation_count = 0
            for prior_node in reversed(graph_nodes[:i]):
                if prior_node["final_id"] == current_node_id:
                    continue
                if prior_node["type"] not in {"definition", "theorem", "lemma", "corollary", "algorithm"}:
                    continue
                if _is_node_name_cited(prior_node, node_text):
                    register_valid_dependency(found_edges, prior_node["final_id"], current_node_id, valid_id_set)
                    named_citation_count += 1
                if named_citation_count >= 3:
                    break

        # Maintenance of tracking state
        if node["type"] in {"theorem", "lemma", "corollary"}:
            last_referenced_node = node
        if node["type"] in latest_seen_by_type or node["type"] in {"definition", "theorem", "lemma", "corollary",
                                                                   "algorithm"}:
            latest_seen_by_type[node["type"]] = node

    return found_edges


def assemble_graph_connections(node_collection):
    """
    Orchestrates the deterministic edge generation by scanning nodes and
    applying the cross-reference heuristics.
    """
    node_by_key, node_by_label, node_by_final_id = generate_identifier_indexes(node_collection)
    valid_ids = set(node_by_final_id)

    node_by_struct_id = {
        node.get("structural_id"): node
        for node in node_collection
        if node.get("structural_id")
    }

    # Mapping for textual reference resolution
    type_to_char = {
        "theorem": "T", "lemma": "L", "corollary": "C", "proposition": "T",
        "claim": "T", "conjecture": "T", "definition": "D", "algorithm": "A",
    }

    connection_set = set()
    history_by_category = {}
    last_significant_node = None

    for idx, current_node in enumerate(node_collection):
        target_id = current_node["final_id"]
        raw_text = current_node.get("content", "") or current_node.get("statement", "")
        text_lower = raw_text.lower()

        # 1. Map Proof -> Node links
        if current_node["type"] == "proof" and current_node.get("proves_final_id"):
            register_valid_dependency(connection_set, target_id, current_node["proves_final_id"], valid_ids)

        # 2. Extract LaTeX label references
        latex_refs = re.findall(r"\\(?:ref|eqref|autoref|pageref|cref|Cref)\*?\{([^{}]+)\}", raw_text)
        for label in latex_refs[:8]:
            linked_node = node_by_label.get(label) or node_by_key.get(label)
            if linked_node:
                register_valid_dependency(connection_set, linked_node["final_id"], target_id, valid_ids)

        # 3. Resolve textual structural numbering (e.g., "Theorem 5.14")
        textual_edges_found = 0
        for ref_word, ref_num in re.findall(
                r"\b(Theorem|Lemma|Corollary|Proposition|Claim|Conjecture|Definition|Algorithm)s?\s*~?\s*"
                r"([0-9]+(?:\.[0-9]+){0,3})\b", raw_text, flags=re.IGNORECASE):

            struct_id = type_to_char[ref_word.lower()] + ref_num
            target_node = node_by_struct_id.get(struct_id)

            if target_node and target_node["final_id"] != target_id:
                if target_node.get("source_order", 0) <= current_node.get("source_order", 0) or current_node[
                    "type"] == "proof":
                    register_valid_dependency(connection_set, target_node["final_id"], target_id, valid_ids)
                    textual_edges_found += 1
            if textual_edges_found >= 4:
                break

        # 4. Resolve relative pointers (e.g., "previous lemma")
        for category in ("lemma", "theorem", "corollary", "definition", "algorithm"):
            if re.search(r"\b(?:previous|preceding|above|last)\s+" + category + r"\b", text_lower):
                referenced_node = history_by_category.get(category)
                if referenced_node:
                    register_valid_dependency(connection_set, referenced_node["final_id"], target_id, valid_ids)

        # 5. Resolve semantic name references (e.g., "by Sauer's lemma")
        if re.search(r"\b(?:by|using|uses|from|apply|applying|follows\s+from|according\s+to)\b", text_lower):
            semantic_edges_found = 0
            for source_node in reversed(node_collection[:idx]):
                if source_node["final_id"] == target_id:
                    continue
                if source_node["type"] not in {"definition", "theorem", "lemma", "corollary", "algorithm"}:
                    continue
                if _is_node_name_cited(source_node, raw_text):
                    register_valid_dependency(connection_set, source_node["final_id"], target_id, valid_ids)
                    semantic_edges_found += 1
                if semantic_edges_found >= 3:
                    break

        # Update historical trackers
        if current_node["type"] in {"theorem", "lemma", "corollary"}:
            last_significant_node = current_node
        if current_node["type"] in history_by_category or current_node["type"] in {"definition", "theorem", "lemma",
                                                                                   "corollary", "algorithm"}:
            history_by_category[current_node["type"]] = current_node

    return connection_set




def filter_and_validate_dag_edges(candidate_connections, node_collection):
    """
    Filters connections to ensure the resulting graph is a strictly Directed Acyclic Graph (DAG)
    and limits the in-degree of non-proof nodes to prevent dense hub creation.
    """
    # LAYER 3 DEFENSE: Graceful exit if no edges exist at all
    if not candidate_connections:
        return set()

    from collections import Counter, defaultdict

    position_index = {node.get("final_id"): node.get("source_order", 0) for node in node_collection}
    proof_node_ids = {node.get("final_id") for node in node_collection if node.get("type") == "proof"}

    # Sort edges to prioritize proofs and physically closer references
    ranked_connections = sorted(
        candidate_connections,
        key=lambda connection: (
            0 if connection[0] in proof_node_ids else 1,
            abs(position_index.get(connection[1], 0) - position_index.get(connection[0], 0)),
            connection,
        ),
    )

    validated_edges = set()
    forward_graph_map = defaultdict(set)


    def _detect_existing_route(origin_id, destination_id):
        """Helper to detect if a path already exists, preventing cycles."""
        search_queue = [origin_id]
        explored_nodes = set()

        while search_queue:
            active_node = search_queue.pop()
            if active_node == destination_id:
                return True
            if active_node in explored_nodes:
                continue

            explored_nodes.add(active_node)
            search_queue.extend(forward_graph_map[active_node] - explored_nodes)

        return False

    in_degree_tracker = Counter()

    for source_node, target_node in ranked_connections:
        # Cap incoming edges to prevent hub-nodes (unless it's a proof edge)
        if in_degree_tracker[target_node] >= 4 and source_node not in proof_node_ids:
            continue

        # Cycle Check: If we can already reach the source from the target,
        # adding this edge would create a loop.
        if _detect_existing_route(target_node, source_node):
            continue

        validated_edges.add((source_node, target_node))
        forward_graph_map[source_node].add(target_node)
        in_degree_tracker[target_node] += 1

    return validated_edges


def build_edge_catalog(node_collection, max_len=22000):
    """
    Replaces: compact_catalog_for_edge_prompt
    Builds a condensed string representation of the nodes for the LLM to process.
    """
    catalog_lines = []

    for node in node_collection:
        if node.get("type") == "proof":
            continue

        line_entry = (
            f"{node.get('final_id')} | {node.get('type')} | "
            f"{node.get('name') or ''} | {node.get('statement') or node.get('content') or ''}"
        )

        # Stop appending if the next line exceeds our character limit for the prompt
        if sum(len(line) + 1 for line in catalog_lines) + len(line_entry) > max_len:
            break

        catalog_lines.append(line_entry)

    return "\n".join(catalog_lines)


def predict_semantic_edges(node_collection, llm_model_id):
    """
    Queries the LLM using a condensed catalog of nodes to find implicit,
    semantic relationships that deterministic parsing missed.
    """
    node_catalog = build_edge_catalog(node_collection)
    if not node_catalog:
        return []

    formatted_prompt = EDGE_PROMPT_TEMPLATE.replace("{catalog}", node_catalog)

    try:
        # Use the standard wrapper we built earlier
        llm_response = call_LLM_model(formatted_prompt, llm_model_id, max_tokens=2200)
    except Exception as api_error:
        print(f"  Semantic edge prediction failed: {api_error}")
        return []

    json_payload = extract_json_from_response(llm_response)
    if not json_payload or "edges" not in json_payload:
        print("  -> LLM returned no edges (or failed to parse). Defaulting to empty list.")
        return []


    semantic_connections = []

    for connection in json_payload.get("edges", []):
        if not isinstance(connection, dict):
            continue

        origin_id = str(connection.get("source") or "").strip()
        destination_id = str(connection.get("target") or "").strip()

        if origin_id and destination_id:
            semantic_connections.append((origin_id, destination_id))

    return semantic_connections


def chunk_fragments_into_batches(fragments, max_chars=8000):
    """
    Groups extracted dictionaries into text batches to fit the LLM context window.
    """
    batches = []
    current_batch = []
    current_length = 0

    for frag in fragments:
        # Create a string representation of the fragment
        frag_text = f"Type: {frag['type']}\nContent: {frag['content']}\n\n"
        frag_len = len(frag_text)

        if current_length + frag_len > max_chars and current_batch:
            batches.append("\n".join(current_batch))
            current_batch = []
            current_length = 0

        current_batch.append(frag_text)
        current_length += frag_len

    if current_batch:
        batches.append("\n".join(current_batch))

    return batches

def compile_knowledge_graph(extracted_nodes, model_identifier, document_metadata):
    """
    Phase 2 Orchestrator: Takes deduplicated nodes, extracts all edges,
    enforces a DAG, and formats the final Gold Standard JSON.
    """
    print(f"Starting Edge Phase for {len(extracted_nodes)} nodes...")

    # Step 1: Deterministic Edge Extraction (Fast, 100% accurate, no API calls)
    # Using the function we rewrote earlier
    deterministic_edges = assemble_graph_connections(extracted_nodes)
    print(f"  -> Found {len(deterministic_edges)} deterministic edges.")

    # Step 2: Semantic Edge Extraction (LLM pass using the Registry method)
    # Only run if we have a reasonable amount of nodes to prevent context blowouts
    semantic_edges = []
    if 1 < len(extracted_nodes) <= 100:
        print("  -> Querying LLM for implicit semantic edges...")
        # Using the function we rewrote earlier
        semantic_edges = predict_semantic_edges(extracted_nodes, model_identifier)
        print(f"  -> Found {len(semantic_edges)} semantic edges.")
    else:
        print("  -> Skipping Semantic LLM edge pass (node count out of bounds).")

    # Step 3: Combine all candidate edges
    # Convert semantic edges to a set of tuples to match deterministic output and remove duplicates
    all_candidate_edges = deterministic_edges.union(set(semantic_edges))

    # Step 4: Graph Theory Enforcement (Break cycles, cap hubs)
    # Using the DAG validation function we rewrote earlier
    print("  -> Enforcing Directed Acyclic Graph (DAG) topology...")
    final_validated_edges = filter_and_validate_dag_edges(all_candidate_edges, extracted_nodes)

    # Step 5: Format to Gold Standard JSON specifications
    formatted_edges = [
        {"source": source_id, "target": target_id}
        for source_id, target_id in sorted(final_validated_edges)
    ]

    # Assemble the final payload
    gold_standard_graph = {
        "metadata": document_metadata,
        "nodes": extracted_nodes,
        "edges": formatted_edges
    }

    return gold_standard_graph


def extract_final_graph(latex_text, model_identifier, use_llm=True):
    """
    The Master Orchestrator: Glues Phase 1 (Nodes) and Phase 2 (Edges) together.
    """
    print("Extracting LaTeX fragments...")
    fragments = extract_fragments(latex_text)

    document_metadata = {
        "title": "Extracted Chapter",
        "annotator": "Automated LLM Pipeline" if use_llm else "Deterministic Parsing Only"
    }

    # If the user passed --no-llm, we skip the API calls.
    # Note: Without the LLM, we don't have standard IDs, so we just return the raw fragments.
    if not use_llm:
        print("LLM is disabled. Returning raw deterministic fragments...")
        return {
            "metadata": document_metadata,
            "nodes": fragments,
            "edges": []
        }

    print("Batching fragments...")
    # Ensure chunk_fragments_into_batches is defined in your extractor.py!
    fragment_batches = chunk_fragments_into_batches(fragments)

    print("Running Phase 1: LLM Node Extraction...")
    raw_nodes = extract_nodes_from_batches(fragment_batches, model_identifier)
    deduplicated_nodes = deduplicate_nodes(raw_nodes)

    print("Running Phase 2: Edge Extraction & Graph Compilation...")
    final_graph = compile_knowledge_graph(deduplicated_nodes, model_identifier, document_metadata)

    return final_graph

