NODE_PROMPT_TEMPLATE = """
You are an expert mathematical knowledge extractor.
I will provide you with a batch of extracted LaTeX text fragments from a textbook.
Your task is to convert these fragments into a strict JSON array of node objects.

RULES FOR IDs:
You MUST generate a strict 'id' for every node using this exact format: 
[TypeLetter][Number]_[snake_case_name]
- TypeLetters: D (Definition), T (Theorem), L (Lemma), C (Corollary), P (Proof), A (Algorithm)
- Example: D2.1_unit_ball, T3.4_singular_value_decomposition, P3.4_singular_value_decomposition

SCHEMA REQUIREMENTS:
Each node must be a JSON object with the following fields:
- "id": The strict ID you generated.
- "type": One of [definition, theorem, lemma, corollary, proof, algorithm].
- "name": The short, human-readable name (e.g., "Singular Value Decomposition") or null.
- "statement": The clean, plain-text math statement (keep inline LaTeX like $x^2$).
- "assumptions": [Array of strings] ONLY if it's a theorem/lemma.
- "conclusions": [Array of strings] ONLY if it's a theorem/lemma.
- "proves": [String ID of the target theorem] ONLY if it's a proof.

TEXT FRAGMENTS:
{fragments_batch}

Output ONLY valid JSON. Start with `[` and end with `]`. Do not write anything else.
"""

EDGE_PROMPT_TEMPLATE = """
You are an expert mathematical knowledge graph builder.
I will provide you with a "Node Catalog"—a list of mathematical concepts, theorems, and definitions extracted from a single chapter.

Your task is to identify implicit, semantic dependencies between these nodes that are NOT obvious from their proximity in the text. 
For example: If a Theorem uses a specific Definition in its statement, or if an Algorithm relies on a Lemma to function, draw an edge from the Definition/Lemma (source) to the Theorem/Algorithm (target).

### STRICT RULES:
1. NO HALLUCINATIONS: You MUST ONLY use the exact IDs provided in the Node Catalog below. Do not invent new IDs.
2. DIRECTIONALITY: The `source` is the foundational concept. The `target` is the concept that builds upon, uses, or proves the source.
3. THE FALLBACK: If there are absolutely no logical dependencies between the nodes provided, you MUST output exactly: {"edges": []}
4. FORMAT: Output ONLY a valid JSON object. No explanations.

### NODE CATALOG:
{catalog}

### EXPECTED JSON FORMAT:
{
  "edges": [
    {"source": "D3.4_singular_value_decomposition", "target": "T3.6_best_rank_k"},
    {"source": "L2.1", "target": "A5_principal_component_analysis"}
  ]
}
"""