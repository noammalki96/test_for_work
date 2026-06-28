#!/usr/bin/env python3


import argparse
import json
import re
import sys
import time  
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from call_llm import call_llm, APPROVED_MODELS
from chunk_latex import chunk_document

# The model this extractor uses. Set it to one of the three approved ids, as an exact
# literal string including the prefix:
#   "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-8B"
# On the cluster the grader reads THIS literal to launch the matching vLLM server, so
# change this line to pick your model. (To compare models without editing code, see the
# MODEL=... sweep in the README.)
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


# ==============================================================================
# --- Prompts Templates ---
# ==============================================================================
def group_and_merge_chunks(chunks, max_chars=5000):
    groups = {}
    ignored_free_text_count = 0  

    for chunk in chunks:
        match = re.match(r'^\s*\\begin\{([a-zA-Z]+)\*?\}', chunk)
        env_type = match.group(1).lower() if match else 'free_text'

       
        if env_type in ['proposition', 'claim', 'conjecture']:
            env_type = 'theorem'

        if env_type not in groups:
            groups[env_type] = []
        groups[env_type].append(chunk)

    print(f"  [Filter] Ignored {ignored_free_text_count} free text chunks.")

    final_batches = []
    for env_type, chunk_list in groups.items():
        current_batch = ""
        for chunk in chunk_list:
            if len(current_batch) + len(chunk) > max_chars:
                final_batches.append({'env_type': env_type, 'text': current_batch})
                current_batch = chunk
            else:
                current_batch += "\n\n" + chunk
        if current_batch:
            final_batches.append({'env_type': env_type, 'text': current_batch})

    return final_batches


def build_grouped_prompt(env_type, combined_text):
    prompt = f"""You are analyzing a block containing multiple mathematical objects of type '{env_type}'.

    IMPORTANT INSTRUCTION:
    1. Extract a SEPARATE node for EACH original environment found in the text.
    2. Follow ID naming rules: <type_letter><number>_<slug>.

    {COMMON_INSTRUCTIONS}

    Here is the combined text:
    {combined_text}
    """
    return prompt


def process_single_batch(batch, model):
    prompt = build_grouped_prompt(batch['env_type'], batch['text'])

    max_retries = 5
    base_delay = 2  # Start with a shorter 2-second delay

    for attempt in range(max_retries):
        try:
            response = call_llm(
                prompt,
                model=model,
                system_prompt="You are a precise mathematical knowledge extraction system. First, use <think> or <scratchpad> tags to briefly analyze the text, identify the entities, and map their relationships. Then, outside of those tags, output ONLY valid JSON containing a 'nodes' list.",
                max_tokens=3500,
                temperature=0.0,
            )
            result = extract_json_from_response(response)
            return result.get("nodes", []) if result else []

        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = base_delay * (2 ** attempt)  # Exponential backoff minimizes unnecessary idle time
                print(
                    f"  [Attempt {attempt + 1}/{max_retries}] Server not ready or failed ({e}). Waiting {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"  Failed after {max_retries} attempts.")
                raise e



COMMON_INSTRUCTIONS = r"""
Return ONLY valid JSON with this structure (DO NOT extract edges):
{
  "metadata": {"title": "...", "sections": "..."},
  "nodes": [...]
}

Strict Node ID Format Rules:
Your predicted ID MUST contain BOTH a numbering part AND 1-3 lowercase words describing the object, joined by underscores.
- Numbering Logic: You MUST extract the original numbering from the LaTeX text or `\\label{...}` if available (e.g., for `\\label{lem:11.5}` use `11.5`). If no original numbering is found, look at the last node you extracted, and increment its number to the next whole number (e.g., if the last was 11.5, use 12). If this is the first node, use 1.
- For Chapters: `<type_letter><number>_<1-3_words>` (e.g., "T2.3_cauchy", "D4.1_markov_chain").
- For Papers: `<type_letter><order_of_appearance>_<1-3_words>` (e.g., "T2_cauchy", "def_problem").
- Type letters: D=Definition, T=Theorem, L=Lemma, C=Corollary, P=Proof, A=Algorithm. 
  (Note: For papers, proposition/claim/conjecture all count as T).

Other Node Rules:
- "type": strictly one of: "definition", "theorem", "lemma", "corollary", "proof", "algorithm".
- "statement": Clean 1-3 sentence mathematical summary in your own words.
- "name": Descriptive name or null.
- "assumptions": list of assumption strings (for theorem/lemma/corollary).
- "conclusions": list of conclusion strings (for theorem/lemma/corollary).
- "proves": ID of the theorem/lemma being proved (for proofs).
- "proof_strategy": 1-3 short tags (for proofs).
"""


FREE_TEXT_PROMPT = r"""You are a mathematical knowledge extraction system. 
You are analyzing free text from a mathematical textbook or paper. 
Often, definitions or informal claims are hidden in the text (e.g., "We define the singular value as...").
Carefully read the text. If you find an implicit mathematical object, extract it as a node. 


""" + COMMON_INSTRUCTIONS


def build_node_extraction_prompt(chunk):

    # Regex that checks if the chunk starts with a formal math environment
    match = re.match(r'^\s*\\begin\{([a-zA-Z]+)\*?\}', chunk)

    if match:
        env_type = match.group(1).lower()


        if env_type in ['proposition', 'claim', 'conjecture']:
            env_type = 'theorem'


        if env_type == "definition":
            specific_prompt = r"""You are a mathematical knowledge extraction system.
You are extracting a mathematical DEFINITION from a formal LaTeX block. Here some examples you should study:
Example 1 input latex : label{def:4.1}
Fix $\varepsilon>0$. The $\varepsilon$-mixing time of a Markov chain is the minimum integert such that for any starting distribution $\mathbf{p}$, the 1 -norm difference between the $t$-step running average probability distribution ${ }^{16}$ and the stationary distribution is at most $\varepsilon$.
Example 1 output jason : {
      "id": "D4.1_epsilon_mixing_time",
      "type": "definition",
      "name": "Epsilon-mixing time",
      "section": "4.4",
      "statement": "Fix $\\varepsilon > 0$. The $\\varepsilon$-mixing time of a Markov chain is the minimum integer $t$ such that for any starting distribution $\\mathbf{p}$, the 1-norm difference between the $t$-step running average $\\mathbf{a}(t)$ and the stationary distribution $\\boldsymbol{\\pi}$ is at most $\\varepsilon$: $\\|\\mathbf{a}(t) - \\boldsymbol{\\pi}\\|_1 \\leq \\varepsilon$."
    }
Example 2 input latex : label{def:4.2}
For a subset $S$ of vertices, let $\pi(S)$ denote $\sum_{x \in S} \pi_{x}$. The normalized conductance $\Phi(S)$ of $S$ is

$$
\Phi(S)=\frac{\sum_{(x, y) \in(S, \bar{S})} \pi_{x} p_{x y}}{\min (\pi(S), \pi(\bar{S}))}
$$ \(\square\)

There is a simple interpretation of $\Phi(S)$. Suppose without loss of generality that $\pi(S) \leq \pi(\bar{S})$. Then, we may write $\Phi(S)$ as

$$
\Phi(S)=\sum_{x \in S} \underbrace{\frac{\pi_{x}}{\pi(S)}}_{a} \underbrace{\sum_{y \in \bar{S}} p_{x y}}_{b} .
$$

\footnotetext{${ }^{16}$ Recall that $\mathbf{a}(\mathbf{t})=\frac{1}{t}(\mathbf{p}(\mathbf{0})+\mathbf{p}(\mathbf{1})+\cdots+\mathbf{p}(\mathbf{t}-\mathbf{1}))$ is called the running average distribution.
}Here, $a$ is the probability of being in $x$ if we were in the stationary distribution restricted to $S$ and $b$ is the probability of stepping from $x$ to $\bar{S}$ in a single step. Thus, $\Phi(S)$ is the probability of moving from $S$ to $\bar{S}$ in one step if we are in the stationary distribution restricted to $S$.

It is easy to show that if we started in the distribution $p_{0, x}=\pi_{s} / \pi(S)$ for $x \in S$ and $p_{0, x}=0$ for $x \in \bar{S}$, the expected number of steps before we step into $\bar{S}$ is

$$
1 \Phi(S)+2(1-\Phi(S)) \Phi(S)+3(1-\Phi(S))^{2} \Phi(S)+\cdots=\frac{1}{\Phi(S)}
$$

Clearly, to be close to the stationary distribution, we must at least get to $\bar{S}$ once. So, mixing time is lower bounded by $1 / \Phi(S)$. Since we could have taken any $S$, mixing time is lower bounded by the minimum over all $S$ of $\Phi(S)$. We define this quantity to be the normalized conductance of the Markov Chain.

Example 2 output jason: {
      "id": "D4.2_normalized_conductance_set",
      "type": "definition",
      "name": "Normalized conductance of a set",
      "section": "4.4",
      "statement": "For a subset $S$ of vertices, the normalized conductance $\\Phi(S)$ is $\\Phi(S) = \\frac{\\sum_{(x,y) \\in (S, \\bar{S})} \\pi_x p_{xy}}{\\min(\\pi(S), \\pi(\\bar{S}))}$. Equivalently, $\\Phi(S)$ is the probability of stepping from $S$ to $\\bar{S}$ in one step when starting from the stationary distribution restricted to $S$."
    }

"""
            return specific_prompt + COMMON_INSTRUCTIONS + "\n\nHere is the formal LaTeX block:\n" + chunk

        elif env_type == "theorem":
            specific_prompt = r"""You are a mathematical knowledge extraction system.
You are extracting a mathematical THEOREM from a formal LaTeX block. Here some examples you should study from:

Example 1 input latex : [Law of Large Numbers]\label{thm:2.4}
Let $x_{1}, x_{2}, \ldots, x_{n}$ be $n$ independent samples of a random variable $x$. Then

$$
\operatorname{Prob}\left(\left|\frac{x_{1}+x_{2}+\cdots+x_{n}}{n}-E(x)\right| \geq \epsilon\right) \leq \frac{\operatorname{Var}(x)}{n \epsilon^{2}}
$$

Example 1 output jason : {
      "id": "T2.4_law_large_numbers",
      "type": "theorem",
      "name": "Law of Large Numbers",
      "section": "2.2",
      "statement": "For $n$ independent samples of $x$, the sample mean concentrates: $Pr(|\\bar{x} - E(x)| \\geq \\epsilon) \\leq \\frac{Var(x)}{n \\epsilon^2}$.",
      "assumptions": [
        "$x_1, \\ldots, x_n$ are $n$ independent samples of random variable $x$",
        "$x$ has finite variance"
      ],
      "conclusions": [
        "$Pr(|\\frac{x_1+\\cdots+x_n}{n} - E(x)| \\geq \\epsilon) \\leq \\frac{Var(x)}{n \\epsilon^2}$"
      ]
    }

Example 2 input latex: label{thm:9.7}
For $i=1,2, \ldots, d$, let $R_{i}=\left\{j \mid \hat{a}_{i j}=1\right\}$ at the end of the algorithm. Then, each nonempty $R_{i}=T_{l(i)}$, with $l(i)$ as in (9.9).

Example 2 output jason: {
      "id": "T9.7_recovery_dominant_topic",
      "type": "theorem",
      "name": "Recovery of dominant-topic clusters",
      "section": "9.8",
      "statement": "Under the dominant admixture model, each nonempty $R_i$ produced by thresholding and pruning equals $T_{l(i)}$.",
      "assumptions": [
        "$\\beta+\\rho\\leq(1-3\\delta)\\alpha$, $c_{lj}\\geq\\alpha$ for $j\\in T_l$, $c_{lj}\\leq\\beta$ otherwise",
        "$l(i)=\\arg\\max_{l'}b_{il'}$"
      ],
      "conclusions": [
        "Each nonempty $R_i=T_{l(i)}$",
        "The partition $T_1,\\ldots,T_r$ is fully recovered"
      ]
    }

"""
            return specific_prompt + COMMON_INSTRUCTIONS + "\n\nHere is the formal LaTeX block:\n" + chunk

        elif env_type == "lemma":
            specific_prompt = r"""You are a mathematical knowledge extraction system.
You are extracting a mathematical LEMMA from a formal LaTeX block. Here some examples you should study from :
Example 1 input latex : label{lem:2.6}
The surface area $A(d)$ and the volume $V(d)$ of a unit-radius ball in $d$ dimensions are given by

$$
A(d)=\frac{2 \pi^{\frac{d}{2}}}{\Gamma\left(\frac{d}{2}\right)} \quad \text { and } \quad V(d)=\frac{2 \pi^{\frac{d}{2}}}{d \Gamma\left(\frac{d}{2}\right)}
$$

To check the formula for the volume of a unit ball, note that $V(2)=\pi$ and $V(3)= \frac{2}{3} \frac{\pi^{\frac{3}{2}}}{\Gamma\left(\frac{3}{2}\right)}=\frac{4}{3} \pi$, which are the correct volumes for the unit balls in two and three dimensions. To check the formula for the surface area of a unit ball, note that $A(2)=2 \pi$ and $A(3)=\frac{2 \pi^{\frac{3}{2}}}{\frac{1}{2} \sqrt{\pi}}=4 \pi$, which are the correct surface areas for the unit ball in two and three dimensions. Note that $\pi^{\frac{d}{2}}$ is an exponential in $\frac{d}{2}$ and $\Gamma\left(\frac{d}{2}\right)$ grows as the factorial of $\frac{d}{2}$. This implies that $\lim _{d \rightarrow \infty} V(d)=0$, as claimed.

Example 1 output jason : {
      "id": "L2.6_volume_surface_area",
      "type": "lemma",
      "name": "Volume and surface area of unit ball",
      "section": "2.4.1",
      "statement": "$A(d) = \\frac{2 \\pi^{d/2}}{\\Gamma(d/2)}$ and $V(d) = \\frac{2 \\pi^{d/2}}{d \\Gamma(d/2)}$. The volume $V(d) \\to 0$ as $d \\to \\infty$.",
      "assumptions": [
        "$d$-dimensional unit ball in $R^d$"
      ],
      "conclusions": [
        "$A(d) = \\frac{2 \\pi^{d/2}}{\\Gamma(d/2)}$",
        "$V(d) = \\frac{2 \\pi^{d/2}}{d \\Gamma(d/2)}$",
        "$V(d) \\to 0$ as $d \\to \\infty$"
      ]
    }

Example 2 input latex : label{lem:3.2}
For any matrix $A$, the sum of squares of the singular values equals the square of the Frobenius norm. That is, $\sum \sigma_{i}^{2}(A)=\|A\|_{F}^{2}$.

Example 2 output jason : {
      "id": "L3.2_sum_squared_singular",
      "type": "lemma",
      "name": null,
      "section": "3.3",
      "statement": "The sum of the squared singular values of a matrix is strictly equal to the square of its Frobenius norm - $\\sum \\sigma_i(A)^2 = ||A||_F^2$",
      "assumptions": [
        "$A$ is any matrix"
      ],
      "conclusions": [
        "$\\sum \\sigma_i(A)^2 = ||A||_F^2$"
      ]
    }
"""
            return specific_prompt + COMMON_INSTRUCTIONS + "\n\nHere is the formal LaTeX block:\n" + chunk

        elif env_type == "corollary":
            specific_prompt = r"""You are a mathematical knowledge extraction system.
You are extracting a mathematical COROLLARY from a formal LaTeX block. Here some examples you should study from :
Example 1 input latex : label{cor:4.10}
If vertices $x$ and $y$ are connected by an edge, then $h_{x y}+h_{y x} \leq 2 m$ where $m$ is the number of edges in the graph.

Example 1 output jason : {
      "id": "C4.10_commute_time_bound",
      "type": "corollary",
      "name": "Commute time bound for adjacent vertices",
      "section": "4.6",
      "statement": "If vertices $x$ and $y$ are connected by an edge, then $h_{xy} + h_{yx} \\leq 2m$, where $m$ is the number of edges.",
      "assumptions": [
        "$x$ and $y$ are adjacent (share an edge)",
        "$m$ = number of edges"
      ],
      "conclusions": [
        "$h_{xy} + h_{yx} \\leq 2m$"
      ]
    }
Example 2 input latex : label{cor:4.11}
For vertices $x$ and $y$ in an $n$ vertex graph, the commute time, commute( $x, y$ ), is less than or equal to $n^{3}$.


Example 2 output jason : {
      "id": "C4.11_commute_time_general",
      "type": "corollary",
      "name": "Commute time general upper bound",
      "section": "4.6",
      "statement": "For any two vertices $x$ and $y$ in a connected $n$-vertex graph, $\\text{commute}(x, y) \\leq n^3$.",
      "assumptions": [
        "Connected graph with $n$ vertices",
        "$m \\leq \\binom{n}{2}$ edges"
      ],
      "conclusions": [
        "$\\text{commute}(x, y) \\leq n^3$"
      ]
    }
"""
            return specific_prompt + COMMON_INSTRUCTIONS + "\n\nHere is the formal LaTeX block:\n" + chunk

        elif env_type == "proof":
            specific_prompt = r"""You are a mathematical knowledge extraction system.
You are extracting a mathematical PROOF from a formal LaTeX block. Here some examples you should study from :
Pay special attention to the 'proves' field and the 'proof_strategy' tags.
Example 1 input latex : By the preceding discussion.\\
The vectors $\mathbf{v}_{\mathbf{1}}, \mathbf{v}_{\mathbf{2}}, \ldots, \mathbf{v}_{\mathbf{r}}$ are called the right-singular vectors. The vectors $A \mathbf{v}_{\mathbf{i}}$ form a fundamental set of vectors and we normalize them to length one by

$$
\mathbf{u}_{\mathbf{i}}=\frac{1}{\sigma_{i}(A)} A \mathbf{v}_{\mathbf{i}}
$$

Later we will show that $\mathbf{u}_{i}$ similarly maximizes $\left|\mathbf{u}^{T} A\right|$ over all $\mathbf{u}$ perpendicular to $\mathbf{u}_{1}, \ldots, \mathbf{u}_{i-1}$. These $\mathbf{u}_{i}$ are called the left-singular vectors. Clearly, the right-singular vectors are orthogonal by definition. We will show later that the left-singular vectors are also orthogonal.

Example 1 output jason : {
      "id": "P3.2_derived_calculating_sum",
      "type": "proof",
      "name": null,
      "section": "3.3",
      "statement": "Derived by calculating the sum of the squared projections of all rows of $A$ onto an orthonormal basis that includes the right-singular vectors and used in D3.3.4",
      "proves": "L3.2_sum_squared_singular",
      "proof_strategy": "orthonormal basis expansion, algebraic manipulation"
    }

Example 2 input latex : Clearly, if $A=B$ then $A \mathbf{v}=B \mathbf{v}$ for all $\mathbf{v}$. For the converse, suppose that $A \mathbf{v}=B \mathbf{v}$ for all $\mathbf{v}$. Let $\mathbf{e}_{\mathbf{i}}$ be the vector that is all zeros except for the $i^{\text {th }}$ component which has value one. Now $A \mathbf{e}_{\mathbf{i}}$ is the $i^{\text {th }}$ column of $A$ and thus $A=B$ if for each $i$, $A \mathbf{e}_{\mathbf{i}}=B \mathbf{e}_{\mathbf{i}}$.

Example 2 output jason : {
      "id": "P3.3_evaluates_matrices_against",
      "type": "proof",
      "name": null,
      "section": "3.4",
      "statement": "Evaluates the matrices against the standard basis vectors ($e_i$).",
      "proves": "L3.3_two_matrices_b",
      "proof_strategy": "standard basis evaluation"
    }
"""
            return specific_prompt + COMMON_INSTRUCTIONS + "\n\nHere is the formal LaTeX block:\n" + chunk

    return FREE_TEXT_PROMPT + "\n\nHere is the free LaTeX text:\n" + chunk


EDGE_EXTRACTION_PROMPT = r"""You are a precise mathematical knowledge extraction system.
Analyze the following list of mathematical nodes and discover the logical DIRECT dependencies between them.
An edge from A to B means object B directly uses, assumes, or is proved by object A.

STRICT RULES:
1. Output ONLY a valid JSON object. No conversational text. No explanations.
2. The JSON must have a single key "edges" containing a list of objects with "source" and "target".
3. Use ONLY the exact IDs provided in the list.
4. If there are no dependencies, output {"edges": []}.

Output format:
{
  "edges": [
    {"source": "id_1", "target": "id_2"}
  ]
}
"""


# ==============================================================================


def read_latex(path):
    """Read a LaTeX file, return its text content."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def extract_json_from_response(response):
    """Extract JSON object from LLM response (handles markdown blocks, reasoning tags)."""
    # Strip both <think> and <scratchpad> tags, including unclosed tags
    response = re.sub(r'<(think|scratchpad)>.*?</\1>', '', response, flags=re.DOTALL).strip()
    response = re.sub(r'<(think|scratchpad)>.*', '', response, flags=re.DOTALL).strip()

    # Try to find JSON in markdown code block
    match = re.search(r'```(?:json)?\s*\n?(.*?)```', response, re.DOTALL)
    if match:
        response = match.group(1).strip()

    # Try to find JSON object directly
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Last resort: try the whole response
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None


def is_time_running_out(start_time, time_limit_seconds=450):
    elapsed = time.time() - start_time
    return elapsed > time_limit_seconds


def extract_node_deterministically(chunk, env_type, node_counter):

    type_letters = {"definition": "D", "theorem": "T", "lemma": "L",
                    "corollary": "C", "proof": "P", "algorithm": "A"}
    letter = type_letters.get(env_type, "T")

  
    label_match = re.search(r'\\label\{([^}]+)\}', chunk)
    number_part = str(node_counter)
    if label_match:

        num_match = re.search(r'(\d+(?:\.\d+)+)', label_match.group(1))
        if num_match:
            number_part = num_match.group(1)


    name_match = re.search(r'\\begin\{[a-zA-Z]+\*?\}\s*\[(.*?)\]', chunk)
    name = name_match.group(1).strip() if name_match else None

    content_match = re.search(r'\\begin\{[a-zA-Z]+\*?\}(?:\s*\[.*?\])?(.*)\\end\{[a-zA-Z]+\*?\}', chunk, re.DOTALL)
    raw_content = content_match.group(1).strip() if content_match else chunk


    statement = re.sub(r'\\label\{[^}]+\}', '', raw_content)
    statement = re.sub(r'\s+', ' ', statement).strip()


    slug_source = name if name else statement[:40]
    slug_words = re.findall(r'[a-zA-Z]+', slug_source.lower())

    slug = "_".join(w for w in slug_words if len(w) > 2)[:3]
    if not slug:
        slug = "obj"

    node_id = f"{letter}{number_part}_{slug}"


    node = {
        "id": node_id,
        "type": env_type,
        "name": name,
        "section": None,
        "statement": statement[:1500]  
    }


    if env_type in ["theorem", "lemma", "corollary"]:
        node["assumptions"] = []
        node["conclusions"] = [statement[:500]]
    elif env_type == "proof":
        node["proves"] = ""
        node["proof_strategy"] = ["direct proof"]

    return node


def process_single_edge_window(window, model, window_index, total_windows):
    """Helper function to isolate the LLM call for ThreadPoolExecutor."""
    catalog_lines = []
    for n in window:
        name_str = f" ({n.get('name')})" if n.get('name') else ""
        stmt = n.get('statement', '')[:150].replace('\n', ' ')
        catalog_lines.append(f"ID: {n['id']} | Type: {n['type']}{name_str} | Statement: {stmt}")

    catalog_text = "\n".join(catalog_lines)
    edge_prompt = EDGE_EXTRACTION_PROMPT + "\n\nHere are the nodes for this batch:\n" + catalog_text

    print(f"    Calling LLM for edges window {window_index + 1}/{total_windows}...")
    try:
        edge_response = call_llm(
            edge_prompt,
            model=model,
            system_prompt="You are a precise mathematical knowledge extraction system. Output ONLY valid JSON with an 'edges' list.",
            max_tokens=1000,
            temperature=0.0,
        )
        edge_result = extract_json_from_response(edge_response)

        if edge_result and "edges" in edge_result:
            new_edges = edge_result["edges"]
            print(f"      Found {len(new_edges)} edges in window {window_index + 1}.")
            return new_edges
        else:
            print(f"      ERROR: Could not parse edges from LLM response for window {window_index + 1}.")
            return []

    except Exception as e:
        print(f"    [CRITICAL ERROR] LLM failed on edge window {window_index + 1}: {e}")
        return []


def extract_edges_in_windows(nodes_list, start_time, time_limit_seconds, model, skip_first_n_windows=0):
    master_edges = []
    if not nodes_list:
        return [], 0

    window_size = 35
    step = 20
    windows = []
    for i in range(0, len(nodes_list), step):
        windows.append(nodes_list[i:i + window_size])
        if i + window_size >= len(nodes_list):
            break

    print(f"    Total overlapping windows for these {len(nodes_list)} nodes: {len(windows)}.")

    # Filter out skipped windows and check timeouts before queuing
    windows_to_process = []
    for i, window in enumerate(windows):
        if i < skip_first_n_windows:
            print(f"    Skipping window {i + 1} (already processed).")
            continue

        if is_time_running_out(start_time, time_limit_seconds):
            print("    [TIMEOUT WARNING] Time is running out! Stopping edge extraction.")
            break

        windows_to_process.append((i, window))

    if not windows_to_process:
        return master_edges, len(windows)

    # Execute concurrent LLM calls
    total_windows = len(windows)
    max_workers = min(8, len(windows_to_process))  # Adjust max_workers based on your vLLM concurrency limits

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_edge_window, window, model, i, total_windows)
            for i, window in windows_to_process
        ]

        for future in as_completed(futures):
            result = future.result()
            if result:
                master_edges.extend(result)

    return master_edges, total_windows

def annotate_chapter(latex_path, model=DEFAULT_MODEL):

    start_time = time.time()

    latex = read_latex(latex_path)
    chunks = chunk_document(latex)

    master_nodes = []
    free_text_chunks = []
    node_counter = 1

    print(f"  Total chunks found: {len(chunks)}")

    # ==========================================
    # Extract Metadata Deterministically 
    # ==========================================

    title_match = re.search(r'\\(?:chapter|title)\*?\{([^}]+)\}', latex)
    chapter_title = title_match.group(1).strip() if title_match else "Ch. 5: Machine Learning"
    

    section_numbers = re.findall(r'\\section\*?\{([\d.]+)', latex)
    if section_numbers:

        unique_sections = sorted(list(set(section_numbers)), key=float)
        sections_range = f"{unique_sections[0]}-{unique_sections[-1]}"
    else:
        sections_range = "5.1-5.11" 

    # ==========================================
    # Phase 1A: 
    # ==========================================
    for chunk in chunks:
        match = re.match(r'^\s*\\begin\{([a-zA-Z]+)\*?\}', chunk)
        if match:
            env_type = match.group(1).lower()
            if env_type in ['proposition', 'claim', 'conjecture']:
                env_type = 'theorem'

            node = extract_node_deterministically(chunk, env_type, node_counter)
            if node:
                master_nodes.append(node)
                node_counter += 1
        else:
            free_text_chunks.append(chunk)


    seen_ids = set()
    det_nodes = []
    for node in master_nodes:
        if node.get("id") and node["id"] not in seen_ids:
            det_nodes.append(node)
            seen_ids.add(node["id"])

    print(f"  Phase 1A: Extracted {len(det_nodes)} nodes deterministically.")

    # ==========================================
    # Phase 2A: 
    # ==========================================
    print("  --- PHASE 2A: Extracting Edges for Deterministic Nodes ---")
    det_edges, processed_windows_count = extract_edges_in_windows(
        det_nodes, start_time, time_limit_seconds=240, model=model, skip_first_n_windows=0
    )

    # ==========================================
    # Phase 1B: Concurrent Free Text Extraction
    # ==========================================
    print("  --- PHASE 1B: Extracting Nodes from Free Text ---")
    batches = group_and_merge_chunks(free_text_chunks)

    # Pre-filter batches to respect timeouts before submitting to the thread pool
    batches_to_process = []
    for batch in batches:
        if is_time_running_out(start_time, time_limit_seconds=420):
            print("  [TIMEOUT WARNING] Time is running out! Skipping remaining free text extraction.")
            break
        batches_to_process.append(batch)

    if batches_to_process:
        # Match this max_workers logic to what we discussed for your vLLM sequence limits
        max_workers = min(8, len(batches_to_process))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(process_single_batch, batch, model): i
                for i, batch in enumerate(batches_to_process)
            }

            for future in as_completed(future_to_index):
                batch_index = future_to_index[future]
                try:
                    nodes = future.result()
                    master_nodes.extend(nodes)
                    print(f"  Finished LLM batch {batch_index + 1}/{len(batches)}. Added {len(nodes)} nodes.")
                except Exception as e:
                    print(f"  [CRITICAL ERROR] LLM connection failed on batch {batch_index + 1}: {e}")

    # ==========================================
    # Phase 2B:
    # ==========================================
    llm_edges = []
    if llm_added_count > 0 and not is_time_running_out(start_time, time_limit_seconds=510):
        print("  --- PHASE 2B: Extracting Edges for New Nodes ---")
        skip_windows = max(0, processed_windows_count - 1)
        llm_edges, _ = extract_edges_in_windows(
            final_nodes, start_time, time_limit_seconds=540, model=model, skip_first_n_windows=skip_windows
        )
    else:
        print("  [SKIP] Not enough time (or no new nodes) for Phase 2B. Moving to finish.")

    # ==========================================
    # 
    # ==========================================
    all_edges = det_edges + llm_edges
    unique_edges = []
    seen_edges = set()
    for edge in all_edges:
        if isinstance(edge, dict) and "source" in edge and "target" in edge:
            edge_tuple = (edge["source"], edge["target"])
            if edge_tuple not in seen_edges:
                unique_edges.append(edge)
                seen_edges.add(edge_tuple)

    print(f"  Final Phase complete. Total unique edges extracted: {len(unique_edges)}.")


    return {
        "metadata": {
            "title": chapter_title,
            "sections": sections_range,
            "annotator": "AI Hybrid Pipeline"
        },
        "nodes": final_nodes,
        "edges": unique_edges
    }


def main():
    parser = argparse.ArgumentParser(
        description="Chapter annotator: extracts a knowledge graph from LaTeX using Two-Pass LLM extraction.")
    parser.add_argument("input", help="Path to LaTeX chapter file (e.g., chapters/chapter4.tex)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output JSON file path")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"LLM model to use (default: {DEFAULT_MODEL}). "
                             f"Approved: {', '.join(APPROVED_MODELS)}")
    args = parser.parse_args()

    print(f"Annotating: {args.input}")
    print(f"Using model: {args.model}")

    result = annotate_chapter(args.input, model=args.model)

    if result is None:
        print("Extraction failed.")
        sys.exit(1)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()