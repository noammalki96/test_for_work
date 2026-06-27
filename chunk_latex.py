import re

def get_latex_body(latex_text):
    """
    Extracts the content between \begin{document} and \end{document}.
    Ignores the preamble.
    """
    match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', latex_text, re.DOTALL)
    if match:
        return match.group(1)
    return latex_text  # Fallback if tags are missing

def chunk_document(latex_text):
    """
    Splits the LaTeX body into chunks based on specific mathematical environments.
    Returns a list of strings (chunks).
    """
    body = get_latex_body(latex_text)
    
    # List of environments to split by. Added an optional '*' for unnumbered versions (e.g., theorem*)
    envs_list = r'theorem|lemma|definition|corollary|proof'
    
    # Regex explanation:
    # \\begin\{( ... \*?)\} -> Matches \begin{theorem}, \begin{lemma*}, etc. (Group 1)
    # (.*?) -> The content inside the environment (lazy match) (Group 2)
    # \\end\{\1\} -> Matches the exact corresponding \end tag
    pattern = re.compile(r'\\begin\{(' + envs_list + r')\*?\}(.*?)\\end\{\1\*?\}', re.DOTALL)
    
    chunks = []
    last_end = 0
    
    for match in pattern.finditer(body):
        start = match.start()
        end = match.end()
        
        # 1. Text before the environment (free text between end and the next begin)
        text_before = body[last_end:start].strip()
        if text_before:
            chunks.append(text_before)
            
        # 2. The environment block itself (from \begin to \end)
        math_block = body[start:end].strip()
        if math_block:
            chunks.append(math_block)
            
        last_end = end
        
    # 3. Remaining text after the last environment
    text_after = body[last_end:].strip()
    if text_after:
        chunks.append(text_after)
        
    return chunks

# Test block - only runs if you execute this file directly
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            content = f.read()
        res = chunk_document(content)
        print(f"Document split into {len(res)} chunks.")
        for i, c in enumerate(res[:10]): # Print first 3 chunks as a sanity check
            print(f"--- Chunk {i+1} ({len(c)} chars) ---")
            print(c[:200] + ("..." if len(c) > 200 else ""))