import os
import ast
import re

def strip_python_comments(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # We can use AST to remove docstrings, and tokenize to remove # comments
        # But an easier robust way to remove # comments without breaking strings:
        import tokenize
        import io
        tokens = tokenize.generate_tokens(io.StringIO(content).readline)
        
        result = []
        last_lineno = -1
        last_col = 0
        
        for tok in tokens:
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                result.append(" " * (start_col - last_col))
                
            if token_type == tokenize.COMMENT:
                pass # Skip comment
            elif token_type == tokenize.STRING:
                # Remove docstrings: if it's a multiline string and standalone
                # Actually, AST is better for docstrings, but skipping COMMENT is enough for # comments
                result.append(token_string)
            else:
                result.append(token_string)
                
            last_lineno = end_line
            last_col = end_col
            
        cleaned_content = "".join(result)
        
        # Remove multiple empty lines
        cleaned_content = re.sub(r'\n\s*\n', '\n\n', cleaned_content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        print(f"Cleaned PY: {filepath}")
    except Exception as e:
        print(f"Error on {filepath}: {e}")

def strip_js_comments(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Regex to remove JS comments (both // and /* */) but not inside strings
        # This regex matches strings or comments, and we only keep the valid strings
        # regex source: https://stackoverflow.com/questions/5989315/regex-for-match-replacing-javascript-comments-both-multiline-and-inline
        pattern = re.compile(
            r'(".*?(?<!\\)"|\'.*?(?<!\\)\'|`.*?`)|(/\*.*?\*/|//[^\r\n]*$)',
            re.MULTILINE | re.DOTALL
        )

        def replacer(match):
            # If group 2 matched, it's a comment, so return nothing
            if match.group(2) is not None:
                return ""
            # Otherwise it's a string, keep it
            else:
                return match.group(1)

        cleaned = pattern.sub(replacer, content)

        # Remove multiple empty lines
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned)
        print(f"Cleaned JS/TS: {filepath}")
    except Exception as e:
        print(f"Error on {filepath}: {e}")

root_dir = r"c:\BTech\IPD\Final code"
skip_dirs = ['node_modules', '.git', '__pycache__', '.pytest_cache', 'android', 'ios', '.expo', 'scratch']

for subdir, dirs, files in os.walk(root_dir):
    dirs[:] = [d for d in dirs if d not in skip_dirs]
    for file in files:
        filepath = os.path.join(subdir, file)
        if file.endswith('.py') and file != 'strip.py':
            strip_python_comments(filepath)
        elif file.endswith(('.js', '.ts', '.jsx', '.tsx')):
            strip_js_comments(filepath)
