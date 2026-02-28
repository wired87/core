import os

path = 'core/guard.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

if_main_start = -1
comment_start = -1
comment_end = -1

for i, line in enumerate(lines):
    if line.startswith('if __name__ == "__main__":'):
        if_main_start = i
    if line.strip() == '"""' and if_main_start != -1 and i > if_main_start:
        if comment_start == -1:
            comment_start = i
        else:
            comment_end = i

print(f"Indices: {if_main_start}, {comment_start}, {comment_end}")

if if_main_start != -1 and comment_start != -1 and comment_end != -1:
    # Extract blocks
    # Block 1: before if_main
    part1 = lines[:if_main_start]
    
    # Block 2: if_main content (from if_main_start to comment_start)
    if_main_block = lines[if_main_start:comment_start]
    
    # Block 3: compile_pattern content (from comment_start+1 to comment_end)
    compile_param_block = lines[comment_start+1:comment_end]
    
    # Reassemble
    # Ensure there's a newline before if_main check
    new_lines = part1 + compile_param_block + ['\n\n'] + if_main_block
    
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("Fixed core/guard.py")
else:
    print("Could not find markers")
