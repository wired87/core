import sys
try:
    with open('output.log', 'r', encoding='utf-16') as f:
        print(f.read())
except Exception as e:
    print(f"Error reading utf-16: {e}")
    try:
        with open('output.log', 'r', encoding='utf-8') as f:
            print(f.read())
    except Exception as e:
        print(f"Error reading utf-8: {e}")
