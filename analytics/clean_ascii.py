"""Strip ALL non-ASCII characters from all Python files in the project"""
import re, glob, os

def clean_file(filepath):
    content = open(filepath, 'r', encoding='utf-8').read()
    # Replace specific Unicode chars with ASCII equivalents
    replacements = {
        '\u2500': '-',   # box drawing horizontal
        '\u2502': '|',   # box drawing vertical
        '\u250c': '+',   # box drawing corner
        '\u2510': '+',
        '\u2514': '+',
        '\u2518': '+',
        '\u251c': '+',
        '\u2524': '+',
        '\u252c': '+',
        '\u2534': '+',
        '\u253c': '+',
        '\u2550': '=',   # double horizontal
        '\u2551': '|',
        '\u2554': '+',
        '\u2557': '+',
        '\u255a': '+',
        '\u255d': '+',
        '\u2560': '+',
        '\u2563': '+',
        '\u2566': '+',
        '\u2569': '+',
        '\u256c': '+',
        '\u2713': '[OK]', # checkmark
        '\u2714': '[OK]',
        '\u2718': '[x]',
        '\u2705': '[OK]', # green checkmark emoji
        '\u274c': '[x]',  # red X emoji
        '\u2192': '->',   # right arrow
        '\u2190': '<-',
        '\u2191': '^',
        '\u2193': 'v',
        '\u00b7': '.',    # middle dot
        '\u00b0': ' deg', # degree
        '\u00e9': 'e',    # e acute
        '\u00e8': 'e',    # e grave
        '\u2248': '~=',   # almost equal
        '\u2265': '>=',
        '\u2264': '<=',
        '\u2260': '!=',
        '\u00d7': 'x',    # multiplication sign
        '\u03bc': 'mu',   # mu
        '\u03c3': 'sigma',
        '\u00b2': '^2',   # superscript 2
    }
    fixed = content
    for bad, good in replacements.items():
        fixed = fixed.replace(bad, good)
    # Final sweep: replace any remaining non-ASCII with '?'
    fixed = re.sub(r'[^\x00-\x7F]', '?', fixed)
    if fixed != content:
        open(filepath, 'w', encoding='utf-8').write(fixed)
        return True
    return False

files = glob.glob('**/*.py', recursive=True)
changed = 0
for f in files:
    if clean_file(f):
        print(f'Cleaned: {f}')
        changed += 1
print(f'Done. Cleaned {changed} files.')
