"""
Script to freeze requirements.txt to exact versions.
Run: python freeze_requirements.py
"""
import subprocess
import re

# Get current installed versions
result = subprocess.run(['pip', 'list', '--format=freeze'], capture_output=True, text=True)
installed = {}
for line in result.stdout.strip().split('\n'):
    if '==' in line:
        pkg, ver = line.split('==')
        installed[pkg.lower()] = ver

# Read current requirements.txt
with open('requirements.txt', 'r') as f:
    lines = f.readlines()

# Replace >= with == using installed versions
new_lines = []
for line in lines:
    line = line.strip()
    
    # Skip comments, empty lines, and special directives
    if not line or line.startswith('#') or line.startswith('--'):
        new_lines.append(line)
        continue
    
    # Extract package name (handle extras like [standard])
    match = re.match(r'^([a-zA-Z0-9_-]+)(\[.*?\])?([><=!]+)(.+?)(\s*#.*)?$', line)
    if match:
        pkg_name = match.group(1).lower()
        extras = match.group(2) or ''
        comment = match.group(5) or ''
        
        if pkg_name in installed:
            new_line = f"{pkg_name}{extras}=={installed[pkg_name]}{comment}"
            new_lines.append(new_line)
        else:
            new_lines.append(line)  # Keep original if not installed
    else:
        new_lines.append(line)

# Write frozen requirements
with open('requirements-frozen.txt', 'w') as f:
    f.write('\n'.join(new_lines) + '\n')

print("✅ Created requirements-frozen.txt with exact versions")
print("📋 Review the file, then replace requirements.txt if satisfied")
