import sys
from pathlib import Path

def define_project_root_path():
    toFind = ("pyproject.toml", "setup.py", ".git")
    
    cwd = Path.cwd().resolve()
    
    for folder in (cwd, *cwd.parents):
        if any((folder / file).exists() for file in toFind ):
            sys.path.insert(0, str(folder))
            return folder
    raise FileNotFoundError(f"Could not find project root: {cwd}")