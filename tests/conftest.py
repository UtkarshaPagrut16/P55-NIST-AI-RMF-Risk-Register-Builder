import sys
from pathlib import Path

# make `import modules` work when pytest is run from any folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
