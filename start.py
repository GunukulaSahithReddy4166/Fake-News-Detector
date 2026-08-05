import os
import sys
from pathlib import Path
current_dir=Path(__file__).parent.resolve()
os.chdir(current_dir)
os.system(f"{sys.executable } app.py")
