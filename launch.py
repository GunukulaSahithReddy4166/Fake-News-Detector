
import os
import subprocess
import sys
from pathlib import Path
# Change to the project directory
current_dir=Path(__file__).parent.resolve()
os.chdir(current_dir)

# Run the Flask app
subprocess.run([sys.executable, "app.py"])
