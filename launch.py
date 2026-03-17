te_path>C:/Users/sahit/OneDrive/Desktop/fakenews detector/launch.py</absolute_path>
<parameter name="content">import os
import subprocess
import sys

# Change to the project directory
os.chdir(r"C:\Users\sahit\OneDrive\Desktop\fakenews detector")

# Run the Flask app
subprocess.run([sys.executable, "app.py"])
