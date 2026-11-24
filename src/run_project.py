
import os
import sys

# Find the main project folder
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# Set the project folder as the working directory
os.chdir(PROJECT_ROOT)

# Allow Python to find files inside src
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

# Check whether both datasets are available
matches_path = os.path.join("data", "matches.csv")
deliveries_path = os.path.join("data", "deliveries.csv")

if not os.path.exists(matches_path):
    raise FileNotFoundError(
        "matches.csv is missing. Please put it inside the data folder."
    )

if not os.path.exists(deliveries_path):
    raise FileNotFoundError(
        "deliveries.csv is missing. Please put it inside the data folder."
    )

print("====================================")
print("CRICKET INTELLIGENCE PROJECT")
print("====================================")

print("\nStep 1: Running data processing and analysis...")

# This import executes data_processing.py
from data_processing import spark

try:
    print("\nStep 2: Running machine learning models...")

    # This import executes machine_learning.py
    import machine_learning

    
    print("PROJECT EXECUTION COMPLETED")
   

finally:
    spark.stop()
    print("Spark session stopped.")
