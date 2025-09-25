import sys
import os

# Add parent directory to sys.path to import nlu.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nlu import OceanographicNLU

def main():
    query = input("Enter your oceanographic query: ")
    nlu = OceanographicNLU()
    result = nlu.process_query(query)
    print("=== Query Frame ===")
    print(result)

if __name__ == "__main__":
    main()
