import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exploration_class import process_single_mouse

def main():
    ROOT_DIR = Path(__file__).parent
    MOUSES_DATA_PATH = ROOT_DIR / "project_data"
    MOUSES_DATA_FOLDERS = [
        os.path.join(MOUSES_DATA_PATH, folder)
        for folder in os.listdir(MOUSES_DATA_PATH)
        if not folder.startswith("__MACOSX")
        and os.path.isdir(os.path.join(MOUSES_DATA_PATH, folder))
    ]

    overall_start = time.time()

    # Parallel processing of mice using multiprocessing
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_single_mouse, MOUSES_DATA_FOLDERS)

    overall_end = time.time()
    total_time = (overall_end - overall_start) / 60
    print(f"\nTotal processing time: {total_time:.2f} minutes")


if __name__ == "__main__":
    main()