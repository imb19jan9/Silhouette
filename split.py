import os
from os import replace
import numpy as np

def main():
    os.makedirs('./size50/train', exist_ok=True)
    os.makedirs('./size50/test', exist_ok=True)

    _, _, filenames = next(os.walk('size50/'))
    sample_idx = np.random.choice(len(filenames), 50000, replace=False)

    for filename in filenames:
        os.rename(f'./size50/{filename}', f'./size50/train/{filename}')

    for idx in sample_idx:
        os.rename(f'./size50/train/{filenames[idx]}', f'./size50/test/{filenames[idx]}')

if __name__ == "__main__":
    main()