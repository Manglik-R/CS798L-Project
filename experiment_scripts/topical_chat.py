import pandas as pd
import time
import os
from DPNE import DPNE



if __name__ == "__main__":
    # Start timing
    start_time = time.time()
    
    # Read the MSNBC data
    file_path = os.path.join('topical_chat', 'topical_chat.csv')
    df = pd.read_csv(file_path)

    grouped = df.groupby('conversation_id')['message'].apply(lambda msgs: ' '.join(str(m) for m in msgs))

    topical_data = grouped.tolist()
    epsilon = 800
    delta = 1e-5
    p = 0.01
    T = 9  

    filename = f'topical_chat_epsilon_{epsilon}_Delta_300'
    # Run the DPNE algorithm
    dpne = DPNE(topical_data, epsilon, delta, p, T, Deltas=[300]*T)
    output = dpne.run(filename)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    if output is not None:
        print("Output:", output)
    
    # Print execution time
    print(f"Execution time: {elapsed_time:.2f} seconds")
    print(f"Execution time: {elapsed_time/60:.2f} minutes")