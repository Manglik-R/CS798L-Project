from DPNE import DPNE
import time
import matplotlib.pyplot as plt
import os

def read_msnbc_data(file_path='msnbc/msnbc990928.seq'):
    """
    Read the MSNBC dataset file.
    
    Each line in the file represents a user session.
    The first number in each line is the number of page categories visited.
    The remaining numbers are the ID of each page category visited in sequence.
    
    Returns:
        list of strings: Each string is a space-separated sequence of page visits
    """
    data = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 0:  # Skip empty lines
                # The first number is the count, the rest are the actual sequence
                sequence = ' '.join(parts[0:])
                data.append(sequence)
    
    return data[3:]

if __name__ == "__main__":
    # Fixed parameters
    epsilon = 4
    delta = 1e-5
    p = 0.01
    T = 9
    
    # Read the MSNBC data (only once)
    msnbc_data = read_msnbc_data()
    
    # List of Delta values to try
    delta_values = [1, 5, 10, 50]

    outputs = []
    
    for delta_value in delta_values:
        print(f"\n{'='*50}")
        print(f"Running with Delta = {delta_value}")
        print(f"{'='*50}")
        
        # Start timing
        start_time = time.time()
        
        # Create DPNE instance with current Delta value
        dpne = DPNE(msnbc_data, epsilon, delta, p, T, Deltas=[delta_value]*T)
        
        # Generate filename based on epsilon and delta
        filename = f"msnbc_epsilon_{epsilon}_Delta_{delta_value}"
        
        # Run the algorithm with the generated filename
        output = dpne.run(filename=filename)

        n_gram_lengths = []

        for n_gram in output:
            n_gram_lengths.append(len(n_gram))

        outputs.append(n_gram_lengths)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        print(f"Run completed for Delta = {delta_value}")
        print(f"Execution time: {elapsed_time/60:.2f} minutes")

    # Plotting the results
    save_dir = os.path.join('plots', f"msnbc_combined_epsilon_{epsilon}.png")
    plt.figure(figsize=(10, 6))
    for i, delta_value in enumerate(delta_values):
        plt.plot(outputs[i], label=f'Delta = {delta_value}')
    plt.xlabel('N')
    plt.ylabel('Number of N-grams')
    plt.title('Number of N-grams vs N for different Delta values')
    plt.legend()
    plt.grid()
    plt.savefig(save_dir)
    