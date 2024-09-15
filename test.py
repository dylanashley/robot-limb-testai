import os
import numpy as np

# Function to calculate moving average
def moving_average(data, window_size):
    if data.ndim > 1:  # Flatten data if not 1D
        data = data.flatten()
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

directory_path = './results/'  # Path where the .npy files are stored

# Open a file to save all results
with open("all_results.txt", "w") as result_file:

    # Process 'brownian' data files
    result_file.write("Brownian Data - Means and Standard Deviations:\n")

    brownian_files = [f for f in os.listdir(directory_path) 
                      if f.startswith('brownian_step_counts_kill') and f.endswith('.npy')]

    for npy_file in brownian_files:
        npy_file_path = os.path.join(directory_path, npy_file)
        data = np.load(npy_file_path)

        mean = np.mean(data)
        std_dev = np.std(data, ddof=1)  # Calculate standard deviation

        print(f"{npy_file} - Mean: {mean}, Standard Deviation: {std_dev}")
        result_file.write(f"{npy_file} - Mean: {mean}, Standard Deviation: {std_dev}\n")

    result_file.write("\n")

    # Moving average window size for PPO/AC data
    window_size = 15

    # Process 'ppo' and 'ac' data files
    algorithm_files = [f for f in os.listdir(directory_path) 
                       if (f.startswith('ppo_step_counts_kill') or f.startswith('ac_step_counts_kill')) 
                       and f.endswith('.npy')]

    # Organize files by algorithm and kill setting
    algorithms = {}
    for f in algorithm_files:
        parts = f.split('_')
        algorithm_name = parts[0]  # "ppo" or "ac"
        kill_setting = parts[3]    # "kill0", "kill1", etc.

        key = f"{algorithm_name}_{kill_setting}"
        if key not in algorithms:
            algorithms[key] = []
        algorithms[key].append(f)

    # Process each group of runs for each algorithm and kill setting
    for key, npy_files in algorithms.items():
        converged_values_list = []

        for npy_file in npy_files:
            npy_file_path = os.path.join(directory_path, npy_file)
            data = np.load(npy_file_path)

            # Apply moving average and get the last (converged) value
            smoothed_curve = moving_average(data, window_size)
            converged_value = smoothed_curve[-1]
            converged_values_list.append(converged_value)

        # Calculate the mean and standard deviation of converged values
        mean_converged_value = np.mean(converged_values_list)
        std_converged_value = np.std(converged_values_list, ddof=1)

        print(f"{key} - Mean Converged Value: {mean_converged_value}, Standard Deviation: {std_converged_value}")
        result_file.write(f"Converged values for {key}:\n")
        for idx, value in enumerate(converged_values_list):
            result_file.write(f"Test {idx+1}: {value}\n")
        result_file.write(f"Mean Converged Value: {mean_converged_value}\n")
        result_file.write(f"Standard Deviation: {std_converged_value}\n\n")

print("Processing completed. All results have been saved to 'all_results.txt'.")
