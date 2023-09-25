import os
import matplotlib.pyplot as plt

# Path to the result files
results_path = './result_files'

# Dictionary to store test numbers and times for each test type
data = {
    'SEQ': {'test_nums': [], 'times': []},
    'MPI': {'test_nums': [], 'times': []},
    'OMP': {'test_nums': [], 'times': []},
    'HIP': {'test_nums': [], 'times': []}
}

# Get all unique test numbers across all files for X-axis
all_test_nums = sorted(list(set(int(filename.split('_')[1]) for filename in os.listdir(results_path))))

# Iterate through each file in the result_files directory
for filename in sorted(os.listdir(results_path)):
    with open(os.path.join(results_path, filename), 'r') as file:
        lines = file.readlines()
        
        # Extract time from the "Time:" line
        time_line = next((line for line in lines if line.startswith("Time:")), None)
        if time_line:
            time_val = float(time_line.split()[1])
            
            # Extract test number and type from the filename (assuming filenames like test_01_seq.txt)
            test_num = int(filename.split('_')[1])
            test_type = filename.split('_')[2].split('.')[0].upper()  # Extracting 'SEQ', 'MPI', 'OMP', or 'HIP' from filename
            
            data[test_type]['test_nums'].append(test_num)
            data[test_type]['times'].append(time_val)

# Plotting the data
plt.figure(figsize=(12, 7))

# Plotting each test type
colors = {'SEQ': 'blue', 'MPI': 'red', 'OMP': 'green', 'HIP': 'purple'}
for test_type, color in colors.items():
    # Filling in times for missing test numbers with None
    filled_times = []
    for test in all_test_nums:
        if test in data[test_type]['test_nums']:
            index = data[test_type]['test_nums'].index(test)
            filled_times.append(data[test_type]['times'][index])
        else:
            filled_times.append(None)
    plt.plot(all_test_nums, filled_times, marker='o', linestyle='-', color=color, label=test_type)

plt.yscale('log')
plt.xlabel('Test Number')
plt.ylabel('Time (in seconds)')
plt.title('Execution Time for Tests')
plt.grid(True, which="both", ls="--", c='0.65')
plt.yticks([0.000001, 0.001, 0.1, 1, 10, 100, 200])
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
