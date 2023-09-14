#!/usr/bin/env python3.6

import sys
import os
import subprocess

# Define the test_suite array with the desired information
test_suite = [
    {
        'size': '35',
        'test_files': ['test_01_a35_p8_w1', 'test_01_a35_p7_w2', 'test_01_a35_p5_w3', 'test_01_a35_p8_w4']
    },
    {
        'size': '30000',
        'test_files': ['test_02_a30k_p20k_w1', 'test_02_a30k_p20k_w2', 'test_02_a30k_p20k_w3', 'test_02_a30k_p20k_w4', 'test_02_a30k_p20k_w5', 'test_02_a30k_p20k_w6']
    },
    {
        'size': '20',
        'test_files': ['test_03_a20_p4_w1']
    },
    {
        'size': '20',
        'test_files': ['test_04_a20_p4_w1']
    },
    {
        'size': '20',
        'test_files': ['test_05_a20_p4_w1']
    },
    {
        'size': '20',
        'test_files': ['test_06_a20_p4_w1']
    },
    {
        'size': '1000000',
        'test_files': ['test_07_a1M_p5k_w1', 'test_07_a1M_p5k_w2', 'test_07_a1M_p5k_w3', 'test_07_a1M_p5k_w4']
    },
    {
        'size': '100000000',
        'test_files': ['test_08_a100M_p1_w1', 'test_08_a100M_p1_w2', 'test_08_a100M_p1_w3']
    },
    {
        'size': '16',
        'test_files': ['test_09_a16-17_p3_w1']
    },
    {
        'size': '17',
        'test_files': ['test_09_a16-17_p3_w1']
    }
]

def verify_result(test_num, mode):
    seq_file = './result_files/test_0{}_seq.txt'.format(test_num)
    compare_file = './result_files/test_0{}_{}.txt'.format(test_num, mode)

    with open(seq_file, 'r') as f_seq, open(compare_file, 'r') as f_compare:
        seq_lines = f_seq.readlines()
        compare_lines = f_compare.readlines()

        seq_result = next((line for line in seq_lines if line.startswith("Result:")), None)
        compare_result = next((line for line in compare_lines if line.startswith("Result:")), None)

        if seq_result == compare_result:
            print("Test {} for mode {} matches seq result.".format(test_num, mode))
        else:
            print("Error: Test {} for mode {} does not match seq result!".format(test_num, mode))

def run_program(prog_name, timeout_duration=600):  # 120 seconds = 2 minutes
    # Ensure result_files directory exists
    if not os.path.exists('./result_files'):
        os.makedirs('./result_files')

    for idx, test in enumerate(test_suite):
        size_arg = test['size']
        files_arg = [os.path.join('./test_files', f) for f in test['test_files']]
        args = ['srun', './' + prog_name, size_arg] + files_arg

#        print("Running command:", " ".join(args))  # Print the command

        mode = 'seq' if 'seq' in prog_name else ('omp' if 'omp' in prog_name else ('mpi' if 'mpi' in prog_name else 'hip'))
        output_file = './result_files/test_0{}_{}.txt'.format(idx+1, mode)
        
        with open(output_file, 'w') as f:
            process = subprocess.Popen(args, stdout=f)
            try:
                _, err = process.communicate(timeout=timeout_duration)
                if process.returncode == 0:
                    print("Test {} for mode {} completed successfully.".format(idx+1, mode))
                else:
                    print("Error: {} returned the error code {} for size {}.".format(prog_name, process.returncode, size_arg))
                    sys.exit(1)
            except subprocess.TimeoutExpired:
                process.kill()
                print("Warning: {} timed out and was terminated for size {}.".format(prog_name, size_arg))

        # Verify results if mode is 'omp', 'mpi', or 'hip'
        if mode in ['omp', 'mpi', 'hip']:
            verify_result(idx+1, mode)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <seq/omp/mpi/hip>")
        sys.exit(1)
    
    arg = sys.argv[1]
    if arg == "seq":
        run_program("energy_storms_seq")
    elif arg == "omp":
        run_program("energy_storms_omp")
    elif arg == "mpi":
        run_program("energy_storms_mpi")
    elif arg == "hip":
        run_program("energy_storms_hip")
    else:
        print("Invalid argument. Please use either 'seq', 'omp', 'mpi', or 'hip'.")
        sys.exit(1)

