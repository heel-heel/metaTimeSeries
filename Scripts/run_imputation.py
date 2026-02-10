import os
import subprocess

Imputation_Algorithms = {
    #'Mean': 'Mean.py',
    #'Median': 'Median.py',
    #'Mode': 'Mode.py',
    #'KNN': 'KNN.py',
    #'HDI': 'HDI.py',
    #'MICE': 'MICE.py',
    'IIM': 'IIM.py',
    #'SI': 'SI.py',
    #'MFI': 'MFI.py',
    #'MissFI': 'MissFI.py',
    #'XGBI': 'XGBI.py',
    #'GAIN': 'GAIN.py',
    #'MIDAE': 'MIDAE.py'
}
script_base_path = "../Imputation_Algorithms"
datasets = {
    "Illness": {"time_column": "date", "target_column": "OT"},
}
#Missing_rate = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
Missing_rate = [90]
base_path = "../Datasets"

def run_imputation(input_path, output_path, target_column, time_column, method, rate):
    input_file = os.path.join(input_path, f'null_all-{rate}.csv')
    output_file = os.path.join(output_path, f'null_all-{method}-{rate}.csv')
    script_name = Imputation_Algorithms[method]
    script = os.path.join(script_base_path, script_name)

    command = [
        'python', script,
        input_file,
        output_file,
        target_column,
        time_column
    ]
    subprocess.run(command, check=True)

    print(f"Completed 'null_all-{method}-{rate}'.")

if __name__ == "__main__":
    for dataset, columns in datasets.items():
        target_column = columns["target_column"]
        time_column = columns["time_column"]
        input_path = os.path.join(base_path, dataset, "null_all")
        for method in Imputation_Algorithms.keys():
            output_path = os.path.join(base_path, dataset, "Imputation", f"null_all-{method}")
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            for rate in Missing_rate:
                run_imputation(input_path, output_path, target_column, time_column, method, rate)
    print("All imputation processes completed.")