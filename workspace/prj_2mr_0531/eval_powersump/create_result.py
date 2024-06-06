import os
import json
from pathlib import Path
import pandas as pd

def list_subdirectories(directory):
    """
    List all subdirectories in the given directory.

    :param directory: Path to the directory
    :return: List of subdirectory names
    """
    try:
        return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    except FileNotFoundError:
        return f"Directory {directory} not found."
    except Exception as e:
        return str(e)

def read_json_file(file_path):
    """
    Read a JSON file and return its content.

    :param file_path: Path to the JSON file
    :return: Content of the JSON file
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return f"File {file_path} not found."
    except Exception as e:
        return str(e)

# Example usage
if __name__ == "__main__":
    rootpath = Path(__file__).parent
    directory_path = rootpath / "result/snn"
    subdirectories = list_subdirectories(directory_path)

    result=[]
    for subdir in subdirectories:
        avg_result_path = directory_path / subdir / "average_result.json"
        latency_stat_path = directory_path / subdir / "latency_stat.json"
        
        avg_result = read_json_file(avg_result_path)
        latency_stat = read_json_file(latency_stat_path)
        
        print(f"Subdirectory: {subdir}")
        print(f"Average Result: {avg_result}")
        print(f"Latency Stat: {latency_stat}")

        result.append(
            {
                "model type":"snn",
                "device":"AKD 1000",
                "size":subdir,
                "acc_mean":avg_result["accuracy_mean"],
                "acc_std":avg_result["accuracy_stddev"],
                "lat_mean":latency_stat["latency_mean"],
                "lat_std":latency_stat["latency_stddev"],
                "power_mean":avg_result["Avg_mean"]/1000, #Wに変換
                "power_std":avg_result["Avg_stddev"]/1000, #Wに変換
                }
        )

    result_db=pd.DataFrame(result)
    result_db = result_db.sort_values(by="power_mean")  # sizeカラムの昇順に並び替え
    result_db.to_csv(rootpath / "result.csv", index=False)

