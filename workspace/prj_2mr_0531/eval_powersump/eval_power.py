import argparse
from pathlib import Path
import numpy as np
import cv2  
import matplotlib.pyplot as plt
import json
import akida
from akida import Model as AkidaModel
from akida import devices, Device
import time
import os
from tqdm import tqdm


def inference_process(device, img_size, savepath):
    model_type = "snn"
    savepath = Path(savepath) / model_type / f"size{img_size}"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    model_root = Path("/home/neurobo/dev/projects/workspace/prj_2mr_0531/models")
    modelname = "snn.fbz"
    modelpath = model_root / f"{model_type.lower()}/size{img_size}/result" / modelname

    model = AkidaModel(str(modelpath))
    model.map(device)
    print("\n")
    model.summary()

    test_root = Path("/home/neurobo/dev/projects/workspace/prj_2mr_0531/test_data")
    test_in = np.load(test_root / f"size{img_size}/test_data.npy")
    test_label = np.load(test_root / f"size{img_size}/test_labels.npy")

    N = 5
    test_in = np.repeat(test_in, N, axis=0)
    test_label = np.repeat(test_label, N, axis=0)

    batchsize = 512
    num_batches = len(test_in) // batchsize

    start_time = time.time()
    acc = 0
    datasize = 0
    result_list=[]
    for i in tqdm(range(num_batches)):
        batch_inputs = test_in[i * batchsize:(i + 1) * batchsize]
        batch_labels = test_label[i * batchsize:(i + 1) * batchsize].astype(np.int8)
        
        out = np.squeeze(model.forward(batch_inputs.astype(np.uint8)))

        predict = np.argmax(out, axis=-1)
        acc_batch = np.sum(predict == batch_labels)
        acc+=acc_batch
        datasize += len(batch_inputs)

        result=model.statistics.powers
        # result["timestamp"]=timestamp
        result["accuracy"] = acc_batch / len(batch_inputs)
        result_list.append(result)

    acc = acc / datasize
    print(f"\n{'='*40}\nAccuracy: {acc:.2%}\n{'='*40}")

    runtime = time.time() - start_time
    print(f"Total Runtime: {runtime:.2f} seconds\n{'='*40}")

    # Save result_list to CSV
    import csv
    keys = result_list[0].keys()
    with open(savepath / 'result_list.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(result_list)

    import json

    # Calculate the average and standard deviation of each column in result_list
    average_result = {key: np.mean([d[key] for d in result_list]) for key in result_list[0].keys()}
    stddev_result = {key: np.std([d[key] for d in result_list]) for key in result_list[0].keys()}

    # Combine average and standard deviation results
    combined_result = {f"{key}_mean": average_result[key] for key in average_result}
    combined_result.update({f"{key}_stddev": stddev_result[key] for key in stddev_result})

    # Save the combined_result to a JSON file
    with open(savepath / 'average_result.json', 'w') as json_file:
        json.dump(combined_result, json_file, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", default=16, type=int)
    parser.add_argument("--savepath", default="result")
    parser.add_argument("--interval", type=int, default=0.1, help="Monitoring interval (seconds)")
    parser.add_argument("--duration", type=int, default=60, help="Monitoring duration (seconds)")
    args = parser.parse_args()

    savepath = Path(__file__).parent / args.savepath
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    device = devices()[0]
    device.soc.power_measurement_enabled = True

    inference_process(device, args.img_size, savepath)

if __name__ == "__main__":
    main()
