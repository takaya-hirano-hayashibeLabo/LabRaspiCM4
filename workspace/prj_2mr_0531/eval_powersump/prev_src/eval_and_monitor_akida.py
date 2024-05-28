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
from filters import *
from multiprocessing import Process, Value

DEVICE = devices()[0]
DEVICE.soc.power_measurement_enabled = True

import time
import csv
import matplotlib.pyplot as plt
from akida import devices, Device
import argparse
from pathlib import Path

def monitor_power_events(device: Device):
    power_event = device.soc.power_meter.events()
    floor_power = device.soc.power_meter.floor

    if len(power_event) == 0:
        return (0, 0, 0)
    
    return (power_event[-1].ts, power_event[-1].power, floor_power)

def monitor_with_interval(device: Device, interval: int, duration: int, output_file: str):
    start_time = time.time()
    data = []
    while time.time() - start_time < duration:
        power_event = monitor_power_events(device)
        print(f"Timestamp: {power_event[0]}, Power: {power_event[1]} mW, Floor Power: {power_event[2]} mW")
        data.append(power_event)
        time.sleep(interval)
    
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Power (mW)", "Floor Power (mW)"])
        writer.writerows(data)
    
    timestamps = [event[0] for event in data]
    powers = [event[1] for event in data]
    
    plt.figure()
    plt.plot(timestamps, powers, label='Power (mW)')
    plt.xlabel('Timestamp')
    plt.ylabel('Power (mW)')
    plt.title('Power Consumption Over Time')
    plt.legend()
    plt.savefig('power_events_plot.png')


def inference_process(device,img_size, savepath):
    model_type = "snn"
    savepath = Path(savepath) / model_type / f"size{img_size}"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    model_root = Path("/home/neurobo/dev/projects/workspace/prj_2mr_0531/models")
    modelname = "snn.fbz"
    modelpath = model_root / f"{model_type.lower()}/size{img_size}/result" / modelname

    model = AkidaModel(str(modelpath))
    model.map(device)
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
    for i in range(num_batches):
        batch_inputs = test_in[i * batchsize:(i + 1) * batchsize]
        batch_labels = test_label[i * batchsize:(i + 1) * batchsize].astype(np.int8)
        
        out = np.squeeze(model.forward(batch_inputs.astype(np.uint8)))
        predict = np.argmax(out, axis=-1)
        acc += np.sum(predict == batch_labels)
        datasize += len(batch_inputs)

        print(model.statistics)

    acc = acc / datasize
    print(f"\n{'='*40}\nAccuracy: {acc:.2%}\n{'='*40}")

    runtime = time.time() - start_time
    print(f"Total Runtime: {runtime:.2f} seconds\n{'='*40}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", default=16, type=int)
    parser.add_argument("--savepath", default="result")
    parser.add_argument("--interval", type=int, default=0.5, help="Monitoring interval (seconds)")
    parser.add_argument("--duration", type=int, default=30, help="Monitoring duration (seconds)")
    args = parser.parse_args()

    savepath = Path(__file__).parent / args.savepath
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    inference_proc = Process(target=inference_process, args=(DEVICE,args.img_size, savepath))
    monitor_proc = Process(target=monitor_with_interval, args=(DEVICE, args.interval, args.duration, savepath / "power_events.csv"))

    inference_proc.start()
    monitor_proc.start()

    inference_proc.join()
    monitor_proc.join()

if __name__ == "__main__":
    main()