import time
import csv
import matplotlib.pyplot as plt
from akida import devices, Device
import argparse
from pathlib import Path

def monitor_power_events(device: Device):
    power_event = device.soc.power_meter.events()
    floor_power = device.soc.power_meter.floor  # idle状態の消費電力

    if len(power_event)==0:
        return (0,0,0)
    
    return (power_event[-1].ts, power_event[-1].power, floor_power)

def monitor_with_interval(device: Device, interval: int, duration: int, output_file: str):
    start_time = time.time()
    data = []
    while time.time() - start_time < duration:
        power_event = monitor_power_events(device)
        print(f"Timestamp: {power_event[0]}, Power: {power_event[1]} mW, Floor Power: {power_event[2]} mW")
        data.append(power_event)
        time.sleep(interval)
    
    # CSVにデータを書き込む
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Power (mW)", "Floor Power (mW)"])
        writer.writerows(data)
    
    # グラフを作成して保存
    timestamps = [event[0] for event in data]
    powers = [event[1] for event in data]
    
    plt.figure()
    plt.plot(timestamps, powers, label='Power (mW)')
    plt.xlabel('Timestamp')
    plt.ylabel('Power (mW)')
    plt.title('Power Consumption Over Time')
    plt.legend()
    plt.savefig('power_events_plot.png')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=0.1, help="Monitoring interval (seconds)")
    parser.add_argument("--duration", type=int, default=60, help="Monitoring duration (seconds)")
    parser.add_argument("--savepath", type=str, default="result", help="Output file name")
    parser.add_argument("--img_size",type=int,default=128,help="Image size")
    args = parser.parse_args()

    savepath=Path(__file__).parent/args.savepath/f"snn/size{args.img_size}"
    if not savepath.exists():
        savepath.makedirs(parents=True, exist_ok=True)
    INTERVAL = args.interval
    DURATION = args.duration
    OUTPUT_FILE = savepath/f"power_events.csv"


    DEVICE = devices()[0]
    DEVICE.soc.power_measurement_enabled = True
    monitor_with_interval(DEVICE, INTERVAL, DURATION, OUTPUT_FILE)

if __name__ == "__main__":
    main()
