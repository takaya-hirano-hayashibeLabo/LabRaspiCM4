from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import os
import cv2
import scipy.sparse
import scipy.sparse.linalg
from numba import njit


def save_video_with_ffmpeg(data,outpath:Path,outname, fps):
    """
    一旦cv2でエンコードして, 最後にffmpegでエンコードする
    これで速い＆vscodeで描画が可能
    """

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    height, width = data.shape[1], data.shape[2]
    is_color = (len(data.shape) == 4) and (data.shape[3] == 3)  # Check if data is color or grayscale
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Temporary codec for .avi file
    tmpout=str(outpath/"tmp.avi")
    out = cv2.VideoWriter(tmpout, fourcc, fps, (width, height))
    # out = cv2.VideoWriter(str(temp_output_path), fourcc, fps, (width, height), is_color)

    if not out.isOpened():
        print("Error: Could not open video writer.")
        return

    for frame in data:
        if not is_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        out.write(frame)

    out.release()

    # Re-encode the video using ffmpeg
    ffmpeg_command = [
        'ffmpeg', '-y','-i', tmpout, 
        '-pix_fmt', 'yuv420p', '-vcodec', 'libx264', 
        '-crf', '23', '-preset', 'medium', str(outpath/outname)
    ]    
    subprocess.run(ffmpeg_command)
    # Remove the temporary file
    os.remove(tmpout)    



def save_heatmap_video_with_ffmpeg(data, outpath: Path, outname, fps):
    """
    一旦cv2でエンコードして, 最後にffmpegでエンコードする
    これで速い＆vscodeで描画が可能
    """

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Save a sample frame with colorbar to determine the size
    plt.imshow(data[0], cmap='hot', interpolation='nearest')
    plt.axis('off')
    plt.colorbar()
    sample_path = outpath / 'sample_heatmap.png'
    plt.savefig(sample_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Read the sample image to get the size
    sample_img = cv2.imread(str(sample_path))
    height, width, _ = sample_img.shape

    # Remove the sample image
    os.remove(sample_path)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Temporary codec for .avi file
    tmpout = str(outpath / "tmp.avi")
    out = cv2.VideoWriter(tmpout, fourcc, fps, (width, height))

    if not out.isOpened():
        print("Error: Could not open video writer.")
        return

    for frame in data:
        # Create a heatmap using matplotlib
        plt.imshow(frame, cmap='hot', interpolation='nearest')
        plt.axis('off')
        
        # Add colorbar
        plt.colorbar()
        
        # Convert the plot to an image
        plt.savefig(outpath / 'temp_heatmap.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Read the image with OpenCV
        heatmap_img = cv2.imread(str(outpath / 'temp_heatmap.png'))
        
        out.write(heatmap_img)

    out.release()

    # Re-encode the video using ffmpeg
    ffmpeg_command = [
        'ffmpeg', '-y', '-i', tmpout, 
        '-pix_fmt', 'yuv420p', '-vcodec', 'libx264', 
        '-crf', '23', '-preset', 'medium', str(outpath / outname)
    ]    
    subprocess.run(ffmpeg_command)
    # Remove the temporary file
    os.remove(tmpout)
    os.remove(outpath / 'temp_heatmap.png')


def calculate_height_from_gradients(gradients_batch, height, width):
    """
    Calculate height from gradients using Poisson solver.

    Parameters:
    gradients_batch (list): A list of numpy arrays containing the gradients for each image.
    height (int): The height of the images.
    width (int): The width of the images.

    Returns:
    list: A list of 2D numpy arrays representing the height maps for each frame.
    """
    height_maps = []

    for gradients in gradients_batch:
        Gx_map = np.zeros((height, width))
        Gy_map = np.zeros((height, width))
        for gradient in gradients:
            _, _, _, x, y, Gx, Gy = gradient
            Gx_map[int(y), int(x)] = Gx
            Gy_map[int(y), int(x)] = Gy

        # Create the Poisson matrix
        A = scipy.sparse.diags([1, 1, -4, 1, 1], [-width, -1, 0, 1, width], shape=(height * width, height * width))
        A = A.tocsc()

        # Create the divergence of the gradient field
        div = np.zeros((height, width))
        div[1:-1, 1:-1] = (Gx_map[1:-1, 1:-1] - Gx_map[1:-1, :-2]) + (Gy_map[1:-1, 1:-1] - Gy_map[:-2, 1:-1])
        div = div.ravel()

        # Solve the Poisson equation
        height_map = scipy.sparse.linalg.spsolve(A, div)
        height_map = height_map.reshape((height, width))

        height_maps.append(height_map)

    return np.array(height_maps)


def format_heightmap(height_maps):
    """
    Convert height maps from [frame x height x width] to [frame x pixels].

    Parameters:
    height_maps (numpy.ndarray): The height maps in [frame x height x width] format.

    Returns:
    numpy.ndarray: The height maps in [frame x pixels] format, where each pixel is represented as [pixel_x, pixel_y, height].
    """
    frames, height, width = height_maps.shape
    converted_height_maps = []

    for frame in range(frames):
        frame_data = []
        for y in range(height):
            for x in range(width):
                frame_data.append([int(x), int(y), height_maps[frame, y, x]])
        converted_height_maps.append(frame_data)

    return np.array(converted_height_maps)


def plot_height_map_3d(height_map, is_view3d=False, outpath=Path(__file__).parent, filename="height_map.png"):
    """
    Plot a 3D height map with lighting effects.

    Parameters:
    height_map (numpy.ndarray): The 2D height map to plot.
    """
    # Flip the height map along the x-axis
    height_map = np.flip(height_map, axis=1)
    
    height, width = height_map.shape
    X, Y = np.meshgrid(np.arange(width), np.arange(height))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize height_map for color mapping
    norm_height_map = (height_map - np.min(height_map)) / (np.max(height_map) - np.min(height_map))
    colors = plt.cm.coolwarm(norm_height_map)  # Use coolwarm colormap for red-blue effect
    
    # Plot with lighting effects
    surf = ax.plot_surface(X, Y, height_map, rstride=1, cstride=1, facecolors=colors, shade=True)
    
    ax.view_init(elev=45, azim=45)  # Adjust the view angle for better lighting effect

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')

    # Set the aspect ratio to be equal
    ax.set_box_aspect([width, height, np.ptp(height_map)])  # Aspect ratio is 1:1:1


    if not os.path.exists(outpath):
        os.makedirs(outpath)
    plt.savefig(outpath / filename)
    if is_view3d:
        plt.show()
    plt.close(fig)

    
def load_yaml_to_dict(filepath):
    """
    指定されたYAMLファイルを読み込み、辞書として返します。
    :param filepath: YAMLファイルのパス
    :return: ファイルの内容を含む辞書
    """
    import yaml
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    return data



def load_and_split_csv(filepath, train_rate=0.8):
    """
    指定されたパスのCSVファイルを読み込み、データをtrain_rate対(1-train_rate)に分割する。

    :param filepath: CSVファイルのパス
    :param train_rate: トレーニングデータの比率 (0 < train_rate < 1)
    :return: (train_data, test_data) のタプル。各データは (R, G, B, x, y) と (Gx, Gy) の値を含む。
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # CSVファイルを読み込む
    data = pd.read_csv(filepath)
    x=data[['R', 'G', 'B', 'x', 'y']].values
    y = data[['Gx', 'Gy']].values
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=(1 - train_rate))
    
    return (X_train, y_train), (X_test, y_test)
    

def normalize(data,px_size=[240,240]):
    """
    データの正規化
    :param data : [batch x [R,G,B,px,py] ]
    :param px_size: [px_max, py_max]
    """

    norm=np.array([255,255,255,px_size[0],px_size[1]])

    return data/norm


def load_json_to_dict(filepath):
    """
    指定されたJSONファイルを読み込み、辞書として返します。

    :param filepath: JSONファイルのパス
    :return: ファイルの内容を含む辞書
    """
    import json
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


def plot_history(json_path, save_path):
    import json
    import matplotlib.pyplot as plt
    from pathlib import Path
    import seaborn as sns

    # JSONファイルの読み込み
    with open(json_path, 'r') as f:
        history = json.load(f)

    # Seabornのスタイルを設定
    sns.set(style="whitegrid")

    # 損失と平均絶対誤のグラフを描画
    plt.figure(figsize=(14, 12))

    # 損失のグラフ
    plt.subplot(2, 1, 1)
    sns.lineplot(data=history['loss'], label='Training Loss', color='blue')
    sns.lineplot(data=history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend()

    # 平均絶対誤差のグラフ
    plt.subplot(2, 1, 2)
    sns.lineplot(data=history['mean_absolute_error'], label='Training MAE', color='blue')
    sns.lineplot(data=history['val_mean_absolute_error'], label='Validation MAE', color='orange')
    plt.title('Mean Absolute Error', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Mean Absolute Error', fontsize=14)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def image2rgbpx(image):
    """
    Convert an image from [h x w x c] format to [R, G, B, pixel_x, pixel_y] format.

    Parameters:
    image (numpy.ndarray): The input image in [h x w x c] format.

    Returns:
    numpy.ndarray: The converted data in [R, G, B, pixel_x, pixel_y] format.
    """
    height, width, channels = image.shape
    assert channels == 3, "The input image must have 3 channels (RGB)."

    pixel_data = []

    for y in range(height):
        for x in range(width):
            R, G, B = image[y, x]
            pixel_data.append([R, G, B, x, y])

    return np.array(pixel_data)


def calc_height(Gx, Gy, height, width):
    """
    Calculate height from gradients using Poisson solver.

    Parameters:
    Gx (numpy.ndarray): Gradient in x direction.
    Gy (numpy.ndarray): Gradient in y direction.
    height (int): The height of the images.
    width (int): The width of the images.

    Returns:
    numpy.ndarray: The height map.
    """
    print(Gx.shape)
    Gx_map = Gx.reshape((height, width))
    Gy_map = Gy.reshape((height, width))

    # Create the Poisson matrix
    A = scipy.sparse.diags([1, 1, -4, 1, 1], [-width, -1, 0, 1, width], shape=(height * width, height * width))
    A = A.tocsc()

    # Create the divergence of the gradient field
    div = np.zeros((height, width))
    div[1:-1, 1:-1] = (Gx_map[1:-1, 1:-1] - Gx_map[1:-1, :-2]) + (Gy_map[1:-1, 1:-1] - Gy_map[:-2, 1:-1])
    div = div.ravel()

    # Solve the Poisson equation using Conjugate Gradient method
    height_map, info = scipy.sparse.linalg.cg(A, div)
    if info != 0:
        raise RuntimeError(f"Conjugate Gradient method did not converge, info: {info}")
    
    height_map = height_map.reshape((height, width))

    return height_map

import time

def img2height(img, model, px_size):
    start_time = time.time()
    
    print(f"img shape : {img.shape}")
    img_px = image2rgbpx(img)
    print(f"image2rgbpx shape : {img_px.shape}")
    print(f"image2rgbpx took {time.time() - start_time:.4f} seconds")
    
    start_time = time.time()
    img_px = normalize(img_px, px_size)
    print(f"normalize took {time.time() - start_time:.4f} seconds")
    
    start_time = time.time()
    grad = model.predict(img_px, batch_size=img_px.shape[0])
    print(f"model.predict took {time.time() - start_time:.4f} seconds")
    
    start_time = time.time()
    height = calc_height(grad[:, 0], grad[:, 1], px_size[0], px_size[1])
    print(f"calc_height took {time.time() - start_time:.4f} seconds")
    
    return height
