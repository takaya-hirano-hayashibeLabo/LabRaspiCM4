from digit_interface import Digit

import time
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from pathlib import Path
from tqdm import tqdm


d = Digit("D20982") # Unique serial number
d.connect()

"""
これで使えるfpsを確認

from digit_interface import Digit
print("Supported streams: \n {}".format(Digit.STREAMS))
"""

#>> fpsと解像度の設定 >>
digit_conf=Digit.STREAMS #ここにconfigが詰まってる
d.set_resolution(digit_conf["QVGA"])
fps=digit_conf["QVGA"]["fps"]["30fps"]
d.set_fps(fps)
#>> fpsと解像度の設定 >>


try:
    fig,ax=plt.subplots()
    frames=[]
    for _ in tqdm(range(fps*1)):

        frame=d.get_frame() #これで今のフレームのリスト[320x240x3]が取れる. すっげー簡単！！
        print(frame.shape)

        im=ax.imshow(frame)
        frames.append([im])
        time.sleep(1.0/fps)

    ani = ArtistAnimation(fig, frames, interval=1.0/fps*1000, blit=True)
    print("="*40+"encoding videos..."+"="*40)
    ani.save(Path(__file__).parent/"getframes.mp4")

finally:
    d.disconnect()