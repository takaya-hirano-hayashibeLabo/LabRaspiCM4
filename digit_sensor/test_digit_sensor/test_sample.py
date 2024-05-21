# 接続してデータを描画するプログラム
# 一発でちゃんとできた！！

from digit_interface import Digit
 
d = Digit("D20982") # Unique serial number
d.connect()
d.show_view()
d.disconnect()