import numpy as np
import cv2

def apply_roberts_filter(data):
    """
    Robertsフィルタを適用してエッジを強調する
    :param data: [time x h x w x c]
    :return: エッジが強調されたデータ
    """
    roberts_cross_v = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_cross_h = np.array([[0, 1], [-1, 0]], dtype=np.float32)

    enhanced_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[3]):  # 各チャンネルにフィルタを適用
            vertical = cv2.filter2D(data[i, :, :, j], -1, roberts_cross_v)
            horizontal = cv2.filter2D(data[i, :, :, j], -1, roberts_cross_h)
            edge_magnitude = np.sqrt(np.square(horizontal) + np.square(vertical)).astype(np.float32)
            if edge_magnitude.size > 0:  # edge_magnitudeが空でないことを確認
                # print(f"edge_magnitude (shape: {edge_magnitude.shape}): {edge_magnitude}")
                edge_magnitude = cv2.normalize(edge_magnitude, None, 0, 255, cv2.NORM_MINMAX)
                enhanced_data[i, :, :, j] = edge_magnitude
            else:
                print(f"Empty edge_magnitude for data[{i}, :, :, {j}]")
    
    return enhanced_data


def apply_canny_edge_detector_(batch_images):
    """
    cany edge detectorを適用してエッジ抽出するフィルタ
    :param batch_images: [batch x h x w x c]
    """
    batch_size, h, w, c = batch_images.shape
    edge_images = np.zeros((batch_size, h, w,1), dtype=np.uint8)
    
    for i in range(batch_size):
        # 画像をグレースケールに変換
        gray_image = cv2.cvtColor(batch_images[i], cv2.COLOR_RGB2GRAY)

        # グレースケール画像のコントラストを上げる
        gray_image = cv2.equalizeHist(gray_image)
          
        # # ノイズを減らすためにガウシアンフィルタを適用
        gray_image = cv2.GaussianBlur(gray_image, (15,15), 2)
        
        # Cannyエッジ検出を適用
        edges = cv2.Canny(gray_image, 10,25)
        # edges=gray_image
        
        # 結果を保存 (channel次元を1として残す)
        edge_images[i, :, :, 0] = edges
        
    return edge_images


def apply_canny_edge_detector(batch_images):
    """
    メジアン処理→2値化処理→Cannyエッジ抽出の順で画像処理を行う
    :param batch_images: [batch x h x w x c]
    :return: エッジが抽出された画像
    """
    batch_size, h, w, c = batch_images.shape
    edge_images = np.zeros((batch_size, h, w, 1), dtype=np.uint8)
    
    for i in range(batch_size):
        # 画像をグレースケールに変換
        filtered_img = cv2.cvtColor(batch_images[i], cv2.COLOR_RGB2GRAY)
        
        # メジアンフィルタを適用
        filtered_img = cv2.medianBlur(filtered_img, 5)
        # コントラストを上げる
        filtered_img = cv2.equalizeHist(filtered_img)   
        
        # エッジ鮮鋭化のためにシャープニングフィルタを適用
        kernel = np.array([[0, -1, 0], 
                           [-1, 5,-1], 
                           [0, -1, 0]])
        filtered_img = cv2.filter2D(filtered_img, -1, kernel)        


        # Cannyエッジ検出を適用
        # filtered_img = cv2.Canny(filtered_img, 100, 200)
        
        # 結果を保存 (channel次元を1として残す)
        edge_images[i, :, :, 0] = filtered_img
                
    return edge_images


# def apply_canny_edge_detector_(batch_images):
#     """
#     cany edge detectorを適用してエッジ抽出するフィルタ
#     :param batch_images: [batch x h x w x c]
#     """
#     batch_size, h, w, c = batch_images.shape
#     edge_images = np.zeros((batch_size, h, w,1), dtype=np.uint8)
    
#     for i in range(batch_size):
#         filtered_img=np.zeros_like(batch_images[i])
#         for rgb_idx in range(3):

#             # # ノイズを減らすためにガウシアンフィルタを適用
#             filtered_img[:,:,rgb_idx] = cv2.GaussianBlur(
#                 batch_images[i,:,:,rgb_idx], (5, 5), 2)
            
#             # Cannyエッジ検出を適用
#             filtered_img[:,:,rgb_idx] = cv2.Canny(
#                 filtered_img[:,:,rgb_idx], 8,15)
        
#         # 結果を保存 (channel次元を1として残す)
#         edge_images[i, :, :, 0] = cv2.cvtColor(filtered_img, cv2.COLOR_RGB2GRAY)
        
#     return edge_images