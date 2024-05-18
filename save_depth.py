import pyrealsense2 as rs
import numpy as np
import cv2
import time

# パイプラインをセットアップ
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# ストリーミング開始
pipeline.start(config)

# 深度データを保存するリスト
depth_values = []

try:
    for i in range(100):
        # フレームを取得
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue
        
        # Depthフレームを取得してNumpy配列に変換
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # 画像の中央ピクセルの深度を取得
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        center_depth = depth_frame.get_distance(width // 2, height // 2)
        
        # 深度値をリストに追加
        depth_values.append(center_depth)
        
        # デバッグ用に値を出力
        print(f"Frame {i + 1}: Depth at center: {center_depth:.4f} meters")
        
        # Depth画像を表示
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('RealSense Depth Stream', depth_colormap)
        
        # qキーが押されたら終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 少し待機
        time.sleep(0.1)

finally:
    # ストリーミングを停止
    pipeline.stop()

    # OpenCVのウィンドウを閉じる
    cv2.destroyAllWindows()

    # 深度データをテキストファイルに保存
    with open("depth_values.txt", "w") as f:
        for depth in depth_values:
            f.write(f"{depth}\n")

    print("Depth data saved to depth_values.txt")
