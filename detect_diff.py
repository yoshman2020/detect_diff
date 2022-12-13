import cv2
import numpy as np
import math

# 2画像の差を検出する
PATH_A = r'testdiff\image\202109_1.png'
PATH_B = r'testdiff\image\202109_2.png'

# 拡大率
RESIZE_RATE = 2
# goodの比率
GOOD_PERCENT = 0.15
# 2値化最小値
THRESH_MIN = 50
# ヒストグラム平坦化コントラスト最大値
CLIP_LIMIT = 3.0
# ヒストグラム平坦化グリッドサイズ（大きいほど全体的に平滑化）
TILE_GRID_SIZE = 3
# オープニングサイズ
OPEN_SIZE = 3
# ブロブ最小値
MIN_AREA = 50


def imfill(im_th: np.ndarray, seed_point: tuple[int, int]) -> np.ndarray:
    """穴埋め

    Args:
        im_th (np.ndarray): 元画像
        seed_point (tuple[int, int]): 穴埋め開始位置

    Returns:
        np.ndarray: 穴埋め後画像
    """
    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, seed_point, 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    return im_out


# 画像読み込み
imgA = cv2.imread(PATH_A)
imgB = cv2.imread(PATH_B)
hA, wA = imgA.shape[:2]

# マッチング精度を上げるために拡大する
imgA_resize = cv2.resize(
    imgA, (imgA.shape[1] * RESIZE_RATE, imgA.shape[0] * RESIZE_RATE))
imgB_resize = cv2.resize(
    imgB, (imgB.shape[1] * RESIZE_RATE, imgB.shape[0] * RESIZE_RATE))

# 必要な場合RGB変換
# imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
# imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)

# 画像サイズを取得
hA_resize, wA_resize, cA_resize = imgA_resize.shape[:3]
hB_resize, wB_resize, cB_resize = imgB_resize.shape[:3]

# 特徴点を抽出
# Initiate AKAZE detector
akaze = cv2.AKAZE_create()

# find the keypoints and descriptors with AKAZE
kpA, desA = akaze.detectAndCompute(imgA_resize, None)
kpB, desB = akaze.detectAndCompute(imgB_resize, None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(desA, desB)

# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)

# store all the good matches as per Lowe's ratio test.
good = matches[:int(len(matches) * GOOD_PERCENT)]

src_pts = np.float32([kpA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kpB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 1.0)

imgB_transform_resize = cv2.warpPerspective(
    imgB_resize, M, (wA_resize, hA_resize))

# 差分
result_resize = cv2.absdiff(imgA_resize, imgB_transform_resize)
result_gray_resize = cv2.cvtColor(result_resize, cv2.COLOR_BGR2GRAY)

# サイズを戻す
result_gray = cv2.resize(result_gray_resize, (hA, wA))

# ヒストグラム平坦化
clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=(
    TILE_GRID_SIZE, TILE_GRID_SIZE))
result_smooth = clahe.apply(result_gray)
# 二値化
_, result_bin = cv2.threshold(
    result_smooth, THRESH_MIN, 255, cv2.THRESH_BINARY)

# オープニング&クロージング
kernel = np.ones((OPEN_SIZE, OPEN_SIZE), np.uint8)
result_bin = cv2.morphologyEx(result_bin, cv2.MORPH_OPEN, kernel)
result_bin = cv2.morphologyEx(result_bin, cv2.MORPH_CLOSE, kernel)

# 外側がブロブ認識されないので枠を描画
# 平行移動量
tx = math.ceil(max(10, M[0, 2]))
ty = math.ceil(max(10, M[1, 2]))

# 穴埋め
result_bin_fill = imfill(result_bin, (tx, ty))

# 反転
result_bin_inv = cv2.bitwise_not(result_bin_fill)

thickness = max(tx, ty)

# 枠描画
result_bin_inv_rect = cv2.rectangle(result_bin_inv, (0, 0), (wA, hA), [
    255, 255, 255], thickness)

# ブロブ
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = MIN_AREA
params.maxArea = 10000
params.filterByColor = False
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False
# params.thresholdStep = 1
# params.minThreshold = 0
# params.maxThreshold = 255
# params.minRepeatability = 1
# params.minDistBetweenBlobs = 0

detector = cv2.SimpleBlobDetector_create(params)
keypointsOrgAll = detector.detect(result_bin_inv_rect)

# 枠部分は除外
keypointsOrg = [p for p in keypointsOrgAll if thickness <
                p.pt[0] and thickness < p.pt[1]]
keypointsOrg = keypointsOrgAll

# ブロブ描画（円が細いので太くする）
blobs = imgA
line_width = 2
for x in range(-line_width, line_width, 1):
    for y in range(-line_width, line_width, 1):
        keypoints = [cv2.KeyPoint(p.pt[0] + x, p.pt[1] + y, p.size)
                     for p in keypointsOrg]
        blobs = cv2.drawKeypoints(blobs, keypoints, None, (0, 0, 255),
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
count = len(keypoints)
print(f'丸の個数: {count}')

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', wA, hA)
cv2.imshow('image', blobs)
cv2.waitKey(0)
cv2.destroyAllWindows()
