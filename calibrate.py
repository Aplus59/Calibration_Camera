import numpy as np
import cv2
import glob
import os
from PIL import Image, ExifTags
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('output', exist_ok=True)

# --- 1. Calibration de la camera (tutoriel OpenCV py_calibration) ---

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
# Grille 9x6 coins interieurs, taille d'une case = 25 mm
square_size = 25.0  # mm
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = sorted(glob.glob('image/*.jpg'))

for idx, fname in enumerate(images, start=1):
    img = cv2.imread(fname, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img_corners = cv2.drawChessboardCorners(img.copy(), (9, 6), corners2, ret)
        cv2.imwrite(f'output/corners_{idx}.jpg', img_corners)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("=== RESULTATS DE CALIBRATION ===")
print("Resolution :", gray.shape[::-1])
print("RMS reprojection error (px) :", ret)
print("Camera matrix (K) :\n", mtx)
print("Distortion coefficients :\n", dist)

# --- 2. Undistortion et erreur de reprojection ---

img = cv2.imread(images[0], cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
cv2.imwrite('output/calibresult_full.jpg', dst)

# crop the image
x, y, w_roi, h_roi = roi
dst_crop = dst[y:y+h_roi, x:x+w_roi]
cv2.imwrite('output/calibresult.png', dst_crop)

# Re-projection Error
mean_error = 0
per_image_rmse = []
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    norm_l2 = cv2.norm(imgpoints[i], imgpoints2.reshape(-1, 2), cv2.NORM_L2)
    error = norm_l2 / len(imgpoints2)
    rmse_i = norm_l2 / np.sqrt(len(imgpoints2))
    mean_error += error
    per_image_rmse.append(rmse_i)

print("total error (L2/N) :", mean_error / len(objpoints))
print("RMSE moyen par image (px) :", np.mean(per_image_rmse))

plt.figure(figsize=(7, 2.5))
plt.bar(range(1, len(per_image_rmse) + 1), per_image_rmse, color='#4a7c59', edgecolor='black', linewidth=0.5)
plt.axhline(ret, color='red', linestyle='--', linewidth=1.2, label=f'RMS global = {ret:.3f} px')
plt.xlabel("Indice de l'image (1 a 24)")
plt.ylabel("Erreur RMS (px)")
plt.title("Erreur de reprojection RMS par image (resolution 4000x2252)")
plt.xticks(range(1, len(per_image_rmse) + 1))
plt.legend()
plt.tight_layout()
plt.savefig('output/reproj_error.png', dpi=180)
plt.close()

# --- 3. EXIF et focale en metres avec calibrationMatrixValues() (PS 3) ---

exif_raw = Image.open(images[0]).getexif()
exif_sub = exif_raw.get_ifd(0x8769) if 0x8769 in exif_raw else {}
exif_info = {ExifTags.TAGS.get(k, k): v for k, v in {**exif_raw, **exif_sub}.items()}
print("\n=== METADONNEES EXIF ===")
for k in ['Make', 'Model', 'FocalLength', 'FocalLengthIn35mmFilm', 'FNumber', 'ExifImageWidth', 'ExifImageHeight']:
    print(f"{k} : {exif_info.get(k)}")

# Capteur ISOCELL HP2 (pixel binning -> 2.4 um) : 4000x2252 px -> 9.60 x 5.405 mm
apertureWidth = 9.60    # mm
apertureHeight = 5.405  # mm

fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(
    mtx, (w, h), apertureWidth, apertureHeight
)

print("\n=== CALIBRATION MATRIX VALUES ===")
print(f"FOV X : {fovx:.2f} deg")
print(f"FOV Y : {fovy:.2f} deg")
print(f"Focale (en mm) : {focalLength:.4f} mm")
print(f"Focale (en metres) : {focalLength / 1000.0:.6f} m")
print(f"Point principal (en mm) : ({principalPoint[0]:.4f}, {principalPoint[1]:.4f}) mm")
print(f"Aspect Ratio (fy/fx) : {aspectRatio:.4f}")

# --- 4. Pose estimation (tutoriel OpenCV py_pose) ---

def draw_axes(img, corners, imgpts):
    corner = tuple(np.int32(corners[0].ravel()))
    imgpts = np.int32(imgpts).reshape(-1, 2)
    img = cv2.line(img, corner, tuple(imgpts[0]), (255, 0, 0), 10)
    img = cv2.line(img, corner, tuple(imgpts[1]), (0, 255, 0), 10)
    img = cv2.line(img, corner, tuple(imgpts[2]), (0, 0, 255), 10)
    return img

def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 8)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 8)
    return img

axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3) * square_size
axis_cube = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                        [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]]) * square_size

print("\n=== POSE ESTIMATION (solvePnP) ===")
for idx, fname in enumerate(images, start=1):
    img = cv2.imread(fname, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret_c, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret_c == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        ret_pnp, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)

        imgpts_axes, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
        img_axes = draw_axes(img.copy(), corners2, imgpts_axes)
        cv2.imwrite(f'output/pose_axes_{idx}.jpg', img_axes)

        imgpts_cube, jac = cv2.projectPoints(axis_cube, rvec, tvec, mtx, dist)
        img_cube = draw_cube(img.copy(), corners2, imgpts_cube)
        cv2.imwrite(f'output/pose_cube_{idx}.jpg', img_cube)

        print(f"Image {idx:2d} ({os.path.basename(fname)}) -> Tz = {tvec[2,0]/10:.1f} cm | norm(T) = {np.linalg.norm(tvec)/10:.1f} cm")

# --- 5. Experience : convergence selon le nombre d'images N ---
print("\n=== CONVERGENCE SELON N ===")
for n in [3, 5, 8, 12, 16, 20, 24]:
    r_n, K_n, d_n, _, _ = cv2.calibrateCamera(objpoints[:n], imgpoints[:n], gray.shape[::-1], None, None)
    f_mm_n = K_n[0, 0] * apertureWidth / w
    print(f"N={n:2d} -> RMS={r_n:.3f} px | fx={K_n[0,0]:.1f} px | f={f_mm_n:.2f} mm")
