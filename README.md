# [TICV2026] Rendu 1 - Calibration de caméra et estimation de pose 3D

**Auteur :** Nguyen Ho Bao Khanh  
**Formation :** Master 2 VMI - Séquences Vidéo (Novembre 2026)

## Structure du dépôt

- `calibrationNguyenHoBaoKhanhVMI2026.pdf` : Rapport du TP (4 pages).
- `calibrate.py` : Script Python réalisant la calibration (`calibrateCamera`), la correction de distorsion (`undistort`), le calcul de la focale en mètres (`calibrationMatrixValues`) et l'estimation de pose 3D (`solvePnP`).
- `chessboard_9x6_25mm_A4.pdf` : Mire d'échiquier (grille de 9x6 coins intérieurs, cases de 25 mm) imprimée sur feuille A4.
- `image/` : Les 24 images originales acquises en résolution native (4000x2252) avec les métadonnées EXIF.
- `output/` : Images générées par le script (détection des coins, image corrigée, graphique d'erreur de reprojection, projection des axes 3D et du cube virtuel).

## Exécution

```bash
pip install numpy opencv-python pillow matplotlib
python calibrate.py
```
