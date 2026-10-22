"""Оцінка руху камери (глобального руху кадру).

Відео знято «з руки»: камера наближається, панорамує й тремтить. Через це
в координатах кадру зміщуються *всі* об'єкти, навіть нерухомі. Щоб
визначити, який об'єкт рухається насправді, потрібно відокремити рух
камери від руху об'єкта.

Алгоритм для кожної пари сусідніх кадрів:

1. На попередньому кадрі шукаємо «добрі для відстеження» кутові точки
   (детектор Ши–Томасі, ``cv2.goodFeaturesToTrack``) — лише на фоні:
   ділянки всередині рамок об'єктів маскуються, бо об'єкти можуть рухатися.
2. Знаходимо ці точки на поточному кадрі пірамідальним оптичним потоком
   Лукаса–Канаде (``cv2.calcOpticalFlowPyrLK``).
3. За парами точок оцінюємо перетворення подібності (зсув + поворот +
   масштаб, 4 ступені свободи) методом RANSAC
   (``cv2.estimateAffinePartial2D``) — він відкидає хибні відповідності.
4. Перемножуючи перетворення від кадру до кадру, отримуємо відображення
   «поточний кадр → перший кадр». У цій спільній системі координат
   нерухомі об'єкти стоять на місці, а рухомі — ні.
"""
from __future__ import annotations

import cv2
import numpy as np


class CameraMotionEstimator:
    """Накопичує перетворення «кадр t → кадр 0» (3×3 матриці)."""

    def __init__(self, max_corners: int = 500, quality: float = 0.01, min_distance: int = 10,
                 ransac_thr: float = 1.5):
        self.max_corners = max_corners
        self.quality = quality
        self.min_distance = min_distance
        self.ransac_thr = ransac_thr
        self.prev_gray: np.ndarray | None = None
        self.prev_mask: np.ndarray | None = None
        self.to_ref = np.eye(3)            # поточний кадр → перший кадр
        self.history: list[np.ndarray] = []  # to_ref для кожного кадру
        self.inliers: list[int] = []         # скільки точок підтвердили модель

    @staticmethod
    def background_mask(shape, boxes, pad: int = 8) -> np.ndarray:
        """Маска фону: 255 скрізь, крім (розширених на ``pad``) рамок об'єктів."""
        m = np.full(shape[:2], 255, np.uint8)
        for x1, y1, x2, y2 in boxes:
            m[max(int(y1) - pad, 0):int(y2) + pad, max(int(x1) - pad, 0):int(x2) + pad] = 0
        return m

    def update(self, frame: np.ndarray, object_boxes) -> np.ndarray:
        """Додає кадр і повертає матрицю 3×3 «цей кадр → перший кадр»."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self.background_mask(gray.shape, object_boxes)

        if self.prev_gray is not None:
            pts_prev = cv2.goodFeaturesToTrack(self.prev_gray, self.max_corners, self.quality,
                                               self.min_distance, mask=self.prev_mask)
            step = np.eye(3)
            n_in = 0
            if pts_prev is not None and len(pts_prev) >= 8:
                pts_cur, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, pts_prev, None,
                                                          winSize=(21, 21), maxLevel=3)
                ok = st.ravel() == 1
                if ok.sum() >= 8:
                    # Модель «поточний → попередній»
                    A, inl = cv2.estimateAffinePartial2D(pts_cur[ok], pts_prev[ok], method=cv2.RANSAC,
                                                         ransacReprojThreshold=self.ransac_thr)
                    if A is not None:
                        step[:2] = A
                        n_in = int(inl.sum())
            self.to_ref = self.to_ref @ step
            self.inliers.append(n_in)
        else:
            self.inliers.append(0)

        self.prev_gray, self.prev_mask = gray, mask
        self.history.append(self.to_ref.copy())
        return self.to_ref

    @staticmethod
    def apply(T: np.ndarray, pts) -> np.ndarray:
        """Переводить точки (N×2) з координат кадру в координати першого кадру."""
        p = np.asarray(pts, np.float64).reshape(-1, 2)
        ph = np.hstack([p, np.ones((len(p), 1))]) @ T.T
        return ph[:, :2] / ph[:, 2:3]
