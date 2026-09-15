import os
import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============================================================================
# 1. MODELO 3D FACIAL CANÓNICO Y CARGADOR DE OBJ
# ============================================================================
LANDMARK_INDICES = [1, 199, 33, 263, 61, 291]

FACE_3D = np.array([
    [0.0, 0.0, 0.0],          # 1:   Punta de la nariz (Origen)
    [0.0, 110.0, 65.0],       # 199: Mentón
    [-75.0, -60.0, 65.0],     # 33:  Comisura exterior ojo izquierdo
    [75.0, -60.0, 65.0],      # 263: Comisura exterior ojo derecho
    [-40.0, 50.0, 35.0],      # 61:  Comisura izquierda de la boca
    [40.0, 50.0, 35.0]        # 291: Comisura derecha de la boca
], dtype=np.float64)

AXES_3D = np.array([
    [0.0, 0.0, 0.0],
    [60.0, 0.0, 0.0],         # X: Derecha
    [0.0, -60.0, 0.0],        # Y: Arriba
    [0.0, 0.0, -60.0]         # Z: Frontal
], dtype=np.float64)


def load_obj_wireframe(obj_path, target_width_mm=165.0):
    """Carga vertices y aristas del .obj alineando la máscara hacia el frente."""
    if not os.path.exists(obj_path):
        return None, None

    raw_vertices = []
    edges = set()

    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()[1:4]
                raw_vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                face_idx = []
                for p in parts:
                    idx = int(p.split("/")[0])
                    face_idx.append(idx - 1 if idx > 0 else len(raw_vertices) + idx)
                for i in range(len(face_idx)):
                    p1, p2 = face_idx[i], face_idx[(i + 1) % len(face_idx)]
                    edges.add(tuple(sorted((p1, p2))))

    verts = np.array(raw_vertices, dtype=np.float64)
    if len(verts) == 0:
        return None, None

    # 1. Centrar el modelo en su centro geométrico
    center = (verts.max(axis=0) + verts.min(axis=0)) / 2.0
    verts -= center

    # 2. Rotación correcta: invertimos X e Y.
    # Mantiene la máscara al derecho (visera arriba, filtros abajo)
    # y hace que mire hacia la cámara (hacia afuera).
    verts[:, 0] = -verts[:, 0]
    verts[:, 1] = -verts[:, 1]

    # 3. Escalar al ancho craneal promedio (mm)
    width = verts[:, 0].max() - verts[:, 0].min()
    scale = target_width_mm / (width if width > 0 else 1.0)
    verts *= scale

    # 4. Ajustes finos de posición respecto a la nariz (0, 0, 0)
    verts[:, 1] += 15.0   # Altura (Y): bajar hacia boca/mentón
    verts[:, 2] += 25.0   # Profundidad (Z): calzar sobre el plano del rostro

    return verts, list(edges)


# Cargar la máscara al iniciar
OBJ_PATH = "model.obj"
HELMET_VERTS, HELMET_EDGES = load_obj_wireframe(OBJ_PATH, target_width_mm=220.0)


# ============================================================================
# 2. INICIALIZACIÓN DE MEDIAPIPE FACE DETECTOR
# ============================================================================
def create_detector():
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return mp.solutions.face_mesh.FaceMesh(max_num_faces=1)

    base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )
    return vision.FaceLandmarker.create_from_options(options)


def detect_landmarks(detector, frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if hasattr(detector, "detect"):
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(mp_img)
        return res.face_landmarks[0] if res.face_landmarks else None
    else:
        res = detector.process(rgb)
        return res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None


# ============================================================================
# 3. ESTIMACIÓN DE POSE 3D (PnP)
# ============================================================================
def estimate_head_pose(pts_2d, cam_matrix, dist_coeffs):
    success, rvec, tvec = cv2.solvePnP(
        FACE_3D, pts_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    R, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(R)

    return {
        "rvec": rvec,
        "tvec": tvec,
        "pitch": float(angles[0]),
        "yaw": float(angles[1]),
        "roll": float(angles[2]),
        "dist_cm": float(tvec[2, 0]) / 10.0,
        "dist_mm": float(tvec[2, 0])
    }


# ============================================================================
# 4. PROYECCIÓN 3D (MÁSCARA Y HUD)
# ============================================================================
def draw_ar_scene(frame, rvec, tvec, cam_matrix, dist_coeffs, show_mask=True, show_axes=True):
    # Proyección del modelo 3D de la máscara
    if show_mask and HELMET_VERTS is not None:
        proj_pts, _ = cv2.projectPoints(HELMET_VERTS, rvec, tvec, cam_matrix, dist_coeffs)
        pts_2d = proj_pts.reshape(-1, 2).astype(int)
        h, w = frame.shape[:2]

        for p1, p2 in HELMET_EDGES:
            pt1, pt2 = pts_2d[p1], pts_2d[p2]
            # Descartar líneas fuera de pantalla para mantener fluidez
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h) or (0 <= pt2[0] < w and 0 <= pt2[1] < h):
                cv2.line(frame, tuple(pt1), tuple(pt2), (0, 240, 255), 1, cv2.LINE_AA)

    # Proyección de ejes ortogonales
    if show_axes:
        proj_axes, _ = cv2.projectPoints(AXES_3D, rvec, tvec, cam_matrix, dist_coeffs)
        pts = proj_axes.reshape(-1, 2).astype(int)
        origin = tuple(pts[0])
        cv2.line(frame, origin, tuple(pts[1]), (0, 0, 255), 3, cv2.LINE_AA)
        cv2.line(frame, origin, tuple(pts[2]), (0, 255, 0), 3, cv2.LINE_AA)
        cv2.line(frame, origin, tuple(pts[3]), (255, 0, 0), 3, cv2.LINE_AA)


def draw_hud(frame, pose, fps, toggles):
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (340, 210), (20, 24, 30), -1)
    cv2.rectangle(overlay, (15, 15), (340, 210), (70, 85, 105), 1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    cv2.putText(frame, "CS4016: AR-Helmet MVP", (25, 38), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:4.1f}", (245, 38), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (25, 48), (330, 48), (70, 85, 105), 1)

    if pose:
        cv2.putText(frame, f"Distancia Z: {pose['dist_cm']:5.1f} cm", (25, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Pitch: {pose['pitch']:+5.1f} deg", (25, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 190, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Yaw:   {pose['yaw']:+5.1f} deg", (25, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 255, 190), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Roll:  {pose['roll']:+5.1f} deg", (25, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 210, 150), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "BUSCANDO ROSTRO...", (25, 100), cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 150, 255), 1, cv2.LINE_AA)

    cv2.line(frame, (25, 160), (330, 160), (70, 85, 105), 1)
    m_col = (0, 255, 120) if toggles["mask"] else (120, 120, 120)
    a_col = (0, 255, 120) if toggles["axes"] else (120, 120, 120)
    cv2.putText(frame, f"[M] Mascara 3D: {'ON' if toggles['mask'] else 'OFF'}", (25, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.40, m_col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"[A] Ejes: {'ON' if toggles['axes'] else 'OFF'}", (210, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.40, a_col, 1, cv2.LINE_AA)


# ============================================================================
# 5. BUCLE PRINCIPAL
# ============================================================================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara web.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ret, sample = cap.read()
    if not ret or sample is None:
        print("[ERROR] No se reciben frames de la cámara.")
        cap.release()
        return

    h, w = sample.shape[:2]
    cam_matrix = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1.0]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    detector = create_detector()
    toggles = {"mask": True, "axes": True}
    prev_time = time.time()
    fps = 30.0

    window_title = "CS4016 - AR Helmet (Demo Tracking 3D)"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    print("[INFO] Demo iniciada con modelo 3D. Presiona 'M' para alternar máscara, 'Q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        landmarks = detect_landmarks(detector, frame)
        pose = None

        if landmarks:
            pts_2d = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in LANDMARK_INDICES], dtype=np.float64)
            pose = estimate_head_pose(pts_2d, cam_matrix, dist_coeffs)

            if pose:
                draw_ar_scene(frame, pose["rvec"], pose["tvec"], cam_matrix, dist_coeffs, toggles["mask"], toggles["axes"])

        draw_hud(frame, pose, fps, toggles)
        cv2.imshow(window_title, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            break
        elif key in [ord('m'), ord('M')]:
            toggles["mask"] = not toggles["mask"]
        elif key in [ord('a'), ord('A')]:
            toggles["axes"] = not toggles["axes"]

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
