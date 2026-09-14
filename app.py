import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============================================================================
# 1. MODELO 3D CANÓNICO Y CONSTANTES (EN MILÍMETROS)
# ============================================================================
# 6 Puntos antropométricos canónicos de MediaPipe Face Mesh:
# 1: Nariz, 199: Mentón, 33: Ojo Izq, 263: Ojo Der, 61: Boca Izq, 291: Boca Der
LANDMARK_INDICES = [1, 199, 33, 263, 61, 291]

FACE_3D = np.array([
    [0.0, 0.0, 0.0],          # 1:   Punta de la nariz (Origen)
    [0.0, 110.0, 65.0],       # 199: Mentón
    [-75.0, -60.0, 65.0],     # 33:  Comisura exterior ojo izquierdo
    [75.0, -60.0, 65.0],      # 263: Comisura exterior ojo derecho
    [-40.0, 50.0, 35.0],      # 61:  Comisura izquierda de la boca
    [40.0, 50.0, 35.0]        # 291: Comisura derecha de la boca
], dtype=np.float64)

# Ejes 3D anclados a la nariz: Origen, X (Rojo), Y (Verde), Z (Azul frontal)
AXES_3D = np.array([
    [0.0, 0.0, 0.0],
    [70.0, 0.0, 0.0],         # X: Derecha
    [0.0, -70.0, 0.0],        # Y: Arriba
    [0.0, 0.0, -70.0]         # Z: Frontal hacia el observador
], dtype=np.float64)

# Cubo Alámbrico 3D (Wireframe Bounding Box)
BOX_3D = np.array([
    # Cara frontal (cercana a la cámara)
    [-85.0, -120.0, -20.0], [85.0, -120.0, -20.0], [85.0, 120.0, -20.0], [-85.0, 120.0, -20.0],
    # Cara posterior (trasera de la cabeza)
    [-85.0, -120.0, 170.0], [85.0, -120.0, 170.0], [85.0, 120.0, 170.0], [-85.0, 120.0, 170.0]
], dtype=np.float64)

BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Cara frontal
    (4, 5), (5, 6), (6, 7), (7, 4),  # Cara posterior
    (0, 4), (1, 5), (2, 6), (3, 7)   # Conexiones de profundidad
]


# ============================================================================
# 2. INICIALIZACIÓN DE MEDIAPIPE FACE DETECTOR
# ============================================================================
def create_detector():
    """Inicializa FaceLandmarker directamente con el modelo face_landmarker.task."""
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
    """Procesa el frame y retorna los landmarks del rostro detectado."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if hasattr(detector, "detect"):
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(mp_img)
        return res.face_landmarks[0] if res.face_landmarks else None
    else:
        res = detector.process(rgb)
        return res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None


# ============================================================================
# 3. ESTIMACIÓN DE POSE 3D (PERSPECTIVE-N-POINT Y ÁNGULOS DE EULER)
# ============================================================================
def estimate_head_pose(pts_2d, cam_matrix, dist_coeffs):
    """Resuelve PnP y descompone rvec en ángulos de Euler (Pitch, Yaw, Roll)."""
    success, rvec, tvec = cv2.solvePnP(
        FACE_3D, pts_2d, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    # Matriz de rotación 3x3 y descomposición RQ para ángulos de Euler
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
# 4. RENDERIZADO DE PRIMITIVAS 3D Y HUD
# ============================================================================
def draw_3d_primitives(frame, rvec, tvec, cam_matrix, dist_coeffs, show_box=True, show_axes=True):
    """Proyecta y dibuja los ejes coordenados y el cubo alámbrico sobre el frame."""
    if show_axes:
        proj_axes, _ = cv2.projectPoints(AXES_3D, rvec, tvec, cam_matrix, dist_coeffs)
        pts = proj_axes.reshape(-1, 2).astype(int)
        origin = tuple(pts[0])
        # X: Rojo, Y: Verde, Z: Azul (en formato BGR)
        cv2.line(frame, origin, tuple(pts[1]), (0, 0, 255), 3, cv2.LINE_AA)
        cv2.line(frame, origin, tuple(pts[2]), (0, 255, 0), 3, cv2.LINE_AA)
        cv2.line(frame, origin, tuple(pts[3]), (255, 0, 0), 3, cv2.LINE_AA)
        for p, label in [(pts[1], "X"), (pts[2], "Y"), (pts[3], "Z")]:
            cv2.putText(frame, label, (p[0] + 5, p[1] + 5), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    if show_box:
        proj_box, _ = cv2.projectPoints(BOX_3D, rvec, tvec, cam_matrix, dist_coeffs)
        pts_box = proj_box.reshape(-1, 2).astype(int)
        for i, (p1, p2) in enumerate(BOX_EDGES):
            color = (0, 240, 255) if i < 4 else ((255, 200, 0) if i < 8 else (255, 255, 200))
            cv2.line(frame, tuple(pts_box[p1]), tuple(pts_box[p2]), color, 2, cv2.LINE_AA)


def draw_hud(frame, pose, fps, toggles):
    """Dibuja el panel de diagnóstico con métricas de 6-DoF y atajos."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (340, 230), (20, 24, 30), -1)
    cv2.rectangle(overlay, (15, 15), (340, 230), (70, 85, 105), 1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    cv2.putText(frame, "CS4016: Tracking 3D (6-DoF)", (25, 38), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:4.1f}", (245, 38), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (25, 48), (330, 48), (70, 85, 105), 1)

    if pose:
        cv2.putText(frame, f"Distancia Z: {pose['dist_cm']:5.1f} cm ({pose['dist_mm']:4.0f} mm)", (25, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Pitch (X): {pose['pitch']:+6.1f} deg", (25, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 190, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Yaw   (Y): {pose['yaw']:+6.1f} deg", (25, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 255, 190), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Roll  (Z): {pose['roll']:+6.1f} deg", (25, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 210, 150), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "BUSCANDO ROSTRO...", (25, 100), cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 150, 255), 1, cv2.LINE_AA)

    cv2.line(frame, (25, 160), (330, 160), (70, 85, 105), 1)
    b_col = (0, 255, 120) if toggles["box"] else (120, 120, 120)
    a_col = (0, 255, 120) if toggles["axes"] else (120, 120, 120)
    m_col = (0, 255, 120) if toggles["mesh"] else (120, 120, 120)

    cv2.putText(frame, f"[B] Caja: {'ON' if toggles['box'] else 'OFF'}", (25, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.38, b_col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"[A] Ejes: {'ON' if toggles['axes'] else 'OFF'}", (175, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.38, a_col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"[M] Puntos: {'ON' if toggles['mesh'] else 'OFF'}", (25, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.38, m_col, 1, cv2.LINE_AA)
    cv2.putText(frame, "[Q] Salir", (175, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)


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
    toggles = {"box": True, "axes": True, "mesh": False}
    prev_time = time.time()
    fps = 30.0

    window_title = "CS4016 - Demo Tracking 3D & Pose 6-DoF"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    print("[INFO] Sistema de tracking iniciado. Presiona 'Q' o ESC para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # Cálculo de FPS
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # Detección facial
        landmarks = detect_landmarks(detector, frame)
        pose = None

        if landmarks:
            pts_2d = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in LANDMARK_INDICES], dtype=np.float64)
            pose = estimate_head_pose(pts_2d, cam_matrix, dist_coeffs)

            if pose:
                draw_3d_primitives(frame, pose["rvec"], pose["tvec"], cam_matrix, dist_coeffs, toggles["box"], toggles["axes"])

            if toggles["mesh"]:
                for pt in pts_2d.astype(int):
                    cv2.circle(frame, tuple(pt), 4, (0, 255, 255), -1, cv2.LINE_AA)

        draw_hud(frame, pose, fps, toggles)
        cv2.imshow(window_title, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            break
        elif key in [ord('b'), ord('B')]:
            toggles["box"] = not toggles["box"]
        elif key in [ord('a'), ord('A')]:
            toggles["axes"] = not toggles["axes"]
        elif key in [ord('m'), ord('M')]:
            toggles["mesh"] = not toggles["mesh"]

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
