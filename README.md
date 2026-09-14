# CS4016: Computación Gráfica - Demo de Tracking 3D y Estimación de Pose Facial (6-DoF)

> **Entrega Semana 6 - Prototipo Base de Tracking 3D y Registro Espacial**  
> Implementación limpia, autocontenida y 100% en Python puro con OpenCV y MediaPipe. No requiere compiladores C++ ni configuración de CMake.

---

## 1. Ejecución Rápida en 2 Pasos

### Paso 1: Instalar dependencias
Abre una terminal en la carpeta del proyecto y ejecuta:

```bash
pip install -r requirements.txt
```

### Paso 2: Ejecutar la aplicación
```bash
python app.py
```
*(En Windows también puedes usar `py app.py` si utilizas el Python Launcher)*.

---

## 2. Controles y Atajos de Teclado

| Tecla | Acción | Descripción |
| :---: | :---: | :--- |
| **`B`** | Alternar Caja 3D | Activa/Desactiva el cubo delimitador alámbrico (*wireframe box*) alrededor de la cabeza. |
| **`A`** | Alternar Ejes 3D | Activa/Desactiva los ejes ortogonales cartesianos anclados a la punta de la nariz. |
| **`M`** | Alternar Puntos Faciales | Muestra/Oculta los 6 puntos canónicos antropométricos detectados en 2D con sus IDs. |
| **`H`** | Alternar HUD | Muestra/Oculta la tarjeta de diagnóstico con métricas de orientación y distancia. |
| **`Q` / `ESC`** | Salir | Cierra la ventana y libera los recursos de la cámara limpiamente. |

---

## 3. Fundamento Matemático y Teórico (Para la Presentación)

### 3.1. Modelo de Cámara Estenopeica (Pinhole Camera Model)
La transformación entre un punto tridimensional en el espacio del objeto/mundo $\mathbf{P}_w = [X_w, Y_w, Z_w, 1]^T$ y su proyección en píxeles $\mathbf{p} = [u, v, 1]^T$ en el plano de la imagen se modela como:

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} [\mathbf{R} \mid \mathbf{t}] \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

Donde:
- **$\mathbf{K}$ (Matriz Intrínseca de Calibración):** Construida dinámicamente con base en las dimensiones del frame ($W \times H$):
  $$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$
  donde $f_x = f_y \approx W$ (distancia focal estimada asumiendo un campo de visión horizontal típico de $55^\circ$ a $60^\circ$) y $(c_x, c_y) = (W/2, H/2)$ como punto principal.
- **$\mathbf{R} \in SO(3)$ y $\mathbf{t} \in \mathbb{R}^3$ (Extrínsecos):** Matriz de rotación y vector de traslación que definen los 6 Grados de Libertad (6-DoF) del rostro respecto a la cámara.

### 3.2. Resolución del Problema Perspective-n-Point (PnP)
Utilizando `cv2.solvePnP` con optimización iterativa de Levenberg-Marquardt (`cv2.SOLVEPNP_ITERATIVE`), el sistema encuentra $[\mathbf{R} \mid \mathbf{t}]$ minimizando el error de reproyección cuadrático medio:

$$\arg\min_{\mathbf{R}, \mathbf{t}} \sum_{i=1}^{n} \left\| \mathbf{p}_i - \pi(\mathbf{K}, \mathbf{R}, \mathbf{t}, \mathbf{P}_i) \right\|^2$$

A partir de los **6 puntos canónicos antropométricos** (Farkas / ISO standard en mm):
1. **Punta de la Nariz (Landmark 1):** Origen local $[0, 0, 0]^T$
2. **Mentón (Landmark 199):** $[0, 110, 65]^T$
3. **Comisura Exterior Ojo Izquierdo (Landmark 33):** $[-75, -60, 65]^T$
4. **Comisura Exterior Ojo Derecho (Landmark 263):** $[75, -60, 65]^T$
5. **Comisura Izquierda Boca (Landmark 61):** $[-40, 50, 35]^T$
6. **Comisura Derecha Boca (Landmark 291):** $[40, 50, 35]^T$

### 3.3. Proyección de Primitivas 3D (`cv2.projectPoints`)
- **Ejes Coordenados 3D:**
  - **Eje X (Rojo - `(0, 0, 255)` BGR):** Dirección lateral derecha $[L, 0, 0]^T$.
  - **Eje Y (Verde - `(0, 255, 0)` BGR):** Dirección vertical superior $[0, -L, 0]^T$.
  - **Eje Z (Azul - `(255, 0, 0)` BGR):** Normal frontal hacia el observador $[0, 0, -L]^T$.
- **Caja Alámbrica 3D (Wireframe Bounding Box):** 8 vértices rígidos que delimitan el volumen cráneo-facial ($170\text{ mm} \times 240\text{ mm} \times 190\text{ mm}$), conectados por 12 aristas proyectadas en perspectiva cónica. Escala naturalmente con la distancia $Z$ y rota de forma congruente con el usuario.

### 3.4. Descomposición de Ángulos de Euler (Tait-Bryan)
El vector de rotación compacto $\mathbf{r}$ se convierte en matriz $\mathbf{R} \in SO(3)$ mediante la fórmula de Rodrigues (`cv2.Rodrigues`) y se descompone en ángulos físicos mediante factorización RQ (`cv2.RQDecomp3x3`):
- **Pitch ($X$):** Inclinación hacia arriba $(-)$ o hacia abajo $(+)$.
- **Yaw ($Y$):** Giro horizontal hacia la izquierda $(-)$ o derecha $(+)$.
- **Roll ($Z$):** Inclinación hacia el hombro izquierdo $(-)$ o derecho $(+)$.

---

## 4. Estructura de Archivos del Proyecto

```text
demo_tracking_p1/
├── app.py              # Script principal autocontenido y comentado
├── requirements.txt    # Dependencias exactas (OpenCV, MediaPipe, NumPy)
├── README.md           # Documentación técnica y guía de ejecución
└── face_landmarker.task # Modelo ligero pre-empaquetado para ejecución 100% offline
```

---

## 5. Compatibilidad
- **Python:** 3.9, 3.10, 3.11, 3.12, 3.13, 3.14.
- **Sistemas Operativos:** Windows, macOS, Linux.
- **Backends MediaPipe:** Compatible automáticamente tanto con `mp.solutions.face_mesh` como con `mediapipe.tasks.python.vision.FaceLandmarker`.
