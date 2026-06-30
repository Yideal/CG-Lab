import taichi as ti
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, 
                             QSlider, QGroupBox, QHBoxLayout)
from PyQt5.QtCore import Qt, QTimer

ti.init(arch=ti.cpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

Ka = ti.field(ti.f32, shape=())
Kd = ti.field(ti.f32, shape=())
Ks = ti.field(ti.f32, shape=())
shininess = ti.field(ti.f32, shape=())

Ka[None] = 0.2
Kd[None] = 0.7
Ks[None] = 0.5
shininess[None] = 32.0


@ti.func
def normalize(v):
    return v / v.norm(1e-5)


@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N


@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        t2 = (-b + ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
        elif t2 > 0:
            t = t2
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal


@ti.func
def intersect_cone(ro, rd, apex, base_y, radius):
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    H = apex.y - base_y
    k = (radius / H) ** 2

    ro_local = ro - apex

    A = rd.x**2 + rd.z**2 - k * rd.y**2
    B = 2.0 * (ro_local.x * rd.x + ro_local.z * rd.z - k * ro_local.y * rd.y)
    C = ro_local.x**2 + ro_local.z**2 - k * ro_local.y**2

    if ti.abs(A) > 1e-5:
        delta = B**2 - 4.0 * A * C
        if delta > 0:
            t1 = (-B - ti.sqrt(delta)) / (2.0 * A)
            t2 = (-B + ti.sqrt(delta)) / (2.0 * A)

            t_first = t1
            t_second = t2
            if t1 > t2:
                t_first, t_second = t_second, t_first

            y1 = ro_local.y + t_first * rd.y
            if t_first > 0 and -H <= y1 <= 0:
                t = t_first
            else:
                y2 = ro_local.y + t_second * rd.y
                if t_second > 0 and -H <= y2 <= 0:
                    t = t_second

            if t > 0:
                p_local = ro_local + rd * t
                normal = normalize(ti.Vector([p_local.x, -k * p_local.y, p_local.z]))

    return t, normal


@ti.kernel
def render():
    for i, j in pixels:
        u = (i - res_x / 2.0) / res_y * 2.0
        v = (j - res_y / 2.0) / res_y * 2.0

        ro = ti.Vector([0.0, 0.0, 5.0])
        rd = normalize(ti.Vector([u, v, -1.0]))

        min_t = 1e10
        hit_normal = ti.Vector([0.0, 0.0, 0.0])
        hit_color = ti.Vector([0.0, 0.0, 0.0])

        t_sph, n_sph = intersect_sphere(ro, rd, ti.Vector([-1.2, -0.2, 0.0]), 1.2)
        if 0 < t_sph < min_t:
            min_t = t_sph
            hit_normal = n_sph
            hit_color = ti.Vector([0.8, 0.1, 0.1])

        t_cone, n_cone = intersect_cone(ro, rd, ti.Vector([1.2, 1.2, 0.0]), -1.4, 1.2)
        if 0 < t_cone < min_t:
            min_t = t_cone
            hit_normal = n_cone
            hit_color = ti.Vector([0.6, 0.2, 0.8])

        color = ti.Vector([0.05, 0.15, 0.15])

        if min_t < 1e9:
            p = ro + rd * min_t
            N = hit_normal

            light_pos = ti.Vector([2.0, 3.0, 4.0])
            light_color = ti.Vector([1.0, 1.0, 1.0])

            L = normalize(light_pos - p)
            V = normalize(ro - p)

            ambient = Ka[None] * light_color * hit_color

            diff = ti.max(0.0, N.dot(L))
            diffuse = Kd[None] * diff * light_color * hit_color

            R = normalize(reflect(-L, N))
            spec = ti.max(0.0, R.dot(V)) ** shininess[None]
            specular = Ks[None] * spec * light_color

            color = ambient + diffuse + specular

        pixels[i, j] = ti.math.clamp(color, 0.0, 1.0)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Phong Shading Demo")
        self.setGeometry(50, 50, 1100, 650)

        layout = QHBoxLayout()

        self.render_label = QLabel()
        layout.addWidget(self.render_label)

        control_panel = QWidget()
        control_layout = QVBoxLayout()

        group_box = QGroupBox("Material Parameters")
        group_layout = QVBoxLayout()

        self.ka_label = QLabel(f"Ka (Ambient): {Ka[None]:.2f}")
        self.ka_slider = QSlider(Qt.Horizontal)
        self.ka_slider.setRange(0, 100)
        self.ka_slider.setValue(int(Ka[None] * 100))
        self.ka_slider.valueChanged.connect(self.update_ka)

        self.kd_label = QLabel(f"Kd (Diffuse): {Kd[None]:.2f}")
        self.kd_slider = QSlider(Qt.Horizontal)
        self.kd_slider.setRange(0, 100)
        self.kd_slider.setValue(int(Kd[None] * 100))
        self.kd_slider.valueChanged.connect(self.update_kd)

        self.ks_label = QLabel(f"Ks (Specular): {Ks[None]:.2f}")
        self.ks_slider = QSlider(Qt.Horizontal)
        self.ks_slider.setRange(0, 100)
        self.ks_slider.setValue(int(Ks[None] * 100))
        self.ks_slider.valueChanged.connect(self.update_ks)

        self.shininess_label = QLabel(f"Shininess: {shininess[None]:.1f}")
        self.shininess_slider = QSlider(Qt.Horizontal)
        self.shininess_slider.setRange(1, 128)
        self.shininess_slider.setValue(int(shininess[None]))
        self.shininess_slider.valueChanged.connect(self.update_shininess)

        group_layout.addWidget(self.ka_label)
        group_layout.addWidget(self.ka_slider)
        group_layout.addWidget(self.kd_label)
        group_layout.addWidget(self.kd_slider)
        group_layout.addWidget(self.ks_label)
        group_layout.addWidget(self.ks_slider)
        group_layout.addWidget(self.shininess_label)
        group_layout.addWidget(self.shininess_slider)

        group_box.setLayout(group_layout)
        control_layout.addWidget(group_box)
        control_layout.addStretch()

        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(280)
        layout.addWidget(control_panel)

        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_ka(self, value):
        Ka[None] = value / 100.0
        self.ka_label.setText(f"Ka (Ambient): {Ka[None]:.2f}")

    def update_kd(self, value):
        Kd[None] = value / 100.0
        self.kd_label.setText(f"Kd (Diffuse): {Kd[None]:.2f}")

    def update_ks(self, value):
        Ks[None] = value / 100.0
        self.ks_label.setText(f"Ks (Specular): {Ks[None]:.2f}")

    def update_shininess(self, value):
        shininess[None] = float(value)
        self.shininess_label.setText(f"Shininess: {shininess[None]:.1f}")

    def update_frame(self):
        render()
        pixels_np = pixels.to_numpy()
        
        import numpy as np
        pixels_rgb = (pixels_np * 255).astype(np.uint8)
        
        from PyQt5.QtGui import QImage, QPixmap
        qimage = QImage(pixels_rgb.data, res_x, res_y, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimage)
        self.render_label.setPixmap(pixmap)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
