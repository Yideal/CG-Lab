import taichi as ti
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, 
                             QSlider, QGroupBox, QHBoxLayout)
from PyQt5.QtCore import Qt, QTimer

ti.init(arch=ti.cpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())
sample_count = ti.field(ti.i32, shape=())

MAT_DIFFUSE = 0
MAT_MIRROR = 1

light_pos_x[None] = 2.0
light_pos_y[None] = 4.0
light_pos_z[None] = 3.0
max_bounces[None] = 3
sample_count[None] = 4


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
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal


@ti.func
def intersect_plane(ro, rd, plane_y):
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 0:
            t = t1
    return t, normal


@ti.func
def scene_intersect(ro, rd):
    min_t = 1e10
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    t, n = intersect_sphere(ro, rd, ti.Vector([-1.5, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([1.0, 0.0, 0.0])
        hit_mat = MAT_DIFFUSE

    t, n = intersect_sphere(ro, rd, ti.Vector([1.5, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.95, 0.95, 0.95])
        hit_mat = MAT_MIRROR

    t, n = intersect_plane(ro, rd, -1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        p = ro + rd * t
        grid_scale = 2.0
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)
        if (ix + iz) % 2 == 0:
            hit_c = ti.Vector([0.3, 0.3, 0.3])
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8])

    return min_t, hit_n, hit_c, hit_mat


@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color = ti.Vector([0.05, 0.15, 0.2])

    for i, j in pixels:
        final_color = ti.Vector([0.0, 0.0, 0.0])
        samples = sample_count[None]

        for s in range(samples):
            u_offset = (ti.random() - 0.5) / res_y * 2.0
            v_offset = (ti.random() - 0.5) / res_y * 2.0

            u = (i - res_x / 2.0) / res_y * 2.0 + u_offset
            v = (j - res_y / 2.0) / res_y * 2.0 + v_offset

            ro = ti.Vector([0.0, 1.0, 5.0])
            rd = normalize(ti.Vector([u, v - 0.2, -1.0]))

            pixel_color = ti.Vector([0.0, 0.0, 0.0])
            throughput = ti.Vector([1.0, 1.0, 1.0])

            for bounce in range(max_bounces[None]):
                t, N, obj_color, mat_id = scene_intersect(ro, rd)

                if t > 1e9:
                    pixel_color += throughput * bg_color
                    break

                p = ro + rd * t

                if mat_id == MAT_MIRROR:
                    ro = p + N * 1e-4
                    rd = normalize(reflect(rd, N))
                    throughput *= 0.8 * obj_color

                elif mat_id == MAT_DIFFUSE:
                    L = normalize(light_pos - p)

                    shadow_ray_orig = p + N * 1e-4
                    shadow_t, _, _, _ = scene_intersect(shadow_ray_orig, L)

                    dist_to_light = (light_pos - p).norm()
                    in_shadow = 0.0
                    if shadow_t < dist_to_light:
                        in_shadow = 1.0

                    ambient = 0.3 * obj_color
                    direct_light = ambient

                    if in_shadow == 0.0:
                        diff = ti.max(0.0, N.dot(L))
                        diffuse = 0.9 * diff * obj_color
                        direct_light += diffuse

                    pixel_color += throughput * direct_light
                    break

            final_color += pixel_color

        final_color /= samples
        pixels[i, j] = ti.math.clamp(final_color, 0.0, 1.0)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Ray Tracing with Anti-Aliasing")
        self.setGeometry(50, 50, 1100, 650)

        layout = QHBoxLayout()

        self.render_label = QLabel()
        layout.addWidget(self.render_label)

        control_panel = QWidget()
        control_layout = QVBoxLayout()

        group_box = QGroupBox("Controls")
        group_layout = QVBoxLayout()

        self.light_x_label = QLabel(f"Light X: {light_pos_x[None]:.1f}")
        self.light_x_slider = QSlider(Qt.Horizontal)
        self.light_x_slider.setRange(-50, 50)
        self.light_x_slider.setValue(int(light_pos_x[None] * 10))
        self.light_x_slider.valueChanged.connect(self.update_light_x)

        self.light_y_label = QLabel(f"Light Y: {light_pos_y[None]:.1f}")
        self.light_y_slider = QSlider(Qt.Horizontal)
        self.light_y_slider.setRange(10, 80)
        self.light_y_slider.setValue(int(light_pos_y[None] * 10))
        self.light_y_slider.valueChanged.connect(self.update_light_y)

        self.light_z_label = QLabel(f"Light Z: {light_pos_z[None]:.1f}")
        self.light_z_slider = QSlider(Qt.Horizontal)
        self.light_z_slider.setRange(-50, 50)
        self.light_z_slider.setValue(int(light_pos_z[None] * 10))
        self.light_z_slider.valueChanged.connect(self.update_light_z)

        self.max_bounces_label = QLabel(f"Max Bounces: {max_bounces[None]}")
        self.max_bounces_slider = QSlider(Qt.Horizontal)
        self.max_bounces_slider.setRange(1, 5)
        self.max_bounces_slider.setValue(int(max_bounces[None]))
        self.max_bounces_slider.valueChanged.connect(self.update_max_bounces)

        self.sample_count_label = QLabel(f"Sample Count: {sample_count[None]}")
        self.sample_count_slider = QSlider(Qt.Horizontal)
        self.sample_count_slider.setRange(1, 16)
        self.sample_count_slider.setValue(int(sample_count[None]))
        self.sample_count_slider.valueChanged.connect(self.update_sample_count)

        group_layout.addWidget(self.light_x_label)
        group_layout.addWidget(self.light_x_slider)
        group_layout.addWidget(self.light_y_label)
        group_layout.addWidget(self.light_y_slider)
        group_layout.addWidget(self.light_z_label)
        group_layout.addWidget(self.light_z_slider)
        group_layout.addWidget(self.max_bounces_label)
        group_layout.addWidget(self.max_bounces_slider)
        group_layout.addWidget(self.sample_count_label)
        group_layout.addWidget(self.sample_count_slider)

        group_box.setLayout(group_layout)
        control_layout.addWidget(group_box)
        control_layout.addStretch()

        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(280)
        layout.addWidget(control_panel)

        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)

    def update_light_x(self, value):
        light_pos_x[None] = value / 10.0
        self.light_x_label.setText(f"Light X: {light_pos_x[None]:.1f}")

    def update_light_y(self, value):
        light_pos_y[None] = value / 10.0
        self.light_y_label.setText(f"Light Y: {light_pos_y[None]:.1f}")

    def update_light_z(self, value):
        light_pos_z[None] = value / 10.0
        self.light_z_label.setText(f"Light Z: {light_pos_z[None]:.1f}")

    def update_max_bounces(self, value):
        max_bounces[None] = value
        self.max_bounces_label.setText(f"Max Bounces: {max_bounces[None]}")

    def update_sample_count(self, value):
        sample_count[None] = value
        self.sample_count_label.setText(f"Sample Count: {sample_count[None]}")

    def update_frame(self):
        render()
        pixels_np = pixels.to_numpy()

        import numpy as np
        pixels_rgb = (pixels_np * 255).astype(np.uint8)
        pixels_rgb = pixels_rgb.transpose((1, 0, 2))
        pixels_rgb = np.flip(pixels_rgb, axis=0)

        from PyQt5.QtGui import QImage, QPixmap
        qimage = QImage(pixels_rgb.tobytes(), res_x, res_y, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.render_label.setPixmap(pixmap)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()