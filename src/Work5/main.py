import taichi as ti
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, 
                             QSlider, QGroupBox, QHBoxLayout)
from PyQt5.QtCore import Qt, QTimer

ti.init(arch=ti.cpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 交互参数
light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())

# 材质常量枚举
MAT_DIFFUSE = 0
MAT_MIRROR = 1

# 初始化参数
light_pos_x[None] = 2.0
light_pos_y[None] = 4.0
light_pos_z[None] = 3.0
max_bounces[None] = 3


@ti.func
def normalize(v):
    return v / v.norm(1e-5)


@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N


@ti.func
def intersect_sphere(ro, rd, center, radius):
    """球体求交，返回 (距离 t, 法线 normal)"""
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
    """水平无限大平面求交"""
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 0:
            t = t1
    return t, normal


@ti.func
def scene_intersect(ro, rd):
    """
    遍历场景，寻找最近交点。
    返回: (t, 法线 N, 颜色 color, 材质 mat_id)
    """
    min_t = 1e10
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    # 1. 检测红色漫反射球
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.5, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([1.0, 0.0, 0.0])
        hit_mat = MAT_DIFFUSE

    # 2. 检测银色镜面球
    t, n = intersect_sphere(ro, rd, ti.Vector([1.5, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.95, 0.95, 0.95])
        hit_mat = MAT_MIRROR

    # 3. 检测地板
    t, n = intersect_plane(ro, rd, -1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        # 生成棋盘格纹理
        p = ro + rd * t
        grid_scale = 2.0
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)
        # 判断坐标的奇偶性来交替颜色
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
        u = (i - res_x / 2.0) / res_y * 2.0
        v = (j - res_y / 2.0) / res_y * 2.0

        ro = ti.Vector([0.0, 1.0, 5.0])
        rd = normalize(ti.Vector([u, v - 0.2, -1.0]))

        final_color = ti.Vector([0.0, 0.0, 0.0])
        throughput = ti.Vector([1.0, 1.0, 1.0])

        # 迭代式光线追踪（代替递归）
        for bounce in range(max_bounces[None]):
            t, N, obj_color, mat_id = scene_intersect(ro, rd)

            # 如果没击中任何物体，加上背景色并结束追踪
            if t > 1e9:
                final_color += throughput * bg_color
                break

            p = ro + rd * t

            # 分支 1：镜面反射材质
            if mat_id == MAT_MIRROR:
                # 生成反射射线，注意必须要加上极其微小的法线偏移（1e-4）防止自相交！
                ro = p + N * 1e-4
                rd = normalize(reflect(rd, N))
                # 镜面吸收一部分能量 (反射率 0.8)
                throughput *= 0.8 * obj_color

            # 分支 2：漫反射材质
            elif mat_id == MAT_DIFFUSE:
                L = normalize(light_pos - p)

                # --- 硬阴影检测 ---
                # 从当前交点向光源发射暗影射线，同样需要法线偏移
                shadow_ray_orig = p + N * 1e-4
                shadow_t, _, _, _ = scene_intersect(shadow_ray_orig, L)

                # 判断：如果去光源的路上没被挡住 (或者遮挡物比光源还远)，则计算光照
                dist_to_light = (light_pos - p).norm()
                in_shadow = 0.0
                if shadow_t < dist_to_light:
                    in_shadow = 1.0

                # 简单的 Phong 光照
                ambient = 0.3 * obj_color

                direct_light = ambient

                # 如果不在阴影里，再额外加上漫反射的光
                if in_shadow == 0.0:
                    diff = ti.max(0.0, N.dot(L))
                    diffuse = 0.9 * diff * obj_color
                    direct_light += diffuse

                # 将当前点的颜色乘以积累的能量，加到最终颜色里
                final_color += throughput * direct_light

                # 漫反射表面会打散光线，Whitted 风格下主射线到此终止
                break

        # 写入像素并进行色调映射
        pixels[i, j] = ti.math.clamp(final_color, 0.0, 1.0)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Whitted-Style Ray Tracing Demo")
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

        group_layout.addWidget(self.light_x_label)
        group_layout.addWidget(self.light_x_slider)
        group_layout.addWidget(self.light_y_label)
        group_layout.addWidget(self.light_y_slider)
        group_layout.addWidget(self.light_z_label)
        group_layout.addWidget(self.light_z_slider)
        group_layout.addWidget(self.max_bounces_label)
        group_layout.addWidget(self.max_bounces_slider)

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