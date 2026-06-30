# src/Work3/main_antialias.py - 选做内容：反走样贝塞尔曲线
import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000  # 曲线采样点数量
PIXEL_RADIUS = 1.5   # 反走样半径


def de_casteljau(points, t):
    """纯 Python 递归实现 De Casteljau 算法"""
    if len(points) == 1:
        return points[0]
    next_points = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        x = (1.0 - t) * p0[0] + t * p1[0]
        y = (1.0 - t) * p0[1] + t * p1[1]
        next_points.append([x, y])
    return de_casteljau(next_points, t)


@ti.kernel
def clear_pixels():
    """并行清空像素缓冲区"""
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])


# 像素缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# 曲线精确坐标缓冲区（浮点数）
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)


@ti.kernel
def draw_curve_antialiased(n: ti.i32):
    """GPU 反走样绘制：基于距离的颜色衰减"""
    for i in range(n):
        pt = curve_points_field[i]

        # 曲线点的像素坐标（浮点数）
        x_center = pt[0] * WIDTH
        y_center = pt[1] * HEIGHT

        # 遍历周围 3x3 像素邻域
        for dx in ti.static(range(-1, 2)):
            for dy in ti.static(range(-1, 2)):
                # 像素中心坐标
                pixel_x = int(x_center) + dx
                pixel_y = int(y_center) + dy

                # 越界检查
                if 0 <= pixel_x < WIDTH and 0 <= pixel_y < HEIGHT:
                    # 像素中心点坐标（浮点数）
                    center_x = float(pixel_x) + 0.5
                    center_y = float(pixel_y) + 0.5

                    # 计算距离（亚像素精度）
                    dist_x = center_x - x_center
                    dist_y = center_y - y_center
                    dist = ti.sqrt(dist_x * dist_x + dist_y * dist_y)

                    # 基于距离计算颜色权重（越近越亮）
                    if dist < PIXEL_RADIUS:
                        # 权重：距离为0时=1，距离>=PIXEL_RADIUS时接近0
                        weight = 1.0 - (dist / PIXEL_RADIUS)
                        weight = weight * weight  # 二次衰减，更平滑

                        # 混合颜色
                        old_color = pixels[pixel_x, pixel_y]
                        new_color = ti.Vector([0.0, weight, 0.0])  # 绿色
                        pixels[pixel_x, pixel_y] = ti.Vector([
                            old_color[0] + new_color[0] * weight,
                            old_color[1] + new_color[1] * weight,
                            old_color[2] + new_color[2] * weight
                        ])
                        # 限制最大值
                        pixels[pixel_x, pixel_y] = ti.Vector([
                            ti.min(1.0, pixels[pixel_x, pixel_y][0]),
                            ti.min(1.0, pixels[pixel_x, pixel_y][1]),
                            ti.min(1.0, pixels[pixel_x, pixel_y][2])
                        ])


def main():
    gui = ti.GUI("Bezier Curve (Anti-Aliased)", res=(WIDTH, HEIGHT))
    control_points = []

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
            elif gui.event.key == 'c' or gui.event.key == 'C':
                control_points = []
                print("Canvas cleared.")
            elif gui.event.key == ti.GUI.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = gui.get_cursor_pos()
                    control_points.append(pos)
                    print(f"Added control point: {pos}")

        clear_pixels()

        current_count = len(control_points)
        if current_count >= 2:
            curve_points_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
            for t_int in range(NUM_SEGMENTS + 1):
                t = t_int / NUM_SEGMENTS
                curve_points_np[t_int] = de_casteljau(control_points, t)

            curve_points_field.from_numpy(curve_points_np)
            draw_curve_antialiased(NUM_SEGMENTS + 1)

        gui.set_image(pixels)

        # 绘制控制点和连线
        if current_count > 0:
            for i in range(current_count):
                gui.circle(control_points[i], radius=5, color=0xFF0000)
            for i in range(current_count - 1):
                gui.line(control_points[i], control_points[i + 1], radius=1, color=0x888888)

        gui.show()


if __name__ == '__main__':
    main()
