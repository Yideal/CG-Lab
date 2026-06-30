# src/Work3/main.py - 贝塞尔曲线
import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 1000  # 曲线采样点数量

# 像素缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# 控制点缓冲区（固定长度）
control_points_np = np.zeros((MAX_CONTROL_POINTS, 2), dtype=np.floating)
control_points_count = 0

# 曲线坐标的 GPU 缓冲区
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)


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


@ti.kernel
def draw_curve_kernel(n: ti.i32):
    """GPU 并行绘制曲线"""
    for i in range(n):
        pt = curve_points_field[i]
        x_pixel = ti.cast(pt[0] * WIDTH, ti.i32)
        y_pixel = ti.cast(pt[1] * HEIGHT, ti.i32)
        if 0 <= x_pixel < WIDTH and 0 <= y_pixel < HEIGHT:
            pixels[x_pixel, y_pixel] = ti.Vector([0.0, 1.0, 0.0])


def main():
    global control_points_np, control_points_count
    
    gui = ti.GUI("Bezier Curve", res=(WIDTH, HEIGHT))
    control_points = []

    while gui.running:
        # 处理事件
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
            elif gui.event.key == 'c' or gui.event.key == 'C':
                control_points = []
                control_points_count = 0
                print("Canvas cleared.")
            elif gui.event.key == ti.GUI.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = gui.get_cursor_pos()
                    control_points.append(pos)
                    control_points_np[len(control_points) - 1] = pos
                    control_points_count = len(control_points)
                    print(f"Added control point: {pos}")

        clear_pixels()

        current_count = len(control_points)
        if current_count >= 2:
            curve_points_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
            for t_int in range(NUM_SEGMENTS + 1):
                t = t_int / NUM_SEGMENTS
                curve_points_np[t_int] = de_casteljau(control_points, t)

            curve_points_field.from_numpy(curve_points_np)
            draw_curve_kernel(NUM_SEGMENTS + 1)

        gui.set_image(pixels)

        # 绘制控制点和连线
        if current_count > 0:
            # 使用 numpy 数组直接绘制
            points_array = np.array(control_points, dtype=np.float32)
            for i in range(current_count):
                gui.circle(control_points[i], radius=5, color=0xFF0000)
            # 绘制连线
            for i in range(current_count - 1):
                gui.line(control_points[i], control_points[i + 1], radius=1, color=0x888888)

        gui.show()


if __name__ == '__main__':
    main()
