# src/Work3/main_bspline.py - 选做内容：B样条曲线
import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

WIDTH = 800
HEIGHT = 800
MAX_CONTROL_POINTS = 100
NUM_SEGMENTS = 100  # 每段曲线的采样点数


def de_casteljau(points, t):
    """De Casteljau 算法 - 贝塞尔曲线"""
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


def cox_de_boor(i, k, u, knot_vector):
    """Cox-de Boor 递归公式计算 B 样条基函数"""
    if k == 1:
        if knot_vector[i] <= u < knot_vector[i + 1]:
            return 1.0
        else:
            return 0.0
    else:
        denom1 = knot_vector[i + k - 1] - knot_vector[i]
        denom2 = knot_vector[i + k] - knot_vector[i + 1]
        if denom1 == 0:
            denom1 = 1.0
        if denom2 == 0:
            denom2 = 1.0
        
        result = 0.0
        if denom1 != 0:
            result += ((u - knot_vector[i]) / denom1) * cox_de_boor(i, k - 1, u, knot_vector)
        if denom2 != 0:
            result += ((knot_vector[i + k] - u) / denom2) * cox_de_boor(i + 1, k - 1, u, knot_vector)
        return result


def compute_bspline_point(control_points, u, k, knot_vector):
    """计算 B 样条曲线上参数 u 对应的点"""
    n = len(control_points) - 1
    point = np.array([0.0, 0.0])
    
    for i in range(len(control_points)):
        basis = cox_de_boor(i, k, u, knot_vector)
        point += basis * control_points[i]
    
    return point


# 像素缓冲区
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))

# 曲线坐标的 GPU 缓冲区
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=1000)


@ti.kernel
def clear_pixels():
    """并行清空像素缓冲区"""
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def draw_curve_kernel(n: ti.i32, color_r: ti.f32, color_g: ti.f32, color_b: ti.f32):
    """GPU 并行绘制曲线"""
    for i in range(n):
        pt = curve_points_field[i]
        x_pixel = ti.cast(pt[0] * WIDTH, ti.i32)
        y_pixel = ti.cast(pt[1] * HEIGHT, ti.i32)
        if 0 <= x_pixel < WIDTH and 0 <= y_pixel < HEIGHT:
            pixels[x_pixel, y_pixel] = ti.Vector([color_r, color_g, color_b])


def main():
    gui = ti.GUI("Bezier & B-Spline Curve", res=(WIDTH, HEIGHT))
    control_points = []
    use_bspline = False  # 默认显示贝塞尔曲线

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
            elif gui.event.key == 'c' or gui.event.key == 'C':
                control_points = []
                print("Canvas cleared.")
            elif gui.event.key == 'b' or gui.event.key == 'B':
                use_bspline = not use_bspline
                if use_bspline:
                    print("Switched to B-Spline mode (blue)")
                else:
                    print("Switched to Bezier mode (green)")
            elif gui.event.key == ti.GUI.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    pos = gui.get_cursor_pos()
                    control_points.append(pos)
                    print(f"Added control point: {pos}")

        clear_pixels()

        current_count = len(control_points)
        if current_count >= 2:
            curve_points_list = []
            
            if use_bspline and current_count >= 4:
                # B 样条曲线
                k = 4  # 三次 B 样条
                n = current_count - 1
                # 均匀节点向量
                knot_vector = np.zeros(n + k + 1)
                for j in range(k):
                    knot_vector[j] = 0.0
                    knot_vector[n + j] = 1.0
                for j in range(n - k + 2):
                    knot_vector[j + k] = j / (n - k + 1)
                
                # 生成 B 样条曲线点
                t_start = knot_vector[k - 1]
                t_end = knot_vector[n]
                num_total = current_count * NUM_SEGMENTS
                
                control_np = np.array(control_points, dtype=np.float32)
                for t_int in range(num_total):
                    t = t_start + (t_end - t_start) * t_int / num_total
                    if t <= t_end:
                        point = compute_bspline_point(control_np, t, k, knot_vector)
                        curve_points_list.append(point)
                        
            else:
                # 贝塞尔曲线
                for t_int in range(NUM_SEGMENTS * current_count):
                    t = t_int / (NUM_SEGMENTS * current_count)
                    point = de_casteljau(control_points, t)
                    curve_points_list.append(point)

            # 绘制曲线
            curve_np = np.array(curve_points_list, dtype=np.float32)
            if len(curve_np) > 0 and len(curve_np) <= 1000:
                curve_points_field.from_numpy(curve_np)
                if use_bspline and current_count >= 4:
                    draw_curve_kernel(len(curve_np), 0.0, 0.0, 1.0)  # 蓝色 B样条
                else:
                    draw_curve_kernel(len(curve_np), 0.0, 1.0, 0.0)  # 绿色 贝塞尔

        gui.set_image(pixels)

        # 绘制控制点和连线
        if current_count > 0:
            for i in range(current_count):
                gui.circle(control_points[i], radius=5, color=0xFF0000)
            for i in range(current_count - 1):
                gui.line(control_points[i], control_points[i + 1], radius=1, color=0x888888)

        # 显示当前模式
        mode_text = "B-Spline (Blue)" if use_bspline else "Bezier (Green)"
        gui.text(f"Mode: {mode_text} | Press 'b' to switch | 'c' to clear", (0.0, 0.95))
        
        gui.show()


if __name__ == '__main__':
    main()
