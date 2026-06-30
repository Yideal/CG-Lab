import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

N = 20
mass = 1.0
dt = 5e-4
k_s = 10000.0
k_d = 1.0
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0

width, height = 600, 600

sphere_center = ti.Vector.field(3, dtype=float, shape=())
sphere_radius = ti.field(dtype=float, shape=())

x = ti.Vector.field(3, dtype=float, shape=N * N)
v = ti.Vector.field(3, dtype=float, shape=N * N)
f = ti.Vector.field(3, dtype=float, shape=N * N)
is_fixed = ti.field(dtype=int, shape=N * N)

x_next = ti.Vector.field(3, dtype=float, shape=N * N)
v_next = ti.Vector.field(3, dtype=float, shape=N * N)
f_next = ti.Vector.field(3, dtype=float, shape=N * N)

max_springs = N * N * 4
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_lengths = ti.field(dtype=float, shape=max_springs)
num_springs = ti.field(dtype=int, shape=())


@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0
    sphere_center[None] = ti.Vector([0.0, -0.2, 0.0])
    sphere_radius[None] = 0.4


@ti.kernel
def init_springs():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        if i < N - 1:
            idx_right = (i + 1) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_right])
            spring_lengths[c] = (x[idx] - x[idx_right]).norm()
        if j < N - 1:
            idx_down = i * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_down])
            spring_lengths[c] = (x[idx] - x[idx_down]).norm()


def init_cloth():
    num_springs[None] = 0
    init_positions()
    init_springs()


@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template()):
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]
    for i in range(num_springs[None]):
        idx_a = spring_pairs[i][0]
        idx_b = spring_pairs[i][1]
        pos_a = pos[idx_a]
        pos_b = pos[idx_b]
        d = pos_a - pos_b
        dist = d.norm()
        if dist > 1e-6:
            d_normalized = d / dist
            f_spring = -k_s * (dist - spring_lengths[i]) * d_normalized
            ti.atomic_add(force[idx_a], f_spring)
            ti.atomic_add(force[idx_b], -f_spring)


@ti.func
def clamp_velocity(vel: ti.template(), idx: int):
    vel_norm = vel[idx].norm()
    if vel_norm > max_velocity:
        vel[idx] = vel[idx] / vel_norm * max_velocity


@ti.func
def handle_sphere_collision(pos: ti.template(), vel: ti.template(), idx: int):
    center = sphere_center[None]
    radius = sphere_radius[None]
    p = pos[idx]
    d = p - center
    dist = d.norm()
    
    if dist < radius:
        normal = d / dist
        pos[idx] = center + normal * (radius + 1e-4)
        vel[idx] = vel[idx] - vel[idx].dot(normal) * normal * 0.8


@ti.kernel
def step_explicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            handle_sphere_collision(x, v, i)


@ti.kernel
def step_semi_implicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt
            handle_sphere_collision(x, v, i)


@ti.kernel
def step_implicit_iter():
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]
    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)
        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt
    for i in range(N * N):
        v[i] = v_next[i]
        x[i] = x_next[i]
        if is_fixed[i] == 0:
            handle_sphere_collision(x, v, i)


def project_to_screen(px, py, pz):
    scale = 1.0 / (1.0 - pz * 0.5)
    nx = px * scale + 0.5
    ny = 1.0 - (py + 1.0) / 2.0
    return nx, ny


def draw_point(img, x, y, radius, color):
    x_pixel = int(x * width)
    y_pixel = int(y * height)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            nx = x_pixel + dx
            ny = y_pixel + dy
            if 0 <= nx < width and 0 <= ny < height:
                dist = (dx * dx + dy * dy) ** 0.5
                if dist <= radius:
                    img[ny, nx] = color


def draw_line(img, x1, y1, x2, y2, color):
    x1_pixel = int(x1 * width)
    y1_pixel = int(y1 * height)
    x2_pixel = int(x2 * width)
    y2_pixel = int(y2 * height)
    
    dx = abs(x2_pixel - x1_pixel)
    dy = abs(y2_pixel - y1_pixel)
    sx = 1 if x1_pixel < x2_pixel else -1
    sy = 1 if y1_pixel < y2_pixel else -1
    err = dx - dy
    
    while True:
        if 0 <= x1_pixel < width and 0 <= y1_pixel < height:
            img[y1_pixel, x1_pixel] = color
        if x1_pixel == x2_pixel and y1_pixel == y2_pixel:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1_pixel += sx
        if e2 < dx:
            err += dx
            y1_pixel += sy


def draw_circle(img, cx, cy, radius, color):
    cx_pixel = int(cx * width)
    cy_pixel = int(cy * height)
    r_pixel = int(radius * width)
    for angle in np.linspace(0, 2 * np.pi, 120):
        sx = cx_pixel + int(r_pixel * np.cos(angle))
        sy = cy_pixel + int(r_pixel * np.sin(angle))
        if 0 <= sx < width and 0 <= sy < height:
            img[sy, sx] = color


def main():
    init_cloth()
    gui = ti.GUI("Mass-Spring System (Collision)", (width, height))
    
    frame_count = 0
    paused = False
    current_method = 1
    
    method_names = ["Explicit Euler", "Semi-Implicit Euler", "Implicit Euler"]
    
    btn_explicit = gui.button("Explicit Euler")
    btn_semi = gui.button("Semi-Implicit Euler")
    btn_implicit = gui.button("Implicit Euler")
    btn_pause = gui.button("Pause/Resume")
    btn_reset = gui.button("Reset")
    
    prev_btn_explicit = False
    prev_btn_semi = False
    prev_btn_implicit = False
    prev_btn_pause = False
    prev_btn_reset = False
    
    print("Running... Use buttons or keys: 0/1/2 - Method, Space - Pause, R - Reset. Press ESC to exit")
    
    while gui.running:
        image = np.ones((height, width, 3), dtype=np.float32) * 0.1
        
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                gui.running = False
            elif e.key == '0':
                current_method = 0
                init_cloth()
            elif e.key == '1':
                current_method = 1
                init_cloth()
            elif e.key == '2':
                current_method = 2
                init_cloth()
            elif e.key == ' ':
                paused = not paused
            elif e.key == 'r':
                init_cloth()
        
        curr_btn_explicit = gui.is_pressed(btn_explicit)
        curr_btn_semi = gui.is_pressed(btn_semi)
        curr_btn_implicit = gui.is_pressed(btn_implicit)
        curr_btn_pause = gui.is_pressed(btn_pause)
        curr_btn_reset = gui.is_pressed(btn_reset)
        
        if curr_btn_explicit and not prev_btn_explicit:
            current_method = 0
            init_cloth()
        if curr_btn_semi and not prev_btn_semi:
            current_method = 1
            init_cloth()
        if curr_btn_implicit and not prev_btn_implicit:
            current_method = 2
            init_cloth()
        if curr_btn_pause and not prev_btn_pause:
            paused = not paused
        if curr_btn_reset and not prev_btn_reset:
            init_cloth()
        
        prev_btn_explicit = curr_btn_explicit
        prev_btn_semi = curr_btn_semi
        prev_btn_implicit = curr_btn_implicit
        prev_btn_pause = curr_btn_pause
        prev_btn_reset = curr_btn_reset
        
        if not paused:
            for _ in range(40):
                if current_method == 0:
                    step_explicit()
                elif current_method == 1:
                    step_semi_implicit()
                elif current_method == 2:
                    step_implicit_iter()
            frame_count += 1
        
        x_np = x.to_numpy()
        
        for i in range(num_springs[None]):
            idx_a = spring_pairs[i][0]
            idx_b = spring_pairs[i][1]
            px1, py1, pz1 = x_np[idx_a]
            px2, py2, pz2 = x_np[idx_b]
            x1, y1 = project_to_screen(px1, py1, pz1)
            x2, y2 = project_to_screen(px2, py2, pz2)
            if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
                draw_line(image, x1, y1, x2, y2, [0.5, 0.5, 0.5])
        
        center = sphere_center[None]
        cx, cy = project_to_screen(center[0], center[1], center[2])
        radius = sphere_radius[None] * 1.5
        draw_circle(image, cx, cy, radius, [1.0, 0.3, 0.3])
        
        for idx in range(N * N):
            px, py, pz = x_np[idx]
            sx, sy = project_to_screen(px, py, pz)
            if 0 <= sx <= 1 and 0 <= sy <= 1:
                if is_fixed[idx] == 1:
                    draw_point(image, sx, sy, 4, [1.0, 0.8, 0.0])
                else:
                    draw_point(image, sx, sy, 3, [0.3, 0.7, 1.0])
        
        gui.set_image(image.swapaxes(0, 1))
        gui.text(f"Method: {method_names[current_method]}", (0.02, 0.98), font_size=15, color=0xffffff)
        gui.text(f"Status: {'Paused' if paused else 'Running'}", (0.02, 0.95), font_size=12, color=0xaaaaaa)
        gui.text(f"Frame: {frame_count}", (0.02, 0.92), font_size=12, color=0xaaaaaa)
        gui.text("Keys: 0/1/2-Method, Space-Pause, R-Reset", (0.02, 0.05), font_size=10, color=0x666666)
        gui.show()
    
    print("Exited")


if __name__ == '__main__':
    main()