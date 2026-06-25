# src/Work2/main_cube.py - 选做内容：3D立方体与旋转插值
import taichi as ti
import math

ti.init(arch=ti.cpu)

vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=8)
screen_coords_interp = ti.Vector.field(2, dtype=ti.f32, shape=8)

edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

@ti.func
def get_model_matrix_z(angle: ti.f32):
    rad = angle * math.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)
    return ti.Matrix([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_model_matrix_y(angle: ti.f32):
    rad = angle * math.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)
    return ti.Matrix([
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_view_matrix(eye_pos):
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
    n = -zNear
    f = -zFar
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r
    
    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    M_ortho_scale = ti.Matrix([
        [2.0 / (r - l), 0.0, 0.0, 0.0],
        [0.0, 2.0 / (t - b), 0.0, 0.0],
        [0.0, 0.0, 2.0 / (n - f), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    M_ortho_trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r + l) / 2.0],
        [0.0, 1.0, 0.0, -(t + b) / 2.0],
        [0.0, 0.0, 1.0, -(n + f) / 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    M_ortho = M_ortho_scale @ M_ortho_trans
    return M_ortho @ M_p2o

@ti.kernel
def compute_transform(angle_z: ti.f32, angle_y: ti.f32, offset_x: ti.f32):
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    model_z = get_model_matrix_z(angle_z)
    model_y = get_model_matrix_y(angle_y)
    model = model_y @ model_z
    
    model_trans = ti.Matrix([
        [1.0, 0.0, 0.0, offset_x],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    model = model_trans @ model
    
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    mvp = proj @ view @ model
    
    for i in range(8):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        v_ndc = v_clip / v_clip[3]
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

@ti.kernel
def compute_transform_interp(angle_z: ti.f32, angle_y: ti.f32, t: ti.f32):
    eye_pos = ti.Vector([0.0, 0.0, 5.0])
    
    model_z_start = get_model_matrix_z(0.0)
    model_y_start = get_model_matrix_y(0.0)
    model_start = model_y_start @ model_z_start
    
    model_z_end = get_model_matrix_z(angle_z)
    model_y_end = get_model_matrix_y(angle_y)
    model_end = model_y_end @ model_z_end
    
    model = ti.Matrix.zero(ti.f32, 4, 4)
    for i in ti.static(range(4)):
        for j in ti.static(range(4)):
            model[i, j] = model_start[i, j] * (1.0 - t) + model_end[i, j] * t
    
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
    mvp = proj @ view @ model
    
    for i in range(8):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        v_ndc = v_clip / v_clip[3]
        screen_coords_interp[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords_interp[i][1] = (v_ndc[1] + 1.0) / 2.0

def main():
    vertices[0] = [-1.0, -1.0, -1.0]
    vertices[1] = [1.0, -1.0, -1.0]
    vertices[2] = [1.0, 1.0, -1.0]
    vertices[3] = [-1.0, 1.0, -1.0]
    vertices[4] = [-1.0, -1.0, 1.0]
    vertices[5] = [1.0, -1.0, 1.0]
    vertices[6] = [1.0, 1.0, 1.0]
    vertices[7] = [-1.0, 1.0, 1.0]
    
    gui = ti.GUI("3D Cube Rotation & Interpolation", res=(700, 700))
    angle_z = 0.0
    angle_y = 0.0
    interp_t = 0.0
    interp_direction = 1.0
    
    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 'a':
                angle_z += 10.0
            elif gui.event.key == 'd':
                angle_z -= 10.0
            elif gui.event.key == 'w':
                angle_y += 10.0
            elif gui.event.key == 's':
                angle_y -= 10.0
            elif gui.event.key == ti.GUI.ESCAPE:
                gui.running = False
        
        angle_z += 0.5
        angle_y += 0.3
        
        interp_t += 0.01 * interp_direction
        if interp_t >= 1.0 or interp_t <= 0.0:
            interp_direction *= -1
        
        compute_transform(0.0, 0.0, -1.5)
        for (i, j) in edges:
            gui.line(screen_coords[i], screen_coords[j], radius=2, color=0xFF0000)
        
        compute_transform(angle_z, angle_y, 1.5)
        for (i, j) in edges:
            gui.line(screen_coords[i], screen_coords[j], radius=2, color=0x00FF00)
        
        compute_transform_interp(angle_z, angle_y, interp_t)
        for (i, j) in edges:
            gui.line(screen_coords_interp[i], screen_coords_interp[j], radius=2, color=0x0000FF)
        
        gui.text(f"Z: {angle_z:.1f}°", pos=(0.05, 0.95), color=0xFFFFFF)
        gui.text(f"Y: {angle_y:.1f}°", pos=(0.05, 0.90), color=0xFFFFFF)
        gui.text(f"Interp: {interp_t:.2f}", pos=(0.05, 0.85), color=0xFFFFFF)
        
        gui.show()

if __name__ == '__main__':
    main()
