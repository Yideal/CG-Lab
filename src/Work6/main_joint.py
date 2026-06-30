import taichi as ti
import math

ti.init(arch=ti.cpu)

res = 256

target_pixels_r = ti.field(dtype=ti.f32, shape=(res, res))
target_pixels_g = ti.field(dtype=ti.f32, shape=(res, res))
target_pixels_b = ti.field(dtype=ti.f32, shape=(res, res))
display_pixels = ti.field(dtype=ti.f32, shape=(res * 2, res, 3))

loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
light_pos = ti.Vector.field(3, dtype=ti.f32, shape=(), needs_grad=True)
diffuse_color = ti.Vector.field(3, dtype=ti.f32, shape=(), needs_grad=True)

sphere_center = ti.Vector([0.5, 0.5, 0.5])
sphere_radius = 0.3
TARGET_LIGHT = [0.8, 0.8, 0.2]
TARGET_COLOR = [0.8, 0.2, 0.2]


@ti.kernel
def generate_target():
    for i, j in target_pixels_r:
        x = (i + 0.5) / res
        y = (j + 0.5) / res
        dx = x - sphere_center[0]
        dy = y - sphere_center[1]
        dist_sq = dx**2 + dy**2

        if dist_sq < sphere_radius**2:
            dz = ti.sqrt(sphere_radius**2 - dist_sq)
            z = sphere_center[2] - dz
            p = ti.Vector([x, y, z])
            n = (p - sphere_center).normalized()
            
            target_light_vec = ti.Vector(TARGET_LIGHT)
            l_dir = (target_light_vec - p).normalized()

            dot_val = n.dot(l_dir)
            intensity = ti.max(0.0, ti.min(1.0, dot_val))
            
            target_pixels_r[i, j] = intensity * TARGET_COLOR[0]
            target_pixels_g[i, j] = intensity * TARGET_COLOR[1]
            target_pixels_b[i, j] = intensity * TARGET_COLOR[2]
        else:
            target_pixels_r[i, j] = 0.0
            target_pixels_g[i, j] = 0.0
            target_pixels_b[i, j] = 0.0


@ti.kernel
def render_and_compute_loss():
    for i, j in target_pixels_r:
        x = (i + 0.5) / res
        y = (j + 0.5) / res
        dx = x - sphere_center[0]
        dy = y - sphere_center[1]
        dist_sq = dx**2 + dy**2

        r_intensity = 0.0
        g_intensity = 0.0
        b_intensity = 0.0
        
        if dist_sq < sphere_radius**2:
            dz = ti.sqrt(sphere_radius**2 - dist_sq)
            z = sphere_center[2] - dz
            p = ti.Vector([x, y, z])
            n = (p - sphere_center).normalized()
            l_dir = (light_pos[None] - p).normalized()

            dot_val = n.dot(l_dir)
            leaky_intensity = ti.max(0.1 * dot_val, dot_val)
            
            r_intensity = leaky_intensity * diffuse_color[None][0]
            g_intensity = leaky_intensity * diffuse_color[None][1]
            b_intensity = leaky_intensity * diffuse_color[None][2]
        
        diff_r = r_intensity - target_pixels_r[i, j]
        diff_g = g_intensity - target_pixels_g[i, j]
        diff_b = b_intensity - target_pixels_b[i, j]
        loss[None] += (1.0 / (res * res * 3)) * (diff_r ** 2 + diff_g ** 2 + diff_b ** 2)
        
        display_pixels[i, j, 0] = target_pixels_r[i, j]
        display_pixels[i, j, 1] = target_pixels_g[i, j]
        display_pixels[i, j, 2] = target_pixels_b[i, j]
        
        display_pixels[i + res, j, 0] = ti.max(0.0, ti.min(1.0, r_intensity))
        display_pixels[i + res, j, 1] = ti.max(0.0, ti.min(1.0, g_intensity))
        display_pixels[i + res, j, 2] = ti.max(0.0, ti.min(1.0, b_intensity))


def main():
    generate_target()
    
    light_pos[None] = [0.2, 0.2, 0.8]
    diffuse_color[None] = [0.2, 0.8, 0.2]
    
    m_light = [0.0, 0.0, 0.0]
    v_light = [0.0, 0.0, 0.0]
    m_color = [0.0, 0.0, 0.0]
    v_color = [0.0, 0.0, 0.0]
    
    beta1 = 0.9
    beta2 = 0.999
    lr_light = 0.02
    lr_color = 0.01
    eps = 1e-8

    gui = ti.GUI("Joint Optimization (Left: Target, Right: Current)", res=(res * 2, res))

    print(f"Target Light Position: {TARGET_LIGHT}")
    print(f"Target Color: {TARGET_COLOR}")
    print(f"Initial Light Position: [{light_pos[None][0]:.3f}, {light_pos[None][1]:.3f}, {light_pos[None][2]:.3f}]")
    print(f"Initial Color: [{diffuse_color[None][0]:.3f}, {diffuse_color[None][1]:.3f}, {diffuse_color[None][2]:.3f}]")
    print("-" * 60)

    for iter in range(1, 501):
        loss[None] = 0.0
        
        with ti.ad.Tape(loss=loss):
            render_and_compute_loss()

        grad_light = light_pos.grad[None]
        grad_color = diffuse_color.grad[None]

        for c in range(3):
            m_light[c] = beta1 * m_light[c] + (1 - beta1) * grad_light[c]
            v_light[c] = beta2 * v_light[c] + (1 - beta2) * grad_light[c] * grad_light[c]
            
            m_hat_light = m_light[c] / (1 - beta1**iter)
            v_hat_light = v_light[c] / (1 - beta2**iter)
            
            light_pos[None][c] -= lr_light * m_hat_light / (math.sqrt(v_hat_light) + eps)
            
            m_color[c] = beta1 * m_color[c] + (1 - beta1) * grad_color[c]
            v_color[c] = beta2 * v_color[c] + (1 - beta2) * grad_color[c] * grad_color[c]
            
            m_hat_color = m_color[c] / (1 - beta1**iter)
            v_hat_color = v_color[c] / (1 - beta2**iter)
            
            diffuse_color[None][c] -= lr_color * m_hat_color / (math.sqrt(v_hat_color) + eps)

        diffuse_color[None][0] = max(0.0, min(1.0, diffuse_color[None][0]))
        diffuse_color[None][1] = max(0.0, min(1.0, diffuse_color[None][1]))
        diffuse_color[None][2] = max(0.0, min(1.0, diffuse_color[None][2]))

        if iter % 20 == 0:
            pos = light_pos[None]
            col = diffuse_color[None]
            print(f"Iter {iter:03d} | Loss: {loss[None]:.6f} | "
                  f"Light: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | "
                  f"Color: [{col[0]:.3f}, {col[1]:.3f}, {col[2]:.3f}]")

        gui.set_image(display_pixels)
        gui.show()


if __name__ == "__main__":
    main()