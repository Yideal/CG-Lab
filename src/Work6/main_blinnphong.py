import taichi as ti
import math

ti.init(arch=ti.cpu)

res = 256

target_pixels = ti.field(dtype=ti.f32, shape=(res, res))
display_pixels = ti.field(dtype=ti.f32, shape=(res * 2, res))

loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
light_pos = ti.Vector.field(3, dtype=ti.f32, shape=(), needs_grad=True)
shininess = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

sphere_center = ti.Vector([0.5, 0.5, 0.5])
sphere_radius = 0.3
TARGET_LIGHT = [0.8, 0.8, 0.2]
TARGET_SHININESS = 50.0


@ti.kernel
def generate_target():
    for i, j in target_pixels:
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
            
            v_dir = (ti.Vector([0.5, 0.5, 1.0]) - p).normalized()
            h_dir = (l_dir + v_dir).normalized()

            diff_val = n.dot(l_dir)
            spec_val = n.dot(h_dir)
            
            diffuse = ti.max(0.0, diff_val)
            specular = ti.max(0.0, spec_val) ** TARGET_SHININESS
            
            target_pixels[i, j] = ti.max(0.0, ti.min(1.0, 0.7 * diffuse + 0.3 * specular))
        else:
            target_pixels[i, j] = 0.0


@ti.kernel
def render_and_compute_loss():
    for i, j in target_pixels:
        x = (i + 0.5) / res
        y = (j + 0.5) / res
        dx = x - sphere_center[0]
        dy = y - sphere_center[1]
        dist_sq = dx**2 + dy**2

        intensity = 0.0
        if dist_sq < sphere_radius**2:
            dz = ti.sqrt(sphere_radius**2 - dist_sq)
            z = sphere_center[2] - dz
            p = ti.Vector([x, y, z])
            n = (p - sphere_center).normalized()
            l_dir = (light_pos[None] - p).normalized()
            
            v_dir = (ti.Vector([0.5, 0.5, 1.0]) - p).normalized()
            h_dir = (l_dir + v_dir).normalized()

            diff_val = n.dot(l_dir)
            spec_val = n.dot(h_dir)
            
            leaky_diff = ti.max(0.1 * diff_val, diff_val)
            leaky_spec = ti.max(0.01 * spec_val, spec_val)
            
            diffuse = leaky_diff
            specular = ti.max(0.0, leaky_spec) ** shininess[None]
            
            intensity = 0.7 * diffuse + 0.3 * specular
        
        diff = intensity - target_pixels[i, j]
        loss[None] += (1.0 / (res * res)) * (diff ** 2)
        
        display_pixels[i, j] = target_pixels[i, j]
        display_pixels[i + res, j] = ti.max(0.0, ti.min(1.0, intensity))


def main():
    generate_target()
    
    light_pos[None] = [0.2, 0.2, 0.8]
    shininess[None] = 10.0
    
    m_light = [0.0, 0.0, 0.0]
    v_light = [0.0, 0.0, 0.0]
    m_shininess = 0.0
    v_shininess = 0.0
    
    beta1 = 0.9
    beta2 = 0.999
    lr_light = 0.02
    lr_shininess = 0.5
    eps = 1e-8

    gui = ti.GUI("Blinn-Phong Optimization (Left: Target, Right: Current)", res=(res * 2, res))

    print(f"Target Light Position: {TARGET_LIGHT}")
    print(f"Target Shininess: {TARGET_SHININESS}")
    print(f"Initial Light Position: [{light_pos[None][0]:.3f}, {light_pos[None][1]:.3f}, {light_pos[None][2]:.3f}]")
    print(f"Initial Shininess: {shininess[None]:.1f}")
    print("-" * 60)

    for iter in range(1, 501):
        loss[None] = 0.0
        
        with ti.ad.Tape(loss=loss):
            render_and_compute_loss()

        grad_light = light_pos.grad[None]
        grad_shininess = shininess.grad[None]

        for c in range(3):
            m_light[c] = beta1 * m_light[c] + (1 - beta1) * grad_light[c]
            v_light[c] = beta2 * v_light[c] + (1 - beta2) * grad_light[c] * grad_light[c]
            
            m_hat_light = m_light[c] / (1 - beta1**iter)
            v_hat_light = v_light[c] / (1 - beta2**iter)
            
            light_pos[None][c] -= lr_light * m_hat_light / (math.sqrt(v_hat_light) + eps)
        
        m_shininess = beta1 * m_shininess + (1 - beta1) * grad_shininess
        v_shininess = beta2 * v_shininess + (1 - beta2) * grad_shininess * grad_shininess
        
        m_hat_shininess = m_shininess / (1 - beta1**iter)
        v_hat_shininess = v_shininess / (1 - beta2**iter)
        
        shininess[None] -= lr_shininess * m_hat_shininess / (math.sqrt(v_hat_shininess) + eps)
        
        shininess[None] = max(1.0, min(200.0, shininess[None]))

        if iter % 20 == 0:
            pos = light_pos[None]
            shine = shininess[None]
            print(f"Iter {iter:03d} | Loss: {loss[None]:.6f} | "
                  f"Light: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | "
                  f"Shininess: {shine:.1f}")

        gui.set_image(display_pixels)
        gui.show()


if __name__ == "__main__":
    main()