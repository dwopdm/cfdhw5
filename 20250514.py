import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 参数设置
nx, ny = 101, 101  # 网格点数
lx, ly = 1.0, 1.0  # 区域大小
dx = lx / (nx - 1)
dy = ly / (ny - 1)
nu = 0.001  # 运动粘度
dt = 0.00000001  # 时间步长
nt = 5000   # 时间步数
rho = 1.0   # 密度

# 初始化
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
psi = np.zeros((ny, nx))  # 流函数
omega = np.zeros((ny, nx))  # 涡量

# 网格
x = np.linspace(0, lx, nx)
y = np.linspace(0, ly, ny)
X, Y = np.meshgrid(x, y)

# 边界条件
def set_boundary_conditions():
    # 上边界速度
    u_top = np.sin(np.pi * x)**2
    u[-1, :] = u_top
    v[-1, :] = 0
    
    # 其他边界固定
    u[0, :] = 0  # 下边界
    v[0, :] = 0
    u[:, 0] = 0  # 左边界
    v[:, 0] = 0
    u[:, -1] = 0  # 右边界
    v[:, -1] = 0

# 泊松方程求解器 (使用松弛法)
def poisson2d(p, b, l2_target):
    l2_norm = 1
    iterations = 0
    while l2_norm > l2_target:
        pd = p.copy()
        p[1:-1, 1:-1] = (0.25 * (pd[1:-1, 2:] + pd[1:-1, :-2] +
                                 pd[2:, 1:-1] + pd[:-2, 1:-1]) -
                          b[1:-1, 1:-1] * dx**2 / 4)
        
        # 边界条件 (Neumann)
        p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
        p[:, -1] = p[:, -2] # dp/dx = 0 at x = lx
        p[0, :] = p[1, :]  # dp/dy = 0 at y = 0
        p[-1, :] = 0        # p = 0 at y = ly (参考压力)
        
        l2_norm = np.sqrt(np.sum((p - pd)**2) / np.sum(pd**2))
        iterations += 1
    return p, iterations

# 主循环
for n in range(nt):
    un = u.copy()
    vn = v.copy()
    
    set_boundary_conditions()
    
    # 计算中间速度场
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                     vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[:-2, 1:-1]) +
                     nu * dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                     nu * dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]))
    
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                     vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) +
                     nu * dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                     nu * dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1]))
    
    set_boundary_conditions()
    
    # 计算压力修正
    b = np.zeros((ny, nx))
    b[1:-1, 1:-1] = rho * (1 / dt * 
                          ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) + 
                           (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) -
                          ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx))**2 -
                          2 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)) -
                          ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy))**2)
    
    p, _ = poisson2d(p, b, 1e-4)
    
    # 速度修正
    u[1:-1, 1:-1] = u[1:-1, 1:-1] - dt / rho * (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
    v[1:-1, 1:-1] = v[1:-1, 1:-1] - dt / rho * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)
    
    set_boundary_conditions()
    
    # 计算涡量和流函数
    omega[1:-1, 1:-1] = ((v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx) - 
                         (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy))
    
    # 求解流函数 (ω-ψ公式)
    for _ in range(50):
        psi[1:-1, 1:-1] = 0.25 * (psi[1:-1, 2:] + psi[1:-1, :-2] + 
                                 psi[2:, 1:-1] + psi[:-2, 1:-1] + 
                                 omega[1:-1, 1:-1] * dx**2)
        
        # 边界条件
        psi[0, :] = 0    # 下边界
        psi[-1, :] = 0   # 上边界
        psi[:, 0] = 0    # 左边界
        psi[:, -1] = 0   # 右边界
    
    if n % 100 == 0:
        print(f"Step {n}/{nt} completed")

# 绘制结果
plt.figure(figsize=(12, 10))

# 流线图
plt.subplot(221)
plt.contourf(X, Y, psi, levels=20, cmap=cm.viridis)
plt.colorbar()
plt.streamplot(X, Y, u, v, density=2, color='k', linewidth=1)
plt.title('Streamlines and Stream Function')
plt.xlabel('x')
plt.ylabel('y')

# 涡量图
plt.subplot(222)
plt.contourf(X, Y, omega, levels=20, cmap=cm.coolwarm)
plt.colorbar()
plt.title('Vorticity')
plt.xlabel('x')
plt.ylabel('y')

# 水平中线速度剖面
plt.subplot(223)
mid_y = ny // 2
plt.plot(u[mid_y, :], y, 'b-')
plt.title('Horizontal Velocity at Mid-height')
plt.xlabel('u')
plt.ylabel('y')

# 垂直中线速度剖面
plt.subplot(224)
mid_x = nx // 2
plt.plot(x, v[:, mid_x], 'r-')
plt.title('Vertical Velocity at Mid-width')
plt.xlabel('x')
plt.ylabel('v')

plt.tight_layout()
plt.show()

# 寻找主涡中心
max_psi = np.max(psi)
min_psi = np.min(psi)
max_loc = np.unravel_index(np.argmax(psi), psi.shape)
min_loc = np.unravel_index(np.argmin(psi), psi.shape)

print(f"Main vortex center at ({x[max_loc[1]]:.3f}, {y[max_loc[0]]:.3f}) with ψ = {max_psi:.4f}")
print(f"Secondary vortex center at ({x[min_loc[1]]:.3f}, {y[min_loc[0]]:.3f}) with ψ = {min_psi:.4f}")