import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import ndimage

# 参数设置
nx, ny = 101, 101  # 网格点数
lx, ly = 1.0, 1.0  # 区域大小
dx = lx / (nx - 1)
dy = ly / (ny - 1)
nu = 0.001         # 运动粘度
dt = 0.000001     # 时间步长
nt = 5000         # 时间步数
rho = 1.0          # 密度

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

# 泊松方程求解器
def poisson2d(p, b, l2_target):
    l2_norm = 1
    while l2_norm > l2_target:
        pd = p.copy()
        p[1:-1, 1:-1] = (0.25 * (pd[1:-1, 2:] + pd[1:-1, :-2] +
                         pd[2:, 1:-1] + pd[:-2, 1:-1]) -
                         b[1:-1, 1:-1] * dx**2 / 4)
        
        # 边界条件
        p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
        p[:, -1] = p[:, -2] # dp/dx = 0 at x = lx
        p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
        p[-1, :] = 0        # p = 0 at y = ly
        
        l2_norm = np.sqrt(np.sum((p - pd)**2) / np.sum(pd**2))
    return p

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
    
    p = poisson2d(p, b, 1e-4)
    
    # 速度修正
    u[1:-1, 1:-1] = u[1:-1, 1:-1] - dt / rho * (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
    v[1:-1, 1:-1] = v[1:-1, 1:-1] - dt / rho * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)
    
    set_boundary_conditions()
    
    # 计算涡量和流函数
    omega[1:-1, 1:-1] = ((v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx) - 
                          (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy))
    
    # 求解流函数
    for _ in range(50):
        psi[1:-1, 1:-1] = 0.25 * (psi[1:-1, 2:] + psi[1:-1, :-2] + 
                                 psi[2:, 1:-1] + psi[:-2, 1:-1] + 
                                 omega[1:-1, 1:-1] * dx**2)
        
        # 边界条件
        psi[0, :] = 0    # 下边界
        psi[-1, :] = 0   # 上边界
        psi[:, 0] = 0    # 左边界
        psi[:, -1] = 0   # 右边界
    
    if n % 500 == 0:
        print(f"Step {n}/{nt} completed")

# 改进的涡心检测函数
def find_all_vortex_centers(psi, threshold=0.001):
    data = psi.copy()
    vortex_centers = []
    
    # 寻找所有局部极值点
    max_peaks = (psi > np.roll(psi, 1, axis=0)) & (psi > np.roll(psi, -1, axis=0)) & \
                (psi > np.roll(psi, 1, axis=1)) & (psi > np.roll(psi, -1, axis=1))
    min_peaks = (psi < np.roll(psi, 1, axis=0)) & (psi < np.roll(psi, -1, axis=0)) & \
                (psi < np.roll(psi, 1, axis=1)) & (psi < np.roll(psi, -1, axis=1))
    
    # 获取所有主涡(正涡量)
    max_locs = np.argwhere(max_peaks)
    for loc in max_locs:
        if psi[loc[0], loc[1]] > threshold:
            vortex_centers.append((tuple(loc), psi[loc[0], loc[1]], 'main'))
    
    # 获取所有次涡(负涡量)
    min_locs = np.argwhere(min_peaks)
    for loc in min_locs:
        if psi[loc[0], loc[1]] < -threshold:
            vortex_centers.append((tuple(loc), psi[loc[0], loc[1]], 'secondary'))
    
    # 按流函数绝对值排序
    vortex_centers.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return vortex_centers

# 找到所有涡心
vortex_centers = find_all_vortex_centers(psi)

# 计算速度大小和方向
speed = np.sqrt(u**2 + v**2)
u_normalized = u / speed
v_normalized = v / speed

# 创建更大的画布
plt.figure(figsize=(20, 16))

# 1. 速度矢量图
plt.subplot(331)
plt.quiver(X[::3, ::3], Y[::3, ::3], u[::3, ::3], v[::3, ::3], scale=20, width=0.002)
for loc, psi_val, v_type in vortex_centers[:5]:
    color = 'r' if v_type == 'main' else 'b'
    plt.scatter(x[loc[1]], y[loc[0]], c=color, s=100)
plt.title('Velocity Vector Field with Vortex Centers')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, 1)
plt.ylim(0, 1)

# 2. 流线图
plt.subplot(332)
plt.streamplot(X, Y, u, v, density=2, color='k', linewidth=0.7)
for loc, psi_val, v_type in vortex_centers[:5]:
    color = 'r' if v_type == 'main' else 'b'
    plt.scatter(x[loc[1]], y[loc[0]], c=color, s=100)
plt.title('Streamlines with Vortex Centers')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, 1)
plt.ylim(0, 1)

# 3. 速度大小场
plt.subplot(333)
plt.contourf(X, Y, speed, levels=20, cmap=cm.plasma)
plt.colorbar(label='Velocity Magnitude')
for loc, psi_val, v_type in vortex_centers[:5]:
    color = 'r' if v_type == 'main' else 'b'
    plt.scatter(x[loc[1]], y[loc[0]], c=color, s=100)
plt.title('Velocity Magnitude Field with Vortex Centers')
plt.xlabel('x')
plt.ylabel('y')

# 4. 流函数场
plt.subplot(334)
plt.contourf(X, Y, psi, levels=20, cmap=cm.viridis)
plt.colorbar(label='Stream Function (ψ)')
for loc, psi_val, v_type in vortex_centers[:5]:
    color = 'r' if v_type == 'main' else 'b'
    plt.scatter(x[loc[1]], y[loc[0]], c=color, s=100)
plt.title('Stream Function Field with Vortex Centers')
plt.xlabel('x')
plt.ylabel('y')

# 5. 涡量场
plt.subplot(335)
plt.contourf(X, Y, omega, levels=20, cmap=cm.coolwarm)
plt.colorbar(label='Vorticity (ω)')
for loc, psi_val, v_type in vortex_centers[:5]:
    color = 'r' if v_type == 'main' else 'b'
    plt.scatter(x[loc[1]], y[loc[0]], c=color, s=100)
plt.title('Vorticity Field with Vortex Centers')
plt.xlabel('x')
plt.ylabel('y')

# 6. 压力场
plt.subplot(336)
plt.contourf(X, Y, p, levels=20, cmap=cm.seismic)
plt.colorbar(label='Pressure')
for loc, psi_val, v_type in vortex_centers[:5]:
    color = 'r' if v_type == 'main' else 'b'
    plt.scatter(x[loc[1]], y[loc[0]], c=color, s=100)
plt.title('Pressure Field with Vortex Centers')
plt.xlabel('x')
plt.ylabel('y')

# 7. 主涡区域放大图
if vortex_centers:
    main_loc = vortex_centers[0][0]
    main_x, main_y = x[main_loc[1]], y[main_loc[0]]
    plt.subplot(337)
    plt.contourf(X, Y, psi, levels=20, cmap=cm.viridis)
    plt.colorbar(label='Stream Function (ψ)')
    plt.streamplot(X, Y, u, v, density=3, color='k', linewidth=0.7)
    plt.scatter(main_x, main_y, c='r', s=100)
    plt.xlim(main_x-0.2, main_x+0.2)
    plt.ylim(main_y-0.2, main_y+0.2)
    plt.title('Primary Vortex Region (Zoomed)')
    plt.xlabel('x')
    plt.ylabel('y')

# 8. 左下角区域放大图
plt.subplot(338)
plt.contourf(X, Y, psi, levels=20, cmap=cm.viridis)
plt.colorbar(label='Stream Function (ψ)')
plt.streamplot(X, Y, u, v, density=3, color='k', linewidth=0.7)
plt.xlim(0, 0.2)
plt.ylim(0, 0.2)
plt.title('Bottom-Left Corner (Zoomed)')
plt.xlabel('x')
plt.ylabel('y')

# 9. 右下角区域放大图
plt.subplot(339)
plt.contourf(X, Y, psi, levels=20, cmap=cm.viridis)
plt.colorbar(label='Stream Function (ψ)')
plt.streamplot(X, Y, u, v, density=3, color='k', linewidth=0.7)
plt.xlim(0.8, 1.0)
plt.ylim(0, 0.2)
plt.title('Bottom-Right Corner (Zoomed)')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()

# 打印定量结果
print("\n关键结果汇总:")
for i, (loc, psi_val, v_type) in enumerate(vortex_centers[:5]):
    print(f"{i+1}. {'主涡' if v_type == 'main' else '次涡'}位置: ({x[loc[1]]:.3f}, {y[loc[0]]:.3f})")
    print(f"   流函数值: {psi_val:.4f}")
    print(f"   类型: {'顺时针' if psi_val > 0 else '逆时针'}\n")

print("\n速度剖面极值:")
mid_y = ny // 2
mid_x = nx // 2
print(f"   水平中线(y=0.5)最大u速度: {np.max(u[mid_y, :]):.4f} at x={x[np.argmax(u[mid_y, :])]:.3f}")
print(f"   垂直中线(x=0.5)最大v速度: {np.max(v[:, mid_x]):.4f} at y={y[np.argmax(v[:, mid_x])]:.3f}")