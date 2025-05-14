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
dt = 0.000001         # 时间步长
nt = 5000          # 时间步数
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

# 寻找涡心位置的改进函数
def find_vortex_centers(psi, threshold=0.001):
    data = psi.copy()
    
    # 主涡 (最大流函数值)
    max_psi = np.max(data)
    max_loc = np.unravel_index(np.argmax(data), data.shape)
    
    # 屏蔽主涡区域
    y_mask, x_mask = np.ogrid[-15:16, -15:16]
    mask = x_mask*x_mask + y_mask*y_mask <= 10*10
    data[max(0,max_loc[0]-15):min(ny,max_loc[0]+31), 
         max(0,max_loc[1]-15):min(nx,max_loc[1]+31)][mask] = 0
    
    # 二次涡 (最小流函数值)
    if np.any(data < -threshold):
        min_psi = np.min(data)
        min_loc = np.unravel_index(np.argmin(data), data.shape)
        
        # 屏蔽二次涡区域
        data[max(0,min_loc[0]-10):min(ny,min_loc[0]+11),
             max(0,min_loc[1]-10):min(nx,min_loc[1]+11)] = 0
        
        # 三次涡 (次小流函数值)
        if np.any(data < -threshold):
            third_psi = np.min(data)
            third_loc = np.unravel_index(np.argmin(data), data.shape)
        else:
            third_psi = 0
            third_loc = (0, 0)
    else:
        min_psi = 0
        min_loc = (0, 0)
        third_psi = 0
        third_loc = (0, 0)
    
    return (max_loc, max_psi), (min_loc, min_psi), (third_loc, third_psi)

# 找到所有涡心
(main_loc, main_psi), (sec_loc, sec_psi), (third_loc, third_psi) = find_vortex_centers(psi)

# 绘制结果
plt.figure(figsize=(15, 12))

# 1. 流线图和涡心位置
plt.subplot(231)
plt.contourf(X, Y, psi, levels=20, cmap=cm.viridis)
plt.colorbar(label='Stream Function (ψ)')
plt.streamplot(X, Y, u, v, density=2, color='k', linewidth=0.7)
# 标记涡心位置
plt.scatter(x[main_loc[1]], y[main_loc[0]], c='r', s=100, label=f'Main vortex (ψ={main_psi:.2e})')
if sec_psi < -0.0001:  # 只显示显著的二次涡
    plt.scatter(x[sec_loc[1]], y[sec_loc[0]], c='b', s=80, label=f'Secondary vortex (ψ={sec_psi:.2e})')
if third_psi < -0.0001:  # 只显示显著的三次涡
    plt.scatter(x[third_loc[1]], y[third_loc[0]], c='g', s=60, label=f'Tertiary vortex (ψ={third_psi:.2e})')
plt.title('Streamlines and Vortex Centers')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# 2. 水平中线速度剖面 (u at y=0.5)
plt.subplot(232)
mid_y = ny // 2
plt.plot(x, u[mid_y, :], 'b-', linewidth=2)
plt.title('Horizontal Velocity at y=0.5')
plt.xlabel('x')
plt.ylabel('u velocity')
plt.grid(True)

# 3. 垂直中线速度剖面 (v at x=0.5)
plt.subplot(233)
mid_x = nx // 2
plt.plot(y, v[:, mid_x], 'r-', linewidth=2)
plt.title('Vertical Velocity at x=0.5')
plt.xlabel('y')
plt.ylabel('v velocity')
plt.grid(True)

# 4. 主涡附近流函数放大图
plt.subplot(234)
main_x, main_y = x[main_loc[1]], y[main_loc[0]]
plt.contourf(X, Y, psi, levels=20, cmap=cm.viridis)
plt.colorbar(label='Stream Function (ψ)')
plt.streamplot(X, Y, u, v, density=3, color='k', linewidth=0.7)
plt.xlim(main_x-0.2, main_x+0.2)
plt.ylim(main_y-0.2, main_y+0.2)
plt.title('Main Vortex Region (Zoomed)')
plt.xlabel('x')
plt.ylabel('y')

# 5. 左下角区域放大图
plt.subplot(235)
plt.contourf(X, Y, psi, levels=20, cmap=cm.viridis)
plt.colorbar(label='Stream Function (ψ)')
plt.streamplot(X, Y, u, v, density=3, color='k', linewidth=0.7)
plt.xlim(0, 0.3)
plt.ylim(0, 0.3)
plt.title('Bottom-Left Corner')
plt.xlabel('x')
plt.ylabel('y')

# 6. 右下角区域放大图
plt.subplot(236)
plt.contourf(X, Y, psi, levels=20, cmap=cm.viridis)
plt.colorbar(label='Stream Function (ψ)')
plt.streamplot(X, Y, u, v, density=3, color='k', linewidth=0.7)
plt.xlim(0.7, 1.0)
plt.ylim(0, 0.3)
plt.title('Bottom-Right Corner')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()

# 打印定量结果
print("\n关键结果汇总:")
print(f"1. 主涡中心位置: ({x[main_loc[1]]:.3f}, {y[main_loc[0]]:.3f})")
print(f"   流函数值: {main_psi:.4f}")

if sec_psi < -0.0001:
    print(f"\n2. 二次涡(左下角)位置: ({x[sec_loc[1]]:.3f}, {y[sec_loc[0]]:.3f})")
    print(f"   流函数值: {sec_psi:.4f}")
else:
    print("\n2. 未检测到显著的二次涡")

if third_psi < -0.0001:
    print(f"\n3. 三次涡(右下角)位置: ({x[third_loc[1]]:.3f}, {y[third_loc[0]]:.3f})")
    print(f"   流函数值: {third_psi:.4f}")
else:
    print("\n3. 未检测到显著的三次涡")

print("\n4. 中线速度剖面极值:")
print(f"   水平中线(y=0.5)最大u速度: {np.max(u[mid_y, :]):.4f} at x={x[np.argmax(u[mid_y, :])]:.3f}")
print(f"   垂直中线(x=0.5)最大v速度: {np.max(v[:, mid_x]):.4f} at y={y[np.argmax(v[:, mid_x])]:.3f}")