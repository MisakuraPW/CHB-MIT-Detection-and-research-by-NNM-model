import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_3d_box(ax, center, size, color, label_dims, alpha=0.8):
    """
    绘制一个伪3D长方体来表示张量
    center: (x, y) 盒子前表面的中心
    size: (width, height, depth) -> 对应 (Time, Channels, depth_visual)
    """
    x, y = center
    w, h, d = size
    
    # 颜色变体
    face_color = color
    side_color = [max(0, c - 0.2) for c in color[:3]] + [alpha] # 变暗
    top_color = [min(1, c + 0.1) for c in color[:3]] + [alpha]  # 变亮

    # 1. 前表面 (Front Face)
    rect = patches.Rectangle((x - w/2, y - h/2), w, h, linewidth=1, edgecolor='k', facecolor=face_color, alpha=alpha, zorder=10)
    ax.add_patch(rect)

    # 2. 侧表面 (Side Face - Right)
    side_poly = np.array([
        [x + w/2, y - h/2],
        [x + w/2 + d, y - h/2 + d/2],
        [x + w/2 + d, y + h/2 + d/2],
        [x + w/2, y + h/2]
    ])
    ax.add_patch(patches.Polygon(side_poly, closed=True, linewidth=1, edgecolor='k', facecolor=side_color, zorder=5))

    # 3. 顶表面 (Top Face)
    top_poly = np.array([
        [x - w/2, y + h/2],
        [x + w/2, y + h/2],
        [x + w/2 + d, y + h/2 + d/2],
        [x - w/2 + d, y + h/2 + d/2]
    ])
    ax.add_patch(patches.Polygon(top_poly, closed=True, linewidth=1, edgecolor='k', facecolor=top_color, zorder=4))

    # 添加文字标注
    # 标注通道数 (Height)
    ax.text(x - w/2 - 0.5, y, f"{label_dims[0]}", ha='right', va='center', fontsize=10, weight='bold', color='#333333')
    # 标注时间步 (Width)
    ax.text(x, y - h/2 - 0.8, f"{label_dims[1]}", ha='center', va='top', fontsize=10, weight='bold', color='#333333')

def draw_arrow(ax, start_xy, end_xy, text=""):
    """绘制连接箭头"""
    ax.annotate("", xy=end_xy, xytext=start_xy,
                arrowprops=dict(arrowstyle="->", lw=2, color="#555555"))
    mid_x = (start_xy[0] + end_xy[0]) / 2
    mid_y = (start_xy[1] + end_xy[1]) / 2
    if text:
        ax.text(mid_x, mid_y + 0.5, text, ha='center', va='bottom', fontsize=8, style='italic', backgroundcolor='white')

# --- 设置画布 ---
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 42)
ax.set_ylim(0, 20)
ax.axis('off')
ax.set_aspect('equal')

# 标题
plt.title("Architecture of SeizureCNN", fontsize=18, weight='bold', y=0.95)

# --- 定义层的数据 ---
# 格式: (Channel, Time)
# 注意：这里的 'Size' 是为了绘图美观而设定的视觉尺寸，不是真实比例
layers = [
    {'name': "Input", 'shape': (23, 2560), 'vis_size': (8, 3, 2), 'pos': (5, 10), 'color': (0.6, 0.8, 1.0)}, # 浅蓝
    {'name': "Layer 1\n(Conv+Pool)", 'shape': (32, 640), 'vis_size': (5, 4, 3), 'pos': (14, 10), 'color': (1.0, 0.8, 0.6)}, # 浅橙
    {'name': "Layer 2\n(Conv+Pool)", 'shape': (64, 320), 'vis_size': (3, 6, 4), 'pos': (22, 10), 'color': (1.0, 0.6, 0.6)}, # 浅红
    {'name': "Layer 3\n(Conv+GAP)", 'shape': (128, 1), 'vis_size': (0.5, 9, 5), 'pos': (29, 10), 'color': (0.8, 0.6, 1.0)}, # 紫色
    {'name': "FC Layers", 'shape': (64, "flat"), 'vis_size': (0.5, 6, 0), 'pos': (35, 10), 'color': (0.6, 1.0, 0.6)}, # 绿色
    {'name': "Output", 'shape': (1, ""), 'vis_size': (0.5, 0.5, 0), 'pos': (39, 10), 'color': (0.4, 0.4, 0.4)} # 灰色
]

# --- 绘制循环 ---
for i, layer in enumerate(layers):
    draw_3d_box(ax, layer['pos'], layer['vis_size'], layer['color'], layer['shape'])
    
    # 底部名称标签
    ax.text(layer['pos'][0], layer['pos'][1] - layer['vis_size'][1]/2 - 2.5, layer['name'], 
            ha='center', va='top', fontsize=11, weight='bold')

    # 绘制连接箭头
    if i < len(layers) - 1:
        start = (layer['pos'][0] + layer['vis_size'][0]/2 + 0.5, layer['pos'][1])
        next_layer = layers[i+1]
        end = (next_layer['pos'][0] - next_layer['vis_size'][0]/2 - 0.5, next_layer['pos'][1])
        
        # 定义操作名称
        op_text = ""
        if i == 0: op_text = "Conv1d(k15,s2)\nMaxPool(2)"
        elif i == 1: op_text = "Conv1d(k5,s1)\nMaxPool(2)"
        elif i == 2: op_text = "Conv1d(k3,s1)\nAdaptiveAvgPool"
        elif i == 3: op_text = "Flatten\nLinear(128->64)"
        elif i == 4: op_text = "Linear(64->1)\nSigmoid"
        
        draw_arrow(ax, start, end, op_text)

# 添加图例说明
fig.text(0.5, 0.05, "Dimensions Format: Height=Channels, Width=Time Steps", ha='center', fontsize=12, style='italic', color='gray')

plt.tight_layout()
plt.show()

# 如果你想保存图片，取消下面这行的注释
# plt.savefig('seizure_cnn_architecture.png', dpi=300, bbox_inches='tight')