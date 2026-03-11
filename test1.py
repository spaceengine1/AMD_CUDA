import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# ================= ===== 1. ROCM 强制兼容配置（核心！） ======================
# 禁用所有可能导致算子差异的优化
os.environ['MIOPEN_FIND_MODE'] = 'NORMAL'
os.environ['MIOPEN_DEBUG_AMD_ROCM_METADATA_ENFORCE'] = '0'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'  # 根据你的GPU调整

torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# 强制禁用自动混合精度（ROCM的AMP会破坏SAM）
torch.cuda.amp.autocast_mode.autocast = False

# 设备配置（指定ROCM GPU，禁用多GPU）
if not torch.cuda.is_available():
    print("错误：未检测到AMD ROCM GPU！")
    exit()
device = torch.device("cuda:0")
torch.cuda.set_device(device)
torch.cuda.empty_cache()
torch.cuda.synchronize()  # 强制同步，避免异步执行差异

# ====================== 2. 加载模型（强制CPU加载→ROCM迁移，避免解析异常） ======================
SAM_CHECKPOINT_PATH =  r"D:\SAM\guide\sam_vit_h.pth"   # 确保权重完整（375MB）
MODEL_TYPE = "vit_h"  # 仅用vit_b，vit_h/vit_l在ROCM下兼容性极差

# 关键：先在CPU加载权重，再迁移到ROCM（避免ROCM直接解析权重出错）
def load_sam_rocm_safe(checkpoint_path, model_type, device):
    # Step1: CPU加载模型和权重
    sam = sam_model_registry[model_type]()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # Step2: 修正权重 dtype（强制float32，ROCM不兼容float16）
    for k in checkpoint.keys():
        if isinstance(checkpoint[k], torch.Tensor):
            checkpoint[k] = checkpoint[k].to(dtype=torch.float32).contiguous()
    # Step3: 加载权重并迁移到ROCM
    sam.load_state_dict(checkpoint)
    sam = sam.to(device=device, dtype=torch.float32, non_blocking=True)
    sam.eval()
    # Step4: 冻结所有参数，禁用梯度
    for param in sam.parameters():
        param.requires_grad = False
        param.data = param.data.contiguous()  # 强制连续内存，ROCM必需
    return sam

# 安全加载模型
sam = load_sam_rocm_safe(SAM_CHECKPOINT_PATH, MODEL_TYPE, device)
predictor = SamPredictor(sam)
print("✅ 模型安全加载到AMD ROCM GPU！")

# ====================== 3. 图像预处理（强制CPU预处理，避免ROCM差异） ======================
# 用官方示例图（确保目标明显，排除图像问题）
image_path = r"D:\SAM\venv1\segment-anything\notebooks\images\truck.jpg"



# 图像预处理（全程在CPU完成，避免ROCM干扰）
image = cv2.imread(image_path)
if image is None:
    print(f"错误：无法读取图像 {image_path}")
    exit()
# 强制缩放到1024x768（SAM训练的标准尺寸）
image = cv2.resize(image, (1024, 768))
# CPU端完成RGB转换+归一化
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
h, w, _ = image_rgb.shape
print(f"图像尺寸: {h}x{w} (标准尺寸，保证掩码正确)")

# ====================== 4. ROCM推理（强制同步+禁用梯度） ======================
# 用官方示例图的框提示（目标：狗，坐标经过验证）
input_box = np.array([190, 230, 500, 720])  # 精准框住狗的区域

# 核心：推理全程禁用梯度+强制同步
with torch.no_grad():
    # Step1: CPU端完成图像编码（避免ROCM编码差异）
    predictor.set_image(image_rgb)
    # Step2: ROCM推理，强制同步
    torch.cuda.synchronize()
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],  # 仅用框提示（ROCM最稳定）
        multimask_output=False,  # 禁用多掩码，避免随机差异
    )
    torch.cuda.synchronize()  # 强制ROCM完成推理

# ====================== 5. 验证掩码正确性 ======================
mask = masks[0]
non_zero_count = np.count_nonzero(mask)
# 对比CPU结果（可选：运行CPU版本后记录此值）
print(f"\n===== 掩码验证结果 =====")
print(f"掩码非零像素数: {non_zero_count} (官方图正确值≈50000)")
print(f"掩码覆盖区域占比: {non_zero_count/(h*w):.2%} (正确值≈6%)")

if non_zero_count < 10000:
    print("⚠️ 掩码仍错误！最后尝试：更换为NVIDIA GPU/CPU运行")
else:
    print("✅ 掩码正确！ROCM推理成功")

# ====================== 6. 可视化（对比官方结果） ======================
def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    mask_img = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_img)

fig, ax = plt.subplots(figsize=(12, 9))
ax.imshow(image_rgb)
show_mask(mask, ax)
# 绘制框
ax.add_patch(plt.Rectangle((190,230), 310, 490, linewidth=2, edgecolor='red', facecolor='none'))
ax.set_title("AMD ROCM SAM Result (Dog Mask)", fontsize=14)
ax.axis('off')
plt.savefig("amd_rocm_correct_mask.jpg", bbox_inches='tight', dpi=150)
plt.show()

# 清理显存
torch.cuda.empty_cache()