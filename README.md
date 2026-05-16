# nnInteractive - 交互式医学图像分割工具

基于 [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) 的交互式 3D 医学图像分割 GUI 工具，支持多种标注方式（点、框、涂鸦、套索），可用于医学图像的快速标注与分割。

## 项目简介

本项目在 nnInteractive 推理引擎基础上，提供了基于 Tkinter 的图形化交互界面，支持以下核心功能：

- **多种交互标注工具**：Point（点标注）、Bounding Box（框标注）、Scribble（涂鸦标注）、Lasso（套索标注）
- **预标注 Mask 微调**：支持加载已有的预标注 Mask，在此基础上进行增删微调
- **实时可视化**：标注过程实时显示在图像上，支持多切片浏览
- **撤销/重置**：支持撤销上一步操作或清除所有标注重新开始
- **多格式支持**：支持 `.nii`、`.nii.gz`、`.mha`、`.mhd`、`.dcm` 等医学图像格式

## 环境要求

- Python >= 3.10
- CUDA 支持的 GPU（推荐）
- 操作系统：Linux / Windows

## 安装

### 1. 安装依赖

```bash
pip install nnunetv2>=2.6 torch>=2.6 acvl-utils>=0.2.3 batchgenerators>=0.25.1
pip install SimpleITK Pillow matplotlib
```

或直接安装本项目：

```bash
pip install -e .
```

### 2. 下载模型

从 [Hugging Face](https://huggingface.co/nnInteractive/nnInteractive/tree/main) 下载预训练模型 `nnInteractive_v1.0`，或在 UI 界面中通过 `File -> Download Model` 直接下载。

## 使用方法

### 启动 GUI 界面

```bash
# 启动完整功能版（推荐，支持所有标注工具）
python ui_all_tools.py

# 启动精简版（仅支持点标注）
python ui_test.py
```

### 基本操作流程

#### 模式一：无预标注 Mask（从零标注）

1. **加载模型**：点击 `File -> Set Model Path` 或 `Browse` 按钮，选择模型目录
2. **导入图像**：点击 `File -> Open Image`，选择 `.nii.gz` 格式的医学图像
3. **选择标注工具**：在右侧面板选择 Point / Bounding Box / Scribble / Lasso
4. **执行标注**：在图像上进行标注操作
5. **运行分割**：点击 `Run Segmentation` 获取分割结果
6. **保存结果**：点击 `Save Result` 保存为 `.nii.gz` 格式

#### 模式二：基于预标注 Mask（微调标注）

1. 完成上述步骤 1-2（加载模型和图像）
2. **加载预标注 Mask**：点击 `File -> Load Initial Mask`，选择已有的 Mask 文件
3. **微调操作**：
   - **删除区域**：选择 `Remove` 模式，在需删除的区域添加负点
   - **新增区域**：选择 `Add` 模式，使用任意标注工具在新区域标注
4. **更新 Mask**：点击 `Update Mask` 查看更新后的结果
5. **保存结果**：点击 `Save Result` 保存

### 命令行推理

```python
from test_wechat import tumornnInteractivewithMutilBoxPoint_Inference

# 使用点标注推理
inferencer = tumornnInteractivewithMutilBoxPoint_Inference(propagate_with_type='point')
success, mask = inferencer.network_prediction(
    "your_image.nii.gz",
    unique_labs_list=[[141, 354, 383, 144, 360, 383, 4]]  # x1,y1,z1,x2,y2,z2,label
)
```

## 项目结构

```
nnInteractive/
├── README.md                          # 项目说明文档
├── LICENSE                            # Apache 2.0 许可证
├── pyproject.toml                     # 项目配置与依赖
├── setup.py                           # 安装脚本
├── ui_all_tools.py                    # 完整版 GUI（支持所有标注工具）
├── ui_test.py                         # 精简版 GUI（仅支持点标注）
├── ui_work.py                         # 工作版 GUI（支持点标注 + 多视图）
├── test_wechat.py                     # 命令行推理示例
├── imgs/                              # 文档截图
└── nnInteractive/                     # 核心推理库
    ├── inference/                     # 推理会话管理
    │   ├── inference_session.py       # 推理会话核心类
    │   └── cvpr2025_challenge_baseline/ # CVPR2025 挑战赛基线
    ├── interaction/                   # 交互处理
    ├── trainer/                       # 训练模块
    ├── supervoxel/                    # 超体素分割模块
    └── utils/                         # 工具函数
```

## 标注工具说明

| 工具 | 说明 | 推荐度 |
|------|------|--------|
| **Point** | 单击添加正/负点标注 | ⭐⭐⭐ 推荐 |
| **Scribble** | 按住左键连续拖动绘制涂鸦区域 | ⭐⭐⭐ 推荐 |
| **Lasso** | 左键添加顶点（≥3个），右键闭合区域 | ⭐⭐⭐ 推荐 |
| **Bounding Box** | 拖动绘制矩形框（当前存在已知问题） | ⭐ 暂不推荐 |

## 致谢

- [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) - 核心推理引擎
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) - 基础分割框架
- [SAM 2](https://github.com/facebookresearch/sam2) - 超体素模块引用

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 开源。