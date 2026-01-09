import os
import torch
import SimpleITK as sitk
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import rcParams
from matplotlib.path import Path
import matplotlib.patches as patches
from huggingface_hub import snapshot_download

# 配置Matplotlib
rcParams['image.aspect'] = 'equal'
rcParams['figure.autolayout'] = True

class EnhancedSegmentationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Segmentation Tool")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)
        
        # 布局配置
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=1)
        
        # 数据变量
        self.session = None
        self.input_image = None
        self.img_array = None  # 维度: (z, y, x)
        self.target_tensor = None  # 模型输出缓冲区
        self.current_slice = 0
        self.segmentation_result = None
        
        # 标注与历史记录变量
        self.interactions = []  # 记录所有交互操作
        self.interaction_history = []  # 用于回退操作
        self.current_tool = "point"
        self.current_interaction = None
        self.scribble_points = []
        self.lasso_points = []
        self.bbox_start = None  # 边界框起点 (x, y)
        self.bbox_end = None    # 边界框终点 (x, y)
        
        # Mask相关变量
        self.initial_mask = None  # 加载的初始mask
        self.mask_loaded = False  # 是否已加载mask
        
        # 模型配置
        self.REPO_ID = "nnInteractive/nnInteractive"
        self.MODEL_NAME = "nnInteractive_v1.0"
        self.interaction_type = tk.StringVar(value="positive")  # 交互类型：增/删
        
        # 创建界面
        self.create_widgets()
    
    def create_widgets(self):
        # 菜单栏
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.load_image)
        file_menu.add_command(label="Load Initial Mask", command=self.load_initial_mask)
        file_menu.add_command(label="Set Model Path", command=self.set_model_path)
        file_menu.add_command(label="Download Model", command=self.download_model)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)
        
        # 图像显示区
        self.image_frame = ttk.Frame(self.root, padding="10")
        self.image_frame.grid(row=0, column=0, sticky="nsew")
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)
        
        # 画布和事件绑定
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky="nsew")
        
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.root.bind('<Return>', self.on_enter_press)
        
        # 控制区
        self.control_frame = ttk.LabelFrame(self.root, text="Controls", padding="10")
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10))
        self.control_frame.grid_rowconfigure(20, weight=1)
        self.control_frame.grid_columnconfigure(0, weight=1)
        
        # 模型设置
        ttk.Label(self.control_frame, text="Model Configuration:").grid(row=0, column=0, sticky="w", pady=(0, 10))
        ttk.Label(self.control_frame, text="Model Path:").grid(row=1, column=0, sticky="w", pady=(0, 5))
        self.model_path = tk.StringVar(value="")
        self.model_path_entry = ttk.Entry(self.control_frame, textvariable=self.model_path, state="readonly", width=30)
        self.model_path_entry.grid(row=2, column=0, sticky="ew", pady=(0, 5))
        
        button_frame = ttk.Frame(self.control_frame)
        button_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        ttk.Button(button_frame, text="Browse", command=self.set_model_path).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(button_frame, text="Download", command=self.download_model).grid(row=0, column=1, sticky="ew", padx=(5, 0))
        
        # 工具选择
        ttk.Label(self.control_frame, text="Annotation Tools:").grid(row=4, column=0, sticky="w", pady=(15, 5))
        tool_frame = ttk.LabelFrame(self.control_frame, text="Select Tool")
        tool_frame.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        
        self.tool_var = tk.StringVar(value="point")
        tools = [("Point", "point"), ("Bounding Box", "bbox"), ("Scribble", "scribble"), ("Lasso", "lasso")]
        for i, (text, value) in enumerate(tools):
            ttk.Radiobutton(tool_frame, text=text, variable=self.tool_var, 
                           value=value, command=self.on_tool_change).grid(
                           row=i//2, column=i%2, sticky="w", padx=5, pady=2)
        
        # 交互类型选择（增/删）
        ttk.Label(self.control_frame, text="Operation:").grid(row=6, column=0, sticky="w", pady=(10, 5))
        self.operation_frame = ttk.Frame(self.control_frame)
        self.operation_frame.grid(row=7, column=0, sticky="ew", pady=(0, 10))
        ttk.Radiobutton(self.operation_frame, text="Add", variable=self.interaction_type, value="positive").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(self.operation_frame, text="Remove", variable=self.interaction_type, value="negative").pack(side=tk.LEFT, padx=5)
        
        # Mask状态显示
        self.mask_status = ttk.Label(self.control_frame, text="No initial mask loaded", foreground="gray")
        self.mask_status.grid(row=8, column=0, sticky="w", pady=(10, 5))
        
        # 切片控制
        ttk.Label(self.control_frame, text="Slice (Z-axis):").grid(row=9, column=0, sticky="w", pady=(10, 5))
        self.slice_slider = ttk.Scale(self.control_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_slice)
        self.slice_slider.grid(row=10, column=0, sticky="ew", pady=(0, 5))
        self.slice_label = ttk.Label(self.control_frame, text="0/0")
        self.slice_label.grid(row=11, column=0, sticky="w", pady=(0, 10))
        
        # 动作按钮（恢复Run Segmentation并增加回退）
        action_frame = ttk.Frame(self.control_frame)
        action_frame.grid(row=12, column=0, sticky="ew", pady=5)
        action_frame.grid_columnconfigure(0, weight=1)
        action_frame.grid_columnconfigure(1, weight=1)
        ttk.Button(action_frame, text="Run Segmentation", command=self.run_segmentation).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(action_frame, text="Update Mask", command=self.update_mask).grid(row=0, column=1, sticky="ew", padx=(5, 0))
        
        # 回退和清除按钮
        ttk.Button(self.control_frame, text="Undo Last Edit", command=self.undo_last_edit).grid(row=13, column=0, sticky="ew", pady=5)
        ttk.Button(self.control_frame, text="Clear All Edits", command=self.clear_annotations).grid(row=14, column=0, sticky="ew", pady=5)
        ttk.Button(self.control_frame, text="Save Result", command=self.save_result).grid(row=15, column=0, sticky="ew", pady=5)
        
        # 状态栏
        self.status_var = tk.StringVar(value="Ready: Load image first")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")
    
    # 模型下载和初始化方法
    def download_model(self):
        try:
            self.status_var.set("Downloading model...")
            self.root.update()
            download_dir = filedialog.askdirectory(title="Select Model Directory")
            if not download_dir:
                self.status_var.set("Download cancelled")
                return
            download_path = snapshot_download(
                repo_id=self.REPO_ID,
                allow_patterns=[f"{self.MODEL_NAME}/*"],
                local_dir=download_dir
            )
            model_path = os.path.join(download_dir, self.MODEL_NAME)
            self.model_path.set(model_path)
            self.initialize_model()
        except Exception as e:
            messagebox.showerror("Download Error", f"Failed: {str(e)}")
            self.status_var.set("Download failed")
    
    def set_model_path(self):
        directory = filedialog.askdirectory(title="Select Model Directory")
        if directory:
            self.model_path.set(directory)
            self.initialize_model()
    
    def initialize_model(self):
        if not self.model_path.get():
            return
        try:
            self.status_var.set("Initializing model...")
            self.root.update()
            from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
            self.session = nnInteractiveInferenceSession(
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                use_torch_compile=False,
                verbose=False,
                torch_n_threads=os.cpu_count() or 4,
                do_autozoom=True,
                use_pinned_memory=torch.cuda.is_available(),
            )
            self.session.initialize_from_trained_model_folder(self.model_path.get())
            self.status_var.set(f"Model ready (Device: {self.session.device.type})")
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed: {str(e)}")
            self.status_var.set("Model init failed")
    
    # 图像加载方法
    def load_image(self):
        if self.session is None:
            messagebox.showerror("Error", "Initialize model first")
            return
        file_path = filedialog.askopenfilename(
            filetypes=[("Medical Images", "*.nii *.nii.gz *.mha *.mhd *.dcm"), ("All Files", "*.*")]
        )
        if not file_path:
            return
        try:
            self.input_image = sitk.ReadImage(file_path)
            self.img_array = sitk.GetArrayFromImage(self.input_image)
            if self.img_array.ndim != 3:
                raise ValueError(f"Expected 3D image, got {self.img_array.ndim}D")
            
            # 初始化模型输入
            self.session.set_image(self.img_array[np.newaxis, ...])
            
            # 初始化目标缓冲区（全零）
            self.target_tensor = torch.zeros_like(torch.from_numpy(self.img_array), dtype=torch.uint8)
            self.session.set_target_buffer(self.target_tensor)
            
            # 重置状态
            self.reset_all()
            
            # 更新切片控制
            total_slices = self.img_array.shape[0]
            self.current_slice = total_slices // 2
            self.slice_slider.config(from_=0, to=total_slices - 1)
            self.slice_slider.set(self.current_slice)
            self.slice_label.config(text=f"{self.current_slice + 1}/{total_slices}")
            
            self.display_image()
            self.status_var.set(f"Loaded image: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed: {str(e)}")
            self.status_var.set("Image load failed")
    
    # 加载初始mask（修复显示问题）
    def load_initial_mask(self):
        if self.img_array is None:
            messagebox.showerror("Error", "Load an image first")
            return
            
        file_path = filedialog.askopenfilename(
            filetypes=[("Mask Files", "*.nii *.nii.gz *.mha *.mhd *.dcm"), ("All Files", "*.*")]
        )
        if not file_path:
            return
            
        try:
            # 读取mask
            mask_image = sitk.ReadImage(file_path)
            self.initial_mask = sitk.GetArrayFromImage(mask_image)
            
            # 检查mask与图像尺寸是否匹配
            if self.initial_mask.shape != self.img_array.shape:
                raise ValueError(f"Mask shape {self.initial_mask.shape} does not match image shape {self.img_array.shape}")
            
            # 保存原始mask值用于显示（不转为二值）
            self.initial_mask_original = self.initial_mask.copy()
            # 创建二值版本用于计算
            self.initial_mask = (self.initial_mask > 0).astype(np.uint8)
            
            # 保存初始状态用于回退
            self.interaction_history = []
            self.save_current_state()
            
            # 将mask设置为模型的初始目标缓冲区
            self.target_tensor = torch.from_numpy(self.initial_mask).to(dtype=torch.uint8)
            self.session.set_target_buffer(self.target_tensor)
            
            # 更新状态
            self.mask_loaded = True
            self.segmentation_result = self.initial_mask.copy()  # 显示初始mask
            self.mask_status.config(text=f"Loaded mask: {os.path.basename(file_path)}", foreground="green")
            self.status_var.set(f"Initial mask loaded. You can now edit it.")
            self.display_image()
            
        except Exception as e:
            messagebox.showerror("Mask Error", f"Failed to load mask: {str(e)}")
            self.status_var.set("Mask load failed")
    
    # 主要修改display_image方法，添加标注绘制逻辑
    def display_image(self):
        """显示图像、标注和结果，确保所有标注可见"""
        if self.img_array is None:
            return
        self.ax.clear()
        
        # 显示当前切片图像
        slice_data = self.img_array[self.current_slice, :, :]
        slice_min, slice_max = np.min(slice_data), np.max(slice_data)
        if slice_max - slice_min > 1e-6:
            slice_data = (slice_data - slice_min) / (slice_max - slice_min)
        self.ax.imshow(slice_data, cmap='gray', origin='upper')
        self.ax.set_title(f"Slice {self.current_slice + 1}/{self.img_array.shape[0]}")
        self.ax.axis('off')
        
        # 1. 显示初始mask（蓝色半透明）
        if self.mask_loaded and hasattr(self, 'initial_mask_original'):
            mask_slice = self.initial_mask_original[self.current_slice, :, :]
            if np.any(mask_slice > 0):
                self.ax.imshow(
                    np.ma.masked_where(mask_slice == 0, mask_slice),
                    cmap='Blues', alpha=0.5, origin='upper', vmin=0, vmax=1
                )
        
        # 2. 显示已完成的所有标注（核心修复）
        for interaction in self.interactions:
            tool_type, params, is_positive = interaction
            # 标注颜色：绿色为添加，红色为删除
            color = 'limegreen' if is_positive else 'crimson'
            
            if tool_type == "point":
                z, y, x = params
                if z == self.current_slice:
                    # 绘制点标注（带白色边框的圆形）
                    self.ax.scatter(
                        x, y, 
                        s=100,  # 点大小
                        c=color, 
                        alpha=0.8, 
                        edgecolors='white', 
                        linewidths=1.5,
                        marker='o'  # 圆形标记
                    )
            
            elif tool_type == "bbox":
                z, x1, x2, y1, y2 = params
                if z == self.current_slice:
                    # 绘制边界框（带虚线的矩形）
                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2,
                        edgecolor=color,
                        linestyle='--',  # 虚线
                        facecolor='none',
                        alpha=0.9
                    )
                    self.ax.add_patch(rect)
            
            elif tool_type == "scribble":
                z, points = params
                if z == self.current_slice:
                    points = np.array(points)
                    # 绘制涂鸦线
                    self.ax.plot(
                        points[:, 0], points[:, 1],
                        color=color,
                        linewidth=3,
                        alpha=0.8
                    )
            
            elif tool_type == "lasso":
                z, points = params
                if z == self.current_slice:
                    points = np.array(points)
                    # 绘制套索多边形
                    path = Path(np.vstack([points, points[0]]))  # 闭合多边形
                    patch = patches.PathPatch(
                        path,
                        linewidth=2,
                        edgecolor=color,
                        linestyle='-',
                        facecolor='none',
                        alpha=0.9
                    )
                    self.ax.add_patch(patch)
                    # 绘制套索顶点
                    self.ax.scatter(
                        points[:, 0], points[:, 1],
                        s=50,
                        c=color,
                        alpha=0.8,
                        edgecolors='white',
                        linewidths=1
                    )
        
        # 3. 显示当前正在绘制的临时标注（实时反馈）
        if self.current_interaction:
            # 临时标注颜色：深绿为添加，深红为删除
            temp_color = 'green' if self.interaction_type.get() == "positive" else 'red'
            
            if self.current_interaction == "bbox" and self.bbox_start and self.bbox_end:
                x1, y1 = self.bbox_start
                x2, y2 = self.bbox_end
                rect_x = min(x1, x2)
                rect_y = min(y1, y2)
                rect_width = abs(x2 - x1)
                rect_height = abs(y2 - y1)
                rect = patches.Rectangle(
                    (rect_x, rect_y), rect_width, rect_height,
                    linewidth=2,
                    edgecolor=temp_color,
                    facecolor='none',
                    alpha=0.7
                )
                self.ax.add_patch(rect)
            
            elif self.current_interaction == "scribble" and self.scribble_points:
                points = np.array(self.scribble_points)
                self.ax.plot(
                    points[:, 0], points[:, 1],
                    color=temp_color,
                    linewidth=3,
                    alpha=0.7
                )
            
            elif self.current_interaction == "lasso" and self.lasso_points:
                points = np.array(self.lasso_points)
                # 绘制顶点
                self.ax.scatter(
                    points[:, 0], points[:, 1],
                    s=50,
                    c=temp_color,
                    alpha=0.8,
                    edgecolors='white',
                    linewidths=1
                )
                # 绘制连接线
                if len(points) > 1:
                    self.ax.plot(
                        points[:, 0], points[:, 1],
                        color=temp_color,
                        linewidth=2,
                        alpha=0.7
                    )
                # 绘制闭合提示线
                if len(points) >= 3:
                    self.ax.plot(
                        [points[-1, 0], points[0, 0]], 
                        [points[-1, 1], points[0, 1]],
                        color=temp_color,
                        linestyle='--',
                        linewidth=1.5,
                        alpha=0.5
                    )
        
        # 4. 显示分割结果（红色半透明）
        if self.segmentation_result is not None:
            seg_slice = self.segmentation_result[self.current_slice, :, :]
            mask = seg_slice > 0
            if np.any(mask):
                self.ax.imshow(
                    np.ma.masked_where(seg_slice == 0, seg_slice),
                    cmap='Reds', alpha=0.4, origin='upper', vmin=0, vmax=1
                )
        
        # 强制刷新画布
        self.canvas.draw_idle()
    
    def save_current_state(self):
        """保存当前状态用于回退"""
        if self.session is None or self.target_tensor is None:
            return
        # 保存当前缓冲区状态和交互记录
        state = {
            'target_tensor': self.target_tensor.clone(),
            'interactions': [i for i in self.interactions]
        }
        self.interaction_history.append(state)
        # 限制历史记录长度
        if len(self.interaction_history) > 50:
            self.interaction_history.pop(0)
    
    def undo_last_edit(self):
        """回退到上一步状态"""
        if len(self.interaction_history) < 2:
            messagebox.showinfo("Info", "No more history to undo")
            return
            
        # 恢复到上一个状态
        prev_state = self.interaction_history[-2]
        self.target_tensor = prev_state['target_tensor'].clone()
        self.interactions = prev_state['interactions']
        self.session.set_target_buffer(self.target_tensor)
        
        # 更新显示
        self.segmentation_result = self.target_tensor.clone().numpy()
        self.interaction_history.pop()  # 移除当前状态
        self._reset_current_interaction()
        self.display_image()
        self.status_var.set("Undo last edit")
    
    def _get_image_coords(self, event):
        """获取有效图像坐标"""
        if event.inaxes != self.ax or self.img_array is None:
            return None
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        z = self.current_slice
        z_max, y_max, x_max = self.img_array.shape
        if 0 <= x < x_max and 0 <= y < y_max and 0 <= z < z_max:
            return (x, y, z)
        return None
    
    def on_click(self, event):
        """处理鼠标点击"""
        if self.img_array is None or self.session is None:
            return
        coords = self._get_image_coords(event)
        if not coords:
            return
        x, y, z = coords
        tool = self.current_tool
        
        if event.button == 1:  # 左键
            if tool == "point":
                is_positive = self.interaction_type.get() == "positive"
                # 保存当前状态用于回退
                self.save_current_state()
                # 添加点交互
                self.interactions.append(("point", (z, y, x), is_positive))
                self.session.add_point_interaction((z, y, x), include_interaction=is_positive)
                action = "added to" if is_positive else "removed from"
                self.status_var.set(f"Point {action} mask: (z:{z}, y:{y}, x:{x})")
                self.display_image()
                
            elif tool == "bbox":  # 开始绘制边界框
                self.current_interaction = "bbox"
                self.bbox_start = (x, y)
                self.bbox_end = (x, y)
                self.display_image()
                
            elif tool == "scribble":  # 开始绘制涂鸦
                self.current_interaction = "scribble"
                self.scribble_points = [(x, y)]
                self.display_image()
                
            elif tool == "lasso":  # 开始绘制套索
                self.current_interaction = "lasso"
                self.lasso_points.append((x, y))
                self.status_var.set(f"Lasso points: {len(self.lasso_points)} (Right-click to complete)")
                self.display_image()
        
        elif event.button == 3 and tool == "lasso":  # 右键完成套索
            if len(self.lasso_points) >= 3:
                self._complete_lasso()
            else:
                messagebox.showinfo("Info", "Lasso needs at least 3 points")
    
    def on_drag(self, event):
        """处理鼠标拖动"""
        if self.img_array is None or self.current_interaction is None:
            return
        coords = self._get_image_coords(event)
        if not coords:
            return
        x, y, _ = coords
        
        if self.current_interaction == "bbox":  # 更新边界框
            self.bbox_end = (x, y)
            self.display_image()
            
        elif self.current_interaction == "scribble":  # 更新涂鸦
            self.scribble_points.append((x, y))
            self.display_image()
    
    def on_release(self, event):
        """处理鼠标释放（完成标注）"""
        if self.img_array is None or self.current_interaction is None:
            return
        coords = self._get_image_coords(event)
        if not coords:
            self._reset_current_interaction()
            return
        x, y, z = coords
        is_positive = self.interaction_type.get() == "positive"
        action = "added to" if is_positive else "removed from"
        
        if self.current_interaction == "bbox":  # 完成边界框
            x1, y1 = self.bbox_start
            x2, y2 = x, y
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            # 检查边界框有效性
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                self.status_var.set("Bounding box too small (min 5x5 pixels)")
                self._reset_current_interaction()
                return
            
            # 保存当前状态用于回退
            self.save_current_state()
            # 添加边界框交互
            self.interactions.append(("bbox", (z, x1, x2, y1, y2), is_positive))
            bbox_coords = [[x1, x2], [y1, y2], [z, z+1]]
            self.session.add_bbox_interaction(bbox_coords, include_interaction=is_positive)
            self.status_var.set(f"BBox {action} mask: {x2-x1}x{y2-y1}px")
            self._reset_current_interaction()
            self.display_image()
        
        elif self.current_interaction == "scribble":  # 完成涂鸦
            if len(self.scribble_points) < 3:
                self.status_var.set("Scribble too short (min 3 points)")
            else:
                # 保存当前状态用于回退
                self.save_current_state()
                self.interactions.append(("scribble", (z, self.scribble_points.copy()), is_positive))
                self._create_scribble_image(z, is_positive)
                self.status_var.set(f"Scribble {action} mask")
            self._reset_current_interaction()
            self.display_image()
    
    def _complete_lasso(self):
        """完成套索标注"""
        z = self.current_slice
        is_positive = self.interaction_type.get() == "positive"
        action = "added to" if is_positive else "removed from"
        
        # 保存当前状态用于回退
        self.save_current_state()
        self.interactions.append(("lasso", (z, self.lasso_points.copy()), is_positive))
        self._create_lasso_image(z, is_positive)
        self.status_var.set(f"Lasso {action} mask with {len(self.lasso_points)} points")
        self._reset_current_interaction()
        self.display_image()
    
    def _create_scribble_image(self, z, is_positive):
        """创建涂鸦图像并添加到模型"""
        z_max, y_max, x_max = self.img_array.shape
        scribble_img = np.zeros((z_max, y_max, x_max), dtype=np.uint8)
        
        # 绘制涂鸦
        from PIL import Image, ImageDraw
        img = Image.fromarray(scribble_img[z])
        draw = ImageDraw.Draw(img)
        draw.line(self.scribble_points, fill=1, width=3)
        scribble_img[z] = np.array(img)
        
        # 添加到模型（用于编辑mask）
        self.session.add_scribble_interaction(scribble_img, include_interaction=is_positive)
    
    def _create_lasso_image(self, z, is_positive):
        """创建套索图像并添加到模型"""
        z_max, y_max, x_max = self.img_array.shape
        lasso_img = np.zeros((z_max, y_max, x_max), dtype=np.uint8)
        
        # 绘制套索
        from PIL import Image, ImageDraw
        img = Image.fromarray(lasso_img[z])
        draw = ImageDraw.Draw(img)
        draw.polygon(self.lasso_points, fill=1)
        lasso_img[z] = np.array(img)
        
        # 添加到模型（用于编辑mask）
        self.session.add_lasso_interaction(lasso_img, include_interaction=is_positive)
    
    def _reset_current_interaction(self):
        """重置当前交互状态"""
        self.current_interaction = None
        self.scribble_points = []
        self.lasso_points = []
        self.bbox_start = None
        self.bbox_end = None
    
    def on_enter_press(self, event):
        """按Enter键完成套索"""
        if self.current_tool == "lasso" and self.current_interaction == "lasso" and len(self.lasso_points) >= 3:
            self._complete_lasso()
    
    def on_tool_change(self):
        """切换工具时重置状态"""
        self.current_tool = self.tool_var.get()
        self._reset_current_interaction()
        self.display_image()
    
    def update_slice(self, val):
        """更新当前切片"""
        if self.img_array is None:
            return
        self._reset_current_interaction()
        self.current_slice = int(float(val))
        total_slices = self.img_array.shape[0]
        self.slice_label.config(text=f"{self.current_slice + 1}/{total_slices}")
        self.display_image()
    
    def run_segmentation(self):
        """从头开始分割（适用于没有初始mask的情况）"""
        if self.img_array is None or (not self.interactions and not self.mask_loaded):
            messagebox.showinfo("Info", "Load image and add annotations first")
            return
        
        try:
            self.status_var.set("Running segmentation...")
            self.root.update()
            
            # 保存状态
            self.save_current_state()
            # 获取分割结果
            self.segmentation_result = self.session.target_buffer.clone().numpy()
            self.status_var.set("Segmentation completed. You can edit or save.")
            self.display_image()
            
        except Exception as e:
            messagebox.showerror("Segmentation Error", f"Error: {str(e)}")
            self.status_var.set("Segmentation failed")
    
    def update_mask(self):
        """更新mask（基于初始mask和当前编辑，适用于有mask的情况）"""
        if self.img_array is None:
            messagebox.showinfo("Info", "Load an image first")
            return
        
        try:
            self.status_var.set("Updating mask...")
            self.root.update()
            
            # 保存状态
            self.save_current_state()
            # 获取更新后的mask
            self.segmentation_result = self.session.target_buffer.clone().numpy()
            self.status_var.set("Mask updated. Continue editing or save.")
            self.display_image()
            
        except Exception as e:
            messagebox.showerror("Update Error", f"Error: {str(e)}")
            self.status_var.set("Mask update failed")
    
    def clear_annotations(self):
        """清除所有编辑（恢复到初始状态）"""
        if self.session is not None:
            self.session.reset_interactions()
            # 恢复到初始状态
            if self.mask_loaded and self.initial_mask is not None:
                self.target_tensor = torch.from_numpy(self.initial_mask).to(dtype=torch.uint8)
                self.session.set_target_buffer(self.target_tensor)
                self.segmentation_result = self.initial_mask.copy()
                # 重置历史记录，只保留初始状态
                self.interaction_history = []
                self.save_current_state()
            else:
                # 没有初始mask时恢复为全零
                self.target_tensor = torch.zeros_like(torch.from_numpy(self.img_array), dtype=torch.uint8)
                self.session.set_target_buffer(self.target_tensor)
                self.segmentation_result = None
                self.interaction_history = []
        
        self.interactions = []
        self._reset_current_interaction()
        self.display_image()
        self.status_var.set("All edits cleared. Back to initial state.")
    
    def reset_all(self):
        """完全重置（包括初始mask）"""
        self.interactions = []
        self.interaction_history = []
        self.segmentation_result = None
        self._reset_current_interaction()
        
        # 重置mask相关
        self.initial_mask = None
        if hasattr(self, 'initial_mask_original'):
            delattr(self, 'initial_mask_original')
        self.mask_loaded = False
        self.mask_status.config(text="No initial mask loaded", foreground="gray")
        
        if self.session is not None and self.img_array is not None:
            self.session.reset_interactions()
            self.target_tensor = torch.zeros_like(torch.from_numpy(self.img_array), dtype=torch.uint8)
            self.session.set_target_buffer(self.target_tensor)
        
        if self.img_array is not None:
            self.display_image()
    
    def save_result(self):
        """保存结果"""
        if self.segmentation_result is None:
            messagebox.showinfo("Info", "No result to save. Run segmentation or update mask first.")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".nii.gz",
            filetypes=[("NIfTI Files", "*.nii *.nii.gz"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            self.status_var.set("Saving result...")
            self.root.update()
            
            # 转换为SITK图像并保持原始信息
            result_np = self.segmentation_result.astype(np.uint8)
            result_image = sitk.GetImageFromArray(result_np)
            result_image.CopyInformation(self.input_image)
            sitk.WriteImage(result_image, file_path)
            
            self.status_var.set(f"Result saved to: {file_path}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving result: {str(e)}")
            self.status_var.set("Save failed")

if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedSegmentationTool(root)
    root.mainloop()
    