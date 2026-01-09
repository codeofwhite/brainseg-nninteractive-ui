import os
import torch
import SimpleITK as sitk
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import rcParams

# 配置Matplotlib确保图像比例正确
rcParams['image.aspect'] = 'equal'
rcParams['figure.autolayout'] = True  # 自动调整布局避免裁剪

class InteractiveSegmentationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Segmentation Tool")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # 配置网格权重实现自适应布局
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=3)  # 图像区域
        self.root.grid_columnconfigure(1, weight=1)  # 控制区域
        
        # 模型和数据变量（修正维度顺序：z=切片数, y=高度, x=宽度）
        self.session = None
        self.input_image = None          # SITK图像对象
        self.img_array = None            # numpy数组，维度(z, y, x)
        self.target_tensor = None        # PyTorch张量，维度(z, y, x)
        self.current_slice = 0           # 当前切片索引（z轴）
        self.interactions = []           # 交互点列表：[(z, y, x), is_positive]
        self.segmentation_result = None  # 分割结果，维度(z, y, x)
        
        # 创建GUI组件
        self.create_widgets()
    
    def create_widgets(self):
        # 创建菜单栏
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.load_image)
        file_menu.add_command(label="Set Model Path", command=self.set_model_path)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)
        
        # 图像显示框架（左侧）
        self.image_frame = ttk.Frame(self.root, padding="10")
        self.image_frame.grid(row=0, column=0, sticky="nsew")
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)
        
        # 创建图像显示区域（使用正方形初始比例）
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
        # 控制框架（右侧）
        self.control_frame = ttk.LabelFrame(self.root, text="Controls", padding="10")
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10))
        self.control_frame.grid_rowconfigure(10, weight=1)  # 用于垂直填充的空行
        self.control_frame.grid_columnconfigure(0, weight=1)
        
        # 模型路径选择
        ttk.Label(self.control_frame, text="Model Path:").grid(row=0, column=0, sticky="w", pady=(0, 5))
        self.model_path = tk.StringVar(value="")
        self.model_path_entry = ttk.Entry(self.control_frame, textvariable=self.model_path, state="readonly", width=30)
        self.model_path_entry.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        ttk.Button(self.control_frame, text="Browse Model", command=self.set_model_path).grid(row=2, column=0, sticky="ew", pady=(0, 10))
        
        # 切片控制（z轴）
        ttk.Label(self.control_frame, text="Slice (Z-axis):").grid(row=3, column=0, sticky="w", pady=(10, 5))
        self.slice_slider = ttk.Scale(self.control_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_slice)
        self.slice_slider.grid(row=4, column=0, sticky="ew", pady=(0, 5))
        self.slice_label = ttk.Label(self.control_frame, text="0/0")
        self.slice_label.grid(row=5, column=0, sticky="w", pady=(0, 10))
        
        # 交互类型选择
        ttk.Label(self.control_frame, text="Interaction Type:").grid(row=6, column=0, sticky="w", pady=(10, 5))
        self.interaction_type = tk.StringVar(value="positive")
        interaction_frame = ttk.Frame(self.control_frame)
        interaction_frame.grid(row=7, column=0, sticky="ew", pady=(0, 10))
        ttk.Radiobutton(interaction_frame, text="Positive", variable=self.interaction_type, value="positive").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(interaction_frame, text="Negative", variable=self.interaction_type, value="negative").pack(side=tk.LEFT, padx=5)
        
        # 动作按钮
        button_frame = ttk.Frame(self.control_frame)
        button_frame.grid(row=8, column=0, sticky="ew", pady=10)
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        
        ttk.Button(button_frame, text="Run Segmentation", command=self.run_segmentation).grid(row=0, column=0, sticky="ew", padx=(0, 5))
        ttk.Button(button_frame, text="Reset", command=self.reset_segmentation).grid(row=0, column=1, sticky="ew", padx=(5, 0))
        
        ttk.Button(self.control_frame, text="Save Result", command=self.save_result).grid(row=9, column=0, sticky="ew", pady=5)
        
        # 状态栏
        self.status_var = tk.StringVar(value="Ready: Set model path first")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")
    
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
                use_pinned_memory=True if torch.cuda.is_available() else False,
            )
            
            # 加载模型
            self.session.initialize_from_trained_model_folder(self.model_path.get())
            self.status_var.set("Model initialized successfully")
            
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to initialize model: {str(e)}")
            self.status_var.set("Model initialization failed")
    
    def load_image(self):
        if self.session is None:
            messagebox.showerror("Error", "Model not initialized. Please set model path first.")
            return
            
        file_path = filedialog.askopenfilename(
            filetypes=[("Medical Images", "*.nii *.nii.gz *.mha *.mhd *.dcm"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            self.status_var.set("Loading image...")
            self.root.update()
            
            # 读取图像（修正维度处理：z, y, x）
            self.input_image = sitk.ReadImage(file_path)
            self.img_array = sitk.GetArrayFromImage(self.input_image)  # 维度为(z, y, x)
            
            if self.img_array.ndim != 3:
                raise ValueError(f"Expected 3D image (z,y,x), got {self.img_array.ndim}D")
            
            # 设置图像到会话（增加batch维度变为(1, z, y, x)）
            self.session.set_image(self.img_array[np.newaxis, ...])
            
            # 初始化目标缓冲区
            self.target_tensor = torch.zeros_like(torch.from_numpy(self.img_array), dtype=torch.uint8)
            self.session.set_target_buffer(self.target_tensor)
            
            # 重置交互和结果
            self.reset_segmentation()
            
            # 设置切片滑块
            total_slices = self.img_array.shape[0]
            self.current_slice = total_slices // 2  # 默认显示中间切片
            self.slice_slider.config(from_=0, to=total_slices - 1)
            self.slice_slider.set(self.current_slice)
            self.slice_label.config(text=f"{self.current_slice + 1}/{total_slices}")
            
            # 显示图像
            self.display_image()
            
            self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to load image: {str(e)}")
            self.status_var.set("Image loading failed")
    
    def display_image(self):
        if self.img_array is None:
            return
            
        self.ax.clear()
        
        # 获取当前切片数据 (y, x)
        slice_data = self.img_array[self.current_slice, :, :]
        
        # 归一化显示
        slice_min, slice_max = np.min(slice_data), np.max(slice_data)
        if slice_max - slice_min > 1e-6:  # 避免除以零
            slice_data = (slice_data - slice_min) / (slice_max - slice_min)
        
        # 显示灰度图像，使用upper原点匹配医学图像习惯
        self.ax.imshow(slice_data, cmap='gray', origin='upper')
        self.ax.set_title(f"Slice {self.current_slice + 1}/{self.img_array.shape[0]}")
        self.ax.axis('off')  # 隐藏坐标轴，避免干扰比例
        
        # 显示交互点
        for (z, y, x), is_positive in self.interactions:
            if z == self.current_slice:
                color = 'green' if is_positive else 'red'
                self.ax.scatter(x, y, c=color, s=80, alpha=0.7, edgecolors='white', linewidths=0.5)
        
        # 显示分割结果
        if self.segmentation_result is not None:
            seg_slice = self.segmentation_result[self.current_slice, :, :]
            mask = seg_slice > 0
            if np.any(mask):
                self.ax.imshow(seg_slice, cmap='jet', alpha=0.5, origin='upper')
        
        # 确保比例正确并刷新
        self.canvas.draw()
    
    def on_click(self, event):
        if self.img_array is None or event.inaxes != self.ax:
            return
            
        # 正确映射坐标：x对应图像x轴，y对应图像y轴
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        z = self.current_slice
        
        # 检查坐标边界
        z_max, y_max, x_max = self.img_array.shape
        if 0 <= x < x_max and 0 <= y < y_max and 0 <= z < z_max:
            is_positive = (self.interaction_type.get() == "positive")
            self.interactions.append( ((z, y, x), is_positive) )
            self.session.add_point_interaction( (z, y, x), include_interaction=is_positive )
            self.status_var.set(f"Added {self.interaction_type.get()} point: (z:{z}, y:{y}, x:{x})")
            self.display_image()
    
    def update_slice(self, val):
        if self.img_array is None:
            return
            
        self.current_slice = int(float(val))
        total_slices = self.img_array.shape[0]
        self.slice_label.config(text=f"{self.current_slice + 1}/{total_slices}")
        self.display_image()
    
    def run_segmentation(self):
        if self.img_array is None or not self.interactions:
            messagebox.showinfo("Info", "Please load an image and add at least one interaction point")
            return
            
        try:
            self.status_var.set("Running segmentation...")
            self.root.update()
            
            self.segmentation_result = self.session.target_buffer.clone().numpy()
            self.status_var.set("Segmentation completed")
            self.display_image()
            
        except Exception as e:
            messagebox.showerror("Segmentation Error", f"Error: {str(e)}")
            self.status_var.set("Segmentation failed")
    
    def reset_segmentation(self):
        if self.session is not None:
            self.session.reset_interactions()
            if self.img_array is not None:
                self.target_tensor = torch.zeros_like(torch.from_numpy(self.img_array), dtype=torch.uint8)
                self.session.set_target_buffer(self.target_tensor)
        
        self.interactions = []
        self.segmentation_result = None
        
        if self.img_array is not None:
            self.display_image()
        
        self.status_var.set("Segmentation reset")
    
    def save_result(self):
        if self.segmentation_result is None:
            messagebox.showinfo("Info", "No segmentation result to save")
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
            
            # 确保维度匹配原始图像
            result_np = self.segmentation_result.astype(np.uint8)
            
            # 转换为SITK图像并保持原始信息
            result_image = sitk.GetImageFromArray(result_np)
            result_image.CopyInformation(self.input_image)
            sitk.WriteImage(result_image, file_path)
            
            self.status_var.set(f"Result saved to: {file_path}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving result: {str(e)}")
            self.status_var.set("Saving failed")

if __name__ == "__main__":
    root = tk.Tk()
    app = InteractiveSegmentationTool(root)
    root.mainloop()
