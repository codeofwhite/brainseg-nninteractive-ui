import os
import torch
import SimpleITK as sitk
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class InteractiveSegmentationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("NII Image Segmentation Tool")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Configure grid layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=1)
        
        # Data variables
        self.session = None
        self.input_image = None
        self.img_array = None
        self.target_tensor = None
        self.current_slice = 0
        self.interactions = []
        self.segmentation_result = None
        self.spacing = None
        self.direction = None
        self.origin = None
        
        # Create widgets
        self.create_widgets()
    
    def create_widgets(self):
        # Menu bar
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open NII Image", command=self.load_image)
        file_menu.add_command(label="Set Model Path", command=self.set_model_path)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)
        
        # Image frame
        self.image_frame = ttk.Frame(self.root, padding="10")
        self.image_frame.grid(row=0, column=0, sticky="nsew")
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)
        
        # Image display
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Control frame
        self.control_frame = ttk.LabelFrame(self.root, text="Controls", padding="10")
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10))
        self.control_frame.grid_columnconfigure(0, weight=1)
        self.control_frame.grid_rowconfigure(10, weight=1)
        
        # Model path
        ttk.Label(self.control_frame, text="Model Path:").grid(row=0, column=0, sticky="w", pady=(0, 5))
        self.model_path = tk.StringVar(value="")
        ttk.Entry(self.control_frame, textvariable=self.model_path, state="readonly", width=30).grid(row=1, column=0, sticky="ew", pady=(0, 5))
        ttk.Button(self.control_frame, text="Browse Model", command=self.set_model_path).grid(row=2, column=0, sticky="ew", pady=(0, 10))
        
        # Slice control
        ttk.Label(self.control_frame, text="Slice:").grid(row=3, column=0, sticky="w", pady=(10, 5))
        self.slice_slider = ttk.Scale(self.control_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_slice)
        self.slice_slider.grid(row=4, column=0, sticky="ew", pady=(0, 5))
        self.slice_label = ttk.Label(self.control_frame, text="0/0")
        self.slice_label.grid(row=5, column=0, sticky="w", pady=(0, 10))
        
        # View orientation
        ttk.Label(self.control_frame, text="View Orientation:").grid(row=6, column=0, sticky="w", pady=(10, 5))
        self.view_orientation = tk.StringVar(value="axial")
        orientation_frame = ttk.Frame(self.control_frame)
        orientation_frame.grid(row=7, column=0, sticky="ew", pady=(0, 10))
        ttk.Radiobutton(orientation_frame, text="Axial", variable=self.view_orientation, value="axial", command=self.update_orientation).pack(side=tk.LEFT)
        ttk.Radiobutton(orientation_frame, text="Sagittal", variable=self.view_orientation, value="sagittal", command=self.update_orientation).pack(side=tk.LEFT)
        ttk.Radiobutton(orientation_frame, text="Coronal", variable=self.view_orientation, value="coronal", command=self.update_orientation).pack(side=tk.LEFT)
        
        # Interaction type
        ttk.Label(self.control_frame, text="Interaction Type:").grid(row=8, column=0, sticky="w", pady=(10, 5))
        self.interaction_type = tk.StringVar(value="positive")
        interaction_frame = ttk.Frame(self.control_frame)
        interaction_frame.grid(row=9, column=0, sticky="ew", pady=(0, 10))
        ttk.Radiobutton(interaction_frame, text="Positive", variable=self.interaction_type, value="positive").pack(side=tk.LEFT)
        ttk.Radiobutton(interaction_frame, text="Negative", variable=self.interaction_type, value="negative").pack(side=tk.LEFT)
        
        # Buttons
        ttk.Button(self.control_frame, text="Run Segmentation", command=self.run_segmentation).grid(row=10, column=0, sticky="ew", pady=5)
        ttk.Button(self.control_frame, text="Reset", command=self.reset_segmentation).grid(row=11, column=0, sticky="ew", pady=5)
        ttk.Button(self.control_frame, text="Save Result", command=self.save_result).grid(row=12, column=0, sticky="ew", pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
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
                torch_n_threads=os.cpu_count(),
                do_autozoom=True,
                use_pinned_memory=True if torch.cuda.is_available() else False,
            )
            
            self.session.initialize_from_trained_model_folder(self.model_path.get())
            self.status_var.set("Model initialized successfully")
            
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to initialize model: {str(e)}")
            self.status_var.set("Model initialization failed")
    
    def ensure_contiguous_array(self, array):
        """确保数组是连续内存布局，避免负步长问题"""
        if not array.flags['C_CONTIGUOUS'] and not array.flags['F_CONTIGUOUS']:
            # 如果数组不是连续布局，创建副本
            return array.copy()
        return array
    
    def load_image(self):
        if self.session is None:
            messagebox.showerror("Error", "Model not initialized. Please set model path first.")
            return
            
        file_path = filedialog.askopenfilename(
            filetypes=[("NIfTI Files", "*.nii *.nii.gz"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            self.status_var.set("Loading image...")
            self.root.update()
            
            # Read NIfTI image
            self.input_image = sitk.ReadImage(file_path)
            
            # Get important metadata
            self.spacing = self.input_image.GetSpacing()
            self.direction = self.input_image.GetDirection()
            self.origin = self.input_image.GetOrigin()
            
            print(f"Image spacing: {self.spacing}")
            print(f"Image direction: {self.direction}")
            print(f"Image origin: {self.origin}")
            print(f"Image size: {self.input_image.GetSize()}")
            
            # Convert to numpy array
            img_array = sitk.GetArrayFromImage(self.input_image)
            print(f"Original array shape: {img_array.shape}")
            print(f"Array flags: C_CONTIGUOUS={img_array.flags['C_CONTIGUOUS']}, F_CONTIGUOUS={img_array.flags['F_CONTIGUOUS']}")
            
            # 确保数组是连续布局
            img_array = self.ensure_contiguous_array(img_array)
            
            # 重新排列维度为 (z, y, x) 并确保正确的方向
            # 对于大多数医疗图像，我们需要确保方向正确
            img_array = self.standardize_orientation(img_array)
            print(f"After standardization shape: {img_array.shape}")
            
            # 添加通道维度 (1, z, y, x)
            if img_array.ndim == 3:
                img_array = img_array[np.newaxis, ...]
            
            # 再次确保数组是连续的
            self.img_array = self.ensure_contiguous_array(img_array)
            print(f"Final array shape: {self.img_array.shape}")
            print(f"Final array flags: C_CONTIGUOUS={self.img_array.flags['C_CONTIGUOUS']}")
            
            # 验证数组没有负步长
            if any(stride < 0 for stride in self.img_array.strides):
                print("Warning: Array has negative strides, making copy...")
                self.img_array = self.img_array.copy()
            
            # 设置图像到session
            self.session.set_image(self.img_array)
            
            # 初始化目标缓冲区 (z, y, x)
            self.target_tensor = torch.zeros(self.img_array.shape[1:], dtype=torch.uint8)
            self.session.set_target_buffer(self.target_tensor)
            
            # 重置并配置视图
            self.reset_segmentation()
            self.update_orientation()
            
            self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to load image: {str(e)}")
            import traceback
            traceback.print_exc()
            self.status_var.set("Image loading failed")
    
    def standardize_orientation(self, img_array):
        """标准化图像方向到 (z, y, x) 布局"""
        # 确保数组是连续布局
        img_array = self.ensure_contiguous_array(img_array)
        
        # 根据原始数组形状决定如何重新排列
        if img_array.ndim == 3:
            # 已经是3D，确保方向正确
            # 通常SimpleITK返回 (z, y, x)，但我们需要检查方向
            pass
        
        # 确保数据类型合适
        if img_array.dtype != np.float32:
            img_array = img_array.astype(np.float32)
        
        return img_array
    
    def get_current_view_slice(self):
        """基于选择的方位获取适当的切片"""
        if self.img_array is None:
            return None, 0
            
        # 数组维度: (1, z, y, x)
        _, z_dim, y_dim, x_dim = self.img_array.shape
        
        # 根据方位获取适当的切片
        if self.view_orientation.get() == "axial":  # x-y 平面 (轴向视图)
            max_slice = z_dim - 1
            if self.current_slice > max_slice:
                self.current_slice = max_slice
            slice_data = self.img_array[0, self.current_slice, :, :]
            return slice_data, max_slice
            
        elif self.view_orientation.get() == "sagittal":  # y-z 平面 (矢状视图)
            max_slice = x_dim - 1
            if self.current_slice > max_slice:
                self.current_slice = max_slice
            slice_data = self.img_array[0, :, :, self.current_slice]
            return slice_data, max_slice
            
        else:  # coronal, x-z 平面 (冠状视图)
            max_slice = y_dim - 1
            if self.current_slice > max_slice:
                self.current_slice = max_slice
            slice_data = self.img_array[0, :, self.current_slice, :]
            return slice_data, max_slice
    
    def get_aspect_ratio(self):
        """基于NIfTI间距计算正确的宽高比"""
        if self.spacing is None:
            return 'auto'
            
        # 基于方位计算宽高比
        if self.view_orientation.get() == "axial":  # x-y 平面
            return self.spacing[0] / self.spacing[1]
            
        elif self.view_orientation.get() == "sagittal":  # y-z 平面
            return self.spacing[1] / self.spacing[2]
            
        else:  # coronal, x-z 平面
            return self.spacing[0] / self.spacing[2]
    
    def update_orientation(self):
        """当方位改变时更新显示"""
        if self.img_array is None:
            return
            
        # 改变方位时重置到第一切片
        self.current_slice = 0
        
        # 获取新方位的最大切片数
        _, max_slice = self.get_current_view_slice()
        
        # 更新滑块
        self.slice_slider.config(from_=0, to=max_slice)
        self.slice_slider.set(self.current_slice)
        self.slice_label.config(text=f"{self.current_slice + 1}/{max_slice + 1}")
        
        # 用新方位重绘图像
        self.display_image()
    
    def display_image(self):
        if self.img_array is None:
            return
            
        self.ax.clear()
        
        # 获取当前切片和最大切片计数
        slice_data, max_slice = self.get_current_view_slice()
        if slice_data is None:
            return
            
        # 为更好的可视化进行归一化
        slice_min, slice_max = np.min(slice_data), np.max(slice_data)
        if slice_max > slice_min:
            slice_data = (slice_data - slice_min) / (slice_max - slice_min)
        
        # 从NIfTI元数据获取正确的宽高比
        aspect_ratio = self.get_aspect_ratio()
        
        # 用正确的方位和宽高比显示图像
        self.ax.imshow(slice_data, cmap='gray', aspect=aspect_ratio, 
                      origin='lower')
        
        self.ax.set_title(f"{self.view_orientation.get().capitalize()} Slice {self.current_slice + 1}/{max_slice + 1}")
        self.ax.axis('off')
        
        # 显示交互点
        self.show_interaction_points()
        
        # 如果可用，显示分割结果
        self.show_segmentation_result()
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def show_interaction_points(self):
        """基于当前视图正确显示交互点"""
        if not self.interactions:
            return
            
        for (x, y, z), is_positive in self.interactions:
            # 基于当前视图将3D坐标转换为2D
            if self.view_orientation.get() == "axial" and z == self.current_slice:
                # 对于轴向视图：显示 (x, y) 坐标
                self.ax.plot(x, y, 'o', color='green' if is_positive else 'red', 
                           markersize=8, alpha=0.7, markeredgecolor='white')
            elif self.view_orientation.get() == "sagittal" and x == self.current_slice:
                # 对于矢状视图：显示 (z, y) 坐标
                self.ax.plot(z, y, 'o', color='green' if is_positive else 'red', 
                           markersize=8, alpha=0.7, markeredgecolor='white')
            elif self.view_orientation.get() == "coronal" and y == self.current_slice:
                # 对于冠状视图：显示 (z, x) 坐标
                self.ax.plot(z, x, 'o', color='green' if is_positive else 'red', 
                           markersize=8, alpha=0.7, markeredgecolor='white')
    
    def show_segmentation_result(self):
        """用正确的方位显示分割结果"""
        if self.segmentation_result is None:
            return
            
        try:
            # 基于当前方位获取分割切片
            if self.view_orientation.get() == "axial":
                seg_slice = self.segmentation_result[self.current_slice, :, :]
            elif self.view_orientation.get() == "sagittal":
                seg_slice = self.segmentation_result[:, :, self.current_slice]
            else:  # coronal
                seg_slice = self.segmentation_result[:, self.current_slice, :]
            
            # 只显示存在分割的区域
            mask = seg_slice > 0
            if np.any(mask):
                # 创建彩色叠加
                from matplotlib.colors import ListedColormap
                cmap = ListedColormap(['none', 'red'])  # 0为透明，>0为红色
                self.ax.imshow(seg_slice, cmap=cmap, alpha=0.3, 
                              aspect=self.get_aspect_ratio(), origin='lower')
        except Exception as e:
            print(f"Error displaying segmentation: {e}")
    
    def on_click(self, event):
        if self.img_array is None or event.inaxes != self.ax:
            return
            
        # 获取图像维度
        _, z_dim, y_dim, x_dim = self.img_array.shape
        
        # 基于当前视图将2D点击坐标转换为3D
        x, y, z = 0, 0, 0
        
        if event.xdata is None or event.ydata is None:
            return
            
        click_x = int(round(event.xdata))
        click_y = int(round(event.ydata))
        
        if self.view_orientation.get() == "axial":
            # 轴向: (x, y) 来自点击，z来自当前切片
            x, y, z = click_x, click_y, self.current_slice
        elif self.view_orientation.get() == "sagittal":
            # 矢状: x来自当前切片，(z, y)来自点击
            x, y, z = self.current_slice, click_y, click_x
        else:  # coronal
            # 冠状: y来自当前切片，(z, x)来自点击
            x, y, z = click_x, self.current_slice, click_y
        
        # 检查坐标是否在边界内
        if (0 <= x < x_dim and 0 <= y < y_dim and 0 <= z < z_dim):
            is_positive = (self.interaction_type.get() == "positive")
            self.interactions.append( ((x, y, z), is_positive) )
            
            try:
                self.session.add_point_interaction( (x, y, z), include_interaction=is_positive )
                self.status_var.set(f"Added {self.interaction_type.get()} point: ({x}, {y}, {z})")
                self.display_image()
            except Exception as e:
                messagebox.showerror("Interaction Error", f"Failed to add interaction point: {str(e)}")
                # 移除失败的交互相应
                self.interactions.pop()
    
    def update_slice(self, val):
        if self.img_array is None:
            return
            
        self.current_slice = int(float(val))
        _, max_slice = self.get_current_view_slice()
        self.slice_label.config(text=f"{self.current_slice + 1}/{max_slice + 1}")
        self.display_image()
    
    def run_segmentation(self):
        if self.img_array is None or not self.interactions:
            messagebox.showinfo("Info", "Please load an image and add at least one interaction point")
            return
            
        try:
            self.status_var.set("Running segmentation...")
            self.root.update()
            
            # 从session获取分割结果
            self.segmentation_result = self.session.target_buffer.clone().numpy()
            print(f"Segmentation result shape: {self.segmentation_result.shape}")
            
            self.status_var.set("Segmentation completed")
            self.display_image()
            
        except Exception as e:
            messagebox.showerror("Segmentation Error", f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            self.status_var.set("Segmentation failed")
    
    def reset_segmentation(self):
        if self.session is not None:
            self.session.reset_interactions()
            if self.img_array is not None:
                self.target_tensor = torch.zeros(self.img_array.shape[1:], dtype=torch.uint8)
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
            
            # 转换分割结果为适当的数据类型
            result_np = self.segmentation_result.astype(np.uint8)
            
            # 使用正确的元数据创建SimpleITK图像
            result_image = sitk.GetImageFromArray(result_np)
            result_image.CopyInformation(self.input_image)
            
            sitk.WriteImage(result_image, file_path)
            
            self.status_var.set(f"Result saved to: {file_path}")
            messagebox.showinfo("Success", f"Segmentation result saved to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving result: {str(e)}")
            self.status_var.set("Saving failed")

if __name__ == "__main__":
    root = tk.Tk()
    app = InteractiveSegmentationTool(root)
    root.mainloop()