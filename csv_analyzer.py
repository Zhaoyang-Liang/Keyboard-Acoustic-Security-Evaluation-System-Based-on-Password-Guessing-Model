import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import tkinterdnd2 as tkdnd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime
import threading
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

class CSVAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 高级预测结果分析工具")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # 数据存储
        self.df = None
        self.stats = {}
        self.output_dir = "analysis_output"
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置UI界面"""
        
        # 主标题
        title_frame = tk.Frame(self.root, bg='#f0f0f0')
        title_frame.pack(pady=20)
        
        title_label = tk.Label(title_frame, text="🎯 高级预测结果分析工具", 
                              font=('Arial', 20, 'bold'), bg='#f0f0f0', fg='#2E86AB')
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, text="拖拽CSV文件到下方区域，自动生成专业分析图表", 
                                 font=('Arial', 12), bg='#f0f0f0', fg='#666')
        subtitle_label.pack(pady=5)
        
        # 拖拽区域 - 修复relief参数
        self.drop_frame = tk.Frame(self.root, bg='#e8f4f8', relief='ridge', bd=2)
        self.drop_frame.pack(pady=20, padx=50, fill='both', expand=True)
        
        # 注册拖拽
        self.drop_frame.drop_target_register(tkdnd.DND_FILES)
        self.drop_frame.dnd_bind('<<Drop>>', self.on_file_drop)
        
        self.drop_label = tk.Label(self.drop_frame, 
                                  text="📁\n\n拖拽CSV文件到这里\n或点击选择文件\n\n支持高级预测结果CSV分析", 
                                  font=('Arial', 14), bg='#e8f4f8', fg='#2E86AB',
                                  justify='center')
        self.drop_label.pack(expand=True)
        self.drop_label.bind('<Button-1>', self.select_file)
        
        # 控制按钮区域
        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        self.select_btn = tk.Button(button_frame, text="📂 选择文件", 
                                   command=self.select_file, font=('Arial', 12),
                                   bg='#2E86AB', fg='white', padx=20)
        self.select_btn.pack(side='left', padx=10)
        
        self.analyze_btn = tk.Button(button_frame, text="🚀 开始分析", 
                                    command=self.start_analysis, font=('Arial', 12),
                                    bg='#A23B72', fg='white', padx=20, state='disabled')
        self.analyze_btn.pack(side='left', padx=10)
        
        self.open_folder_btn = tk.Button(button_frame, text="📁 打开结果", 
                                        command=self.open_output_folder, font=('Arial', 12),
                                        bg='#F18F01', fg='white', padx=20, state='disabled')
        self.open_folder_btn.pack(side='left', padx=10)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, 
                                           maximum=100, length=600)
        self.progress_bar.pack(pady=10)
        
        # 状态显示区域
        self.status_text = scrolledtext.ScrolledText(self.root, height=8, 
                                                    font=('Consolas', 10))
        self.status_text.pack(pady=10, padx=20, fill='x')
        
        self.log("👋 欢迎使用高级预测结果分析工具!")
        self.log("📋 请拖拽或选择包含预测结果的CSV文件")
        
    def log(self, message):
        """添加日志信息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update()
        
    def on_file_drop(self, event):
        """处理文件拖拽"""
        files = self.root.tk.splitlist(event.data)
        if files:
            file_path = files[0]
            if file_path.lower().endswith('.csv'):
                self.load_csv_file(file_path)
            else:
                messagebox.showerror("错误", "请选择CSV文件!")
                
    def select_file(self, event=None):
        """选择文件对话框"""
        file_path = filedialog.askopenfilename(
            title="选择CSV文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.load_csv_file(file_path)
            
    def load_csv_file(self, file_path):
        """加载CSV文件"""
        try:
            self.log(f"📁 加载文件: {os.path.basename(file_path)}")
            self.df = pd.read_csv(file_path, encoding='utf-8')
            
            # 数据预处理
            self.preprocess_data()
            
            # 更新UI状态
            self.drop_label.config(text=f"✅ 已加载文件\n\n{os.path.basename(file_path)}\n\n数据行数: {len(self.df)}")
            self.analyze_btn.config(state='normal')
            
            self.log(f"✅ 文件加载成功! 共 {len(self.df)} 行数据")
            self.log("🔍 数据预览:")
            
            # 显示基本信息
            valid_df = self.df[self.df['预期序列'].notna() & (self.df['预期序列'] != '')].copy()
            self.log(f"   有效文件数: {len(valid_df)}")
            self.log(f"   列名: {list(self.df.columns)}")
            
        except Exception as e:
            self.log(f"❌ 文件加载失败: {str(e)}")
            messagebox.showerror("错误", f"文件加载失败:\n{str(e)}")
            
    def preprocess_data(self):
        """数据预处理"""
        # 数值列转换
        numeric_columns = ['声音模型字符准确率', '纯Seq2Seq字符准确率', '最终字符准确率']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # 位置列转换
        position_columns = ['声音模型最佳位置', '掩码模板猜对位置']
        for col in position_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                
    def start_analysis(self):
        """开始分析"""
        if self.df is None:
            messagebox.showerror("错误", "请先加载CSV文件!")
            return
            
        # 在新线程中运行分析，避免界面卡顿
        self.analyze_btn.config(state='disabled')
        threading.Thread(target=self.run_analysis, daemon=True).start()
        
    def run_analysis(self):
        """运行分析（在后台线程）"""
        try:
            self.log("🚀 开始数据分析...")
            self.progress_var.set(10)
            
            # 计算统计信息
            self.calculate_stats()
            self.progress_var.set(30)
            
            # 生成图表
            self.generate_charts()
            self.progress_var.set(80)
            
            # 生成报告
            self.generate_report()
            self.progress_var.set(100)
            
            self.log("🎉 分析完成!")
            self.root.after(0, self.analysis_complete)
            
        except Exception as e:
            self.log(f"❌ 分析失败: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("错误", f"分析失败:\n{str(e)}"))
            self.root.after(0, lambda: self.analyze_btn.config(state='normal'))
            
    def calculate_stats(self):
        """计算统计信息"""
        self.log("📊 计算统计信息...")
        
        valid_df = self.df[self.df['预期序列'].notna() & (self.df['预期序列'] != '')].copy()
        total_files = len(valid_df)
        
        self.stats = {
            'total_files': total_files,
            'valid_files': total_files
        }
        
        if total_files == 0:
            return
            
        # 掩码排名统计
        ranking_dist = valid_df['掩码模板猜对位置'].value_counts().sort_index()
        self.stats['ranking_distribution'] = ranking_dist
        
        # Top-K命中率
        top_k_stats = {}
        for k in range(1, 16):
            hits = len(valid_df[(valid_df['掩码模板猜对位置'] > 0) & 
                               (valid_df['掩码模板猜对位置'] <= k)])
            hit_rate = hits / total_files if total_files > 0 else 0
            top_k_stats[f'top_{k}'] = hit_rate
            
        self.stats['top_k_hit_rates'] = top_k_stats
        
        # 准确率统计
        self.stats.update({
            'sound_char_accuracy': valid_df['声音模型字符准确率'].mean(),
            'seq2seq_char_accuracy': valid_df['纯Seq2Seq字符准确率'].mean(),
            'advanced_char_accuracy': valid_df['最终字符准确率'].mean()
        })
        
        # 序列级准确率
        for col_pair in [('声音模型最佳结果', 'sound_sequence_accuracy'),
                        ('纯Seq2Seq最佳结果', 'seq2seq_sequence_accuracy'),
                        ('最终融合最佳结果', 'advanced_sequence_accuracy')]:
            matches = 0
            for idx, row in valid_df.iterrows():
                expected = str(row['预期序列']).strip()
                predicted = str(row[col_pair[0]]).strip()
                if expected and expected != 'nan' and predicted == expected:
                    matches += 1
            self.stats[col_pair[1]] = matches / total_files if total_files > 0 else 0
            
    def generate_charts(self):
        """生成图表"""
        self.log("🎨 生成可视化图表...")
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        valid_df = self.df[self.df['预期序列'].notna() & (self.df['预期序列'] != '')].copy()
        
        # 配色方案
        colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#27AE60',
            'gradient': ['#2E86AB', '#A23B72', '#F18F01', '#27AE60']
        }
        
        # 1. Top-K命中率图表
        self.create_topk_chart(colors)
        
        # 2. 准确率对比图
        self.create_accuracy_comparison(colors)
        
        # 3. 排名分布图
        self.create_ranking_distribution(colors, valid_df)
        
        # 4. 性能提升图
        self.create_improvement_chart(colors)
        
    def create_topk_chart(self, colors):
        """创建Top-K命中率图表"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        k_values = list(range(1, 11))
        hit_rates = [self.stats['top_k_hit_rates'][f'top_{k}'] * 100 for k in k_values]
        
        bars = ax.bar(k_values, hit_rates, color=colors['primary'], alpha=0.8, 
                     edgecolor='white', linewidth=2)
        
        # 添加趋势线
        ax.plot(k_values, hit_rates, color=colors['accent'], linewidth=3, 
               marker='o', markersize=8, alpha=0.7)
        
        # 添加数值标签
        for k, rate in zip(k_values, hit_rates):
            ax.text(k, rate + 1, f'{rate:.1f}%', ha='center', va='bottom', 
                   fontweight='bold', fontsize=11)
        
        ax.set_xlabel('候选排名范围 (Top-K)', fontsize=14, fontweight='bold')
        ax.set_ylabel('累积命中率 (%)', fontsize=14, fontweight='bold')
        ax.set_title('🎯 掩码模板多候选预测性能 - Top-K累积命中率', 
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.set_xticks(k_values)
        ax.set_xticklabels([f'Top-{k}' for k in k_values])
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(hit_rates) * 1.1)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_topk_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_accuracy_comparison(self, colors):
        """创建准确率对比图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 序列级准确率
        models = ['声音模型', '纯Seq2Seq', '高级模型']
        seq_accuracies = [
            self.stats['sound_sequence_accuracy'],
            self.stats['seq2seq_sequence_accuracy'], 
            self.stats['advanced_sequence_accuracy']
        ]
        
        bars1 = ax1.bar(models, seq_accuracies, color=colors['gradient'][:3], alpha=0.8)
        ax1.set_title('序列级准确率对比', fontsize=14, fontweight='bold')
        ax1.set_ylabel('准确率', fontsize=12)
        ax1.set_ylim(0, 1)
        
        for bar, acc in zip(bars1, seq_accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')
        
        # 字符级准确率
        char_accuracies = [
            self.stats['sound_char_accuracy'],
            self.stats['seq2seq_char_accuracy'],
            self.stats['advanced_char_accuracy']
        ]
        
        bars2 = ax2.bar(models, char_accuracies, color=colors['gradient'][:3], alpha=0.8)
        ax2.set_title('字符级准确率对比', fontsize=14, fontweight='bold')
        ax2.set_ylabel('准确率', fontsize=12)
        ax2.set_ylim(0, 1)
        
        for bar, acc in zip(bars2, char_accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')
        
        # Top-K对比
        scenarios = ['Top-1', 'Top-3', 'Top-5', 'Top-10']
        success_rates = [
            self.stats['top_k_hit_rates']['top_1'] * 100,
            self.stats['top_k_hit_rates']['top_3'] * 100,
            self.stats['top_k_hit_rates']['top_5'] * 100,
            self.stats['top_k_hit_rates']['top_10'] * 100
        ]
        
        bars3 = ax3.bar(scenarios, success_rates, color=colors['gradient'], alpha=0.8)
        ax3.set_title('多候选策略效果', fontsize=14, fontweight='bold')
        ax3.set_ylabel('成功率 (%)', fontsize=12)
        
        for bar, rate in zip(bars3, success_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 准确率分布
        valid_df = self.df[self.df['预期序列'].notna() & (self.df['预期序列'] != '')].copy()
        ax4.hist(valid_df['最终字符准确率'], bins=20, alpha=0.7, 
                color=colors['primary'], edgecolor='black')
        ax4.set_title('高级模型准确率分布', fontsize=14, fontweight='bold')
        ax4.set_xlabel('字符准确率', fontsize=12)
        ax4.set_ylabel('文件数量', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/02_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_ranking_distribution(self, colors, valid_df):
        """创建排名分布图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 饼图
        ranking_data = self.stats['ranking_distribution'].copy()
        
        major_ranks = {}
        other_count = 0
        
        for rank, count in ranking_data.items():
            if rank == -1:
                major_ranks['未命中'] = count
            elif rank <= 5 or count >= len(valid_df) * 0.05:
                major_ranks[f'第{rank}位'] = count
            else:
                other_count += count
                
        if other_count > 0:
            major_ranks['其他排名'] = other_count
            
        pie_colors = [colors['accent']] + colors['gradient'] + ['#95A5A6']
        pie_colors = pie_colors[:len(major_ranks)]
        
        wedges, texts, autotexts = ax1.pie(major_ranks.values(), labels=major_ranks.keys(),
                                          autopct='%1.1f%%', startangle=90, colors=pie_colors)
        ax1.set_title('排名分布概览', fontsize=14, fontweight='bold')
        
        # 详细柱状图
        positions = []
        counts = []
        
        for pos, count in self.stats['ranking_distribution'].items():
            if pos != -1 and pos <= 10:
                positions.append(f'第{pos}位')
                counts.append(count)
                
        if positions:
            bars = ax2.bar(positions, counts, color=colors['secondary'], alpha=0.8)
            ax2.set_title('前10名详细分布', fontsize=14, fontweight='bold')
            ax2.set_ylabel('文件数量', fontsize=12)
            ax2.tick_params(axis='x', rotation=45)
            
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_ranking_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_improvement_chart(self, colors):
        """创建性能提升图"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scenarios = ['单一预测\n(Top-1)', '3选1预测\n(Top-3)', '5选1预测\n(Top-5)', '10选1预测\n(Top-10)']
        success_rates = [
            self.stats['top_k_hit_rates']['top_1'] * 100,
            self.stats['top_k_hit_rates']['top_3'] * 100,
            self.stats['top_k_hit_rates']['top_5'] * 100,
            self.stats['top_k_hit_rates']['top_10'] * 100
        ]
        
        bars = ax.bar(scenarios, success_rates, color=colors['gradient'], alpha=0.8)
        
        # 添加改进标注
        base_rate = success_rates[0]
        for i, (bar, rate) in enumerate(zip(bars[1:], success_rates[1:]), 1):
            improvement = ((rate - base_rate) / base_rate * 100) if base_rate > 0 else 0
            height = bar.get_height()
            
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            if improvement > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height/2,
                       f'+{improvement:.1f}%', ha='center', va='center',
                       fontweight='bold', color='white',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor=colors['accent'], alpha=0.8))
        
        ax.text(bars[0].get_x() + bars[0].get_width()/2., bars[0].get_height() + 1,
               f'{success_rates[0]:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('成功率 (%)', fontsize=14, fontweight='bold')
        ax.set_title('🚀 多候选策略效果对比', fontsize=16, fontweight='bold')
        ax.set_ylim(0, max(success_rates) * 1.15)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/04_improvement_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_report(self):
        """生成分析报告"""
        self.log("📝 生成分析报告...")
        
        report_path = f"{self.output_dir}/analysis_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("🎯 高级预测结果分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"📊 数据概况:\n")
            f.write(f"  总文件数: {self.stats['total_files']}\n")
            f.write(f"  有效文件数: {self.stats['valid_files']}\n\n")
            
            f.write(f"🎯 序列级准确率:\n")
            f.write(f"  声音模型: {self.stats['sound_sequence_accuracy']:.2%}\n")
            f.write(f"  纯Seq2Seq: {self.stats['seq2seq_sequence_accuracy']:.2%}\n")
            f.write(f"  高级模型: {self.stats['advanced_sequence_accuracy']:.2%}\n\n")
            
            f.write(f"📊 字符级准确率:\n")
            f.write(f"  声音模型: {self.stats['sound_char_accuracy']:.2%}\n")
            f.write(f"  纯Seq2Seq: {self.stats['seq2seq_char_accuracy']:.2%}\n")
            f.write(f"  高级模型: {self.stats['advanced_char_accuracy']:.2%}\n\n")
            
            f.write(f"🚀 Top-K性能:\n")
            for k in [1, 3, 5, 10]:
                rate = self.stats['top_k_hit_rates'][f'top_{k}']
                f.write(f"  Top-{k}: {rate:.2%}\n")
            
            f.write(f"\n📈 性能提升:\n")
            top1 = self.stats['top_k_hit_rates']['top_1']
            top5 = self.stats['top_k_hit_rates']['top_5']
            top10 = self.stats['top_k_hit_rates']['top_10']
            
            if top1 > 0:
                imp5 = (top5 - top1) / top1 * 100
                imp10 = (top10 - top1) / top1 * 100
                f.write(f"  Top-1 vs Top-5: +{imp5:.1f}%\n")
                f.write(f"  Top-1 vs Top-10: +{imp10:.1f}%\n")
                
    def analysis_complete(self):
        """分析完成后的UI更新"""
        self.analyze_btn.config(state='normal')
        self.open_folder_btn.config(state='normal')
        
        # 显示结果摘要
        self.log("📈 分析结果摘要:")
        top1 = self.stats['top_k_hit_rates']['top_1']
        top5 = self.stats['top_k_hit_rates']['top_5']
        top10 = self.stats['top_k_hit_rates']['top_10']
        
        self.log(f"   🎯 Top-1准确率: {top1:.1%}")
        self.log(f"   🎯 Top-5准确率: {top5:.1%}")
        self.log(f"   🎯 Top-10准确率: {top10:.1%}")
        
        if top1 > 0:
            improvement = (top5 - top1) / top1 * 100
            self.log(f"   📈 多候选提升: +{improvement:.1f}%")
        
        self.log(f"📁 结果保存在: {self.output_dir}/")
        
        messagebox.showinfo("分析完成", 
                           f"✅ 分析完成!\n\n"
                           f"🎯 Top-1准确率: {top1:.1%}\n"
                           f"🎯 Top-5准确率: {top5:.1%}\n"
                           f"📁 结果已保存到: {self.output_dir}/")
        
    def open_output_folder(self):
        """打开输出文件夹"""
        if os.path.exists(self.output_dir):
            os.startfile(self.output_dir) if sys.platform == "win32" else os.system(f"open {self.output_dir}")
        else:
            messagebox.showwarning("提示", "输出文件夹不存在，请先运行分析!")

def main():
    # 检查依赖
    try:
        import tkinterdnd2
    except ImportError:
        print("错误: 缺少 tkinterdnd2 模块")
        print("请运行: pip install tkinterdnd2")
        return
        
    # 创建主窗口
    root = tkdnd.TkinterDnD.Tk()
    app = CSVAnalysisGUI(root)
    
    # 设置窗口图标和其他属性
    try:
        root.iconbitmap(default='icon.ico')  # 如果有图标文件
    except:
        pass
        
    # 运行应用
    root.mainloop()

if __name__ == "__main__":
    main()