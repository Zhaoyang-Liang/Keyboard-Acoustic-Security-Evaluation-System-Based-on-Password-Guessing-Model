# web_server.py - Flask后端服务
import os
import json
import traceback
from datetime import datetime
from collections import defaultdict
from flask import Flask, request, jsonify, render_template_string, Response
from flask_cors import CORS
import threading
import time

# 导入你的现有模块
from config_manager import ConfigManager
from keystroke_recognition import KeystrokeRecognitionSystem

app = Flask(__name__)
CORS(app)  # 允许跨域请求

class TestProgress:
    """测试进度管理类"""
    def __init__(self):
        self.progress = 0
        self.message = "准备中..."
        self.logs = []
        self.results = None
        self.error = None
        self.completed = False

    def update_progress(self, progress, message):
        self.progress = progress
        self.message = message

    def add_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")

    def set_results(self, results):
        self.results = results
        self.completed = True

    def set_error(self, error):
        self.error = error
        self.completed = True

# 全局进度跟踪器
current_test_progress = None

@app.route('/')
def index():
    """提供HTML界面"""
    # 这里我们直接返回HTML文件的内容
    # 在实际部署时，你可以将HTML文件放在templates目录中
    try:
        # 尝试读取HTML文件
        with open('model_test_interface.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # 如果文件不存在，返回一个简单的页面
        return """
        <html>
        <head><title>键盘声音识别系统</title></head>
        <body>
        <h1>键盘声音识别系统</h1>
        <p>请将 model_test_interface.html 文件放在与此服务相同的目录中。</p>
        <p>或者访问 <a href="/api/status">/api/status</a> 查看API状态。</p>
        </body>
        </html>
        """

@app.route('/api/status')
def api_status():
    """API状态检查"""
    return jsonify({
        "status": "running",
        "message": "键盘声音识别系统后端服务运行正常",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

@app.route('/api/test_model', methods=['POST'])
def test_model():
    """执行模型测试的API"""
    global current_test_progress
    
    try:
        config = request.json
        if not config:
            return jsonify({"error": "缺少配置参数"}), 400

        # 创建新的进度跟踪器
        current_test_progress = TestProgress()
        
        # 在后台线程中执行测试
        thread = threading.Thread(target=run_model_test, args=(config,))
        thread.daemon = True
        thread.start()

        # 返回流式响应
        return Response(
            generate_test_stream(),
            mimetype='application/json',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            }
        )

    except Exception as e:
        return jsonify({"error": f"启动测试失败: {str(e)}"}), 500

def generate_test_stream():
    """生成测试流式响应"""
    global current_test_progress
    
    if current_test_progress is None:
        yield json.dumps({"type": "error", "message": "测试未初始化"}) + '\n'
        return

    last_progress = -1
    last_log_count = 0

    while not current_test_progress.completed:
        # 发送进度更新
        if current_test_progress.progress != last_progress:
            yield json.dumps({
                "type": "progress",
                "progress": current_test_progress.progress,
                "message": current_test_progress.message
            }) + '\n'
            last_progress = current_test_progress.progress

        # 发送新的日志
        if len(current_test_progress.logs) > last_log_count:
            for log in current_test_progress.logs[last_log_count:]:
                yield json.dumps({
                    "type": "log",
                    "message": log
                }) + '\n'
            last_log_count = len(current_test_progress.logs)

        time.sleep(0.5)  # 避免过于频繁的更新

    # 发送最终结果
    if current_test_progress.error:
        yield json.dumps({
            "type": "error",
            "message": current_test_progress.error
        }) + '\n'
    elif current_test_progress.results:
        yield json.dumps({
            "type": "result",
            "data": current_test_progress.results
        }) + '\n'

def run_model_test(config):
    """在后台线程中运行模型测试"""
    global current_test_progress
    
    try:
        current_test_progress.add_log("开始初始化测试环境...")
        current_test_progress.update_progress(5, "初始化配置管理器...")

        # 初始化配置管理器
        config_manager = ConfigManager()
        
        # 设置模型目录（如果提供）
        original_model_dir = config_manager.get("paths.model_dir")
        if config.get('modelDir'):
            config_manager.set("paths.model_dir", config['modelDir'])
            current_test_progress.add_log(f"使用自定义模型目录: {config['modelDir']}")
        else:
            current_test_progress.add_log(f"使用默认模型目录: {original_model_dir}")

        current_test_progress.update_progress(10, "初始化识别系统...")
        
        # 创建识别系统
        system = KeystrokeRecognitionSystem(config_manager=config_manager)
        current_test_progress.add_log("识别系统初始化完成")

        # 检查测试目录
        test_dir = config.get('testDir', 'test')
        if not os.path.exists(test_dir):
            raise Exception(f"测试目录不存在: {test_dir}")

        # 获取测试文件
        wav_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]
        if not wav_files:
            raise Exception(f"测试目录中没有WAV文件: {test_dir}")

        current_test_progress.add_log(f"找到 {len(wav_files)} 个测试文件")
        current_test_progress.update_progress(15, f"准备测试 {len(wav_files)} 个文件...")

        # 执行测试
        use_multiple = config.get('useMultiple', False)
        num_candidates = config.get('numCandidates', 10)
        
        current_test_progress.add_log(f"多候选预测: {'启用' if use_multiple else '禁用'}")
        if use_multiple:
            current_test_progress.add_log(f"候选数量: {num_candidates}")

        # 测试统计
        total_chars = 0
        correct_chars = 0
        total_sequences = 0
        correct_sequences = 0
        hit_stats = defaultdict(int)
        no_hit_count = 0
        detailed_results = []

        # 处理每个文件
        for idx, filename in enumerate(wav_files):
            try:
                file_path = os.path.join(test_dir, filename)
                progress = 15 + (idx / len(wav_files)) * 70  # 15-85%的进度用于处理文件
                current_test_progress.update_progress(progress, f"测试文件: {filename}")
                current_test_progress.add_log(f"正在测试: {filename}")

                # 从文件名提取预期序列
                expected_sequence = ''.join(c for c in os.path.splitext(filename)[0] if c.isdigit())
                if not expected_sequence:
                    current_test_progress.add_log(f"警告: 无法从文件名提取预期序列: {filename}")
                    continue

                total_sequences += 1

                # 执行预测
                if use_multiple:
                    candidates = system.predict_from_file_with_candidates(file_path, num_candidates)
                    if not candidates:
                        current_test_progress.add_log(f"预测失败: {filename}")
                        continue
                    
                    predicted = candidates[0][0]
                    candidates_data = [
                        {
                            "sequence": seq,
                            "confidence": conf,
                            "is_correct": seq == expected_sequence
                        }
                        for seq, conf in candidates
                    ]
                    
                    # 多候选分析
                    hit_rank = None
                    for i, (seq, _) in enumerate(candidates):
                        if seq == expected_sequence:
                            hit_rank = i + 1
                            break
                    
                    if hit_rank:
                        hit_stats[hit_rank] += 1
                    else:
                        no_hit_count += 1
                        
                else:
                    predicted = system.predict_from_file(file_path)
                    if not predicted:
                        current_test_progress.add_log(f"预测失败: {filename}")
                        continue
                    candidates_data = []

                # 计算准确率
                min_len = min(len(expected_sequence), len(predicted))
                num_correct = sum(1 for i in range(min_len) if expected_sequence[i] == predicted[i])
                char_accuracy = num_correct / len(expected_sequence) if expected_sequence else 0
                sequence_match = (predicted == expected_sequence)

                if sequence_match:
                    correct_sequences += 1

                correct_chars += num_correct
                total_chars += len(expected_sequence)

                # 记录详细结果
                detailed_results.append({
                    "filename": filename,
                    "expected": expected_sequence,
                    "predicted": predicted,
                    "char_accuracy": char_accuracy,
                    "sequence_match": sequence_match,
                    "success": True,
                    "candidates": candidates_data
                })

                current_test_progress.add_log(f"完成: {filename} - 准确率: {char_accuracy:.2%}")

            except Exception as e:
                current_test_progress.add_log(f"处理文件 {filename} 时出错: {str(e)}")
                detailed_results.append({
                    "filename": filename,
                    "expected": expected_sequence if 'expected_sequence' in locals() else "未知",
                    "predicted": "预测失败",
                    "char_accuracy": 0,
                    "sequence_match": False,
                    "success": False,
                    "candidates": []
                })

        current_test_progress.update_progress(90, "计算最终统计...")

        # 计算最终统计
        char_accuracy = (correct_chars / total_chars * 100) if total_chars > 0 else 0
        sequence_accuracy = (correct_sequences / total_sequences * 100) if total_sequences > 0 else 0
        
        summary = {
            "char_accuracy": round(char_accuracy, 2),
            "sequence_accuracy": round(sequence_accuracy, 2),
            "total_files": len(wav_files),
            "successful_predictions": len([r for r in detailed_results if r["success"]]),
            "multiple_candidates": use_multiple
        }

        if use_multiple:
            total_hits = sum(hit_stats.values())
            top5_hits = sum(hit_stats.get(i, 0) for i in range(1, 6))
            top10_hits = sum(hit_stats.get(i, 0) for i in range(1, 11))
            
            summary.update({
                "total_hit_rate": round((total_hits / total_sequences * 100) if total_sequences > 0 else 0, 2),
                "top5_accuracy": round((top5_hits / total_sequences * 100) if total_sequences > 0 else 0, 2),
                "top10_accuracy": round((top10_hits / total_sequences * 100) if total_sequences > 0 else 0, 2),
                "hit_stats": dict(hit_stats),
                "no_hit_count": no_hit_count
            })

        results = {
            "summary": summary,
            "details": detailed_results
        }

        current_test_progress.update_progress(100, "测试完成!")
        current_test_progress.add_log("所有测试已完成")
        current_test_progress.set_results(results)

        # 恢复原始配置
        if config.get('modelDir'):
            config_manager.set("paths.model_dir", original_model_dir)

    except Exception as e:
        error_msg = f"测试执行失败: {str(e)}"
        current_test_progress.add_log(f"错误: {error_msg}")
        current_test_progress.add_log(f"详细错误: {traceback.format_exc()}")
        current_test_progress.set_error(error_msg)

@app.route('/api/list_directories')
def list_directories():
    """列出可用的目录"""
    try:
        current_dir = os.getcwd()
        directories = []
        
        for item in os.listdir(current_dir):
            if os.path.isdir(item):
                # 检查是否包含模型文件或音频文件
                files = os.listdir(item)
                has_models = any(f.endswith(('.pkl', '.h5', '.json')) for f in files)
                has_audio = any(f.endswith('.wav') for f in files)
                
                directories.append({
                    "name": item,
                    "path": os.path.abspath(item),
                    "has_models": has_models,
                    "has_audio": has_audio,
                    "file_count": len(files)
                })
        
        return jsonify({
            "current_directory": current_dir,
            "directories": directories
        })
    except Exception as e:
        return jsonify({"error": f"获取目录列表失败: {str(e)}"}), 500

@app.route('/api/validate_directory', methods=['POST'])
def validate_directory():
    """验证目录是否有效"""
    try:
        data = request.json
        directory = data.get('directory')
        dir_type = data.get('type')  # 'model' or 'test'
        
        if not directory or not os.path.exists(directory):
            return jsonify({"valid": False, "message": "目录不存在"})
        
        files = os.listdir(directory)
        
        if dir_type == 'model':
            model_files = [f for f in files if f.endswith(('.pkl', '.h5', '.json'))]
            return jsonify({
                "valid": len(model_files) > 0,
                "message": f"找到 {len(model_files)} 个模型文件" if model_files else "未找到模型文件",
                "files": model_files
            })
        elif dir_type == 'test':
            wav_files = [f for f in files if f.endswith('.wav')]
            return jsonify({
                "valid": len(wav_files) > 0,
                "message": f"找到 {len(wav_files)} 个测试文件" if wav_files else "未找到WAV文件",
                "files": wav_files[:10]  # 只返回前10个文件名
            })
        else:
            return jsonify({"valid": False, "message": "未知的目录类型"})
            
    except Exception as e:
        return jsonify({"valid": False, "message": f"验证失败: {str(e)}"})

if __name__ == '__main__':
    print("启动键盘声音识别系统Web服务...")
    print("访问 http://localhost:5000 来使用界面")
    print("API状态: http://localhost:5000/api/status")
    
    app.run(
        host='0.0.0.0',  # 允许外部访问
        port=5000,
        debug=True,
        threaded=True  # 支持多线程
    )