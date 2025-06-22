# ASCP - 声学侧信道安全研究平台

## 🚀 快速开始指南

### 根据您的键盘品牌快速测试：

| 如果您要测试... | 模型目录输入 | 测试目录输入 |
|----------------|--------------|--------------|
| **HP笔记本** | `hp_models` | `hp_test` |
| **Dell笔记本** | `dell_models` | `dell_test` |
| **ThinkPad/Lenovo** | `lenovo_models` | `lenovo_test` |
| **Mac机械键盘** | `mac_models` | `mac_test` |
| **MacBook Air** | `macair_models` | `macair_test` |
| **test797数据** | 留空 | `test797` |

### 三步开始测试：
1. 运行 `python start_web_interface.py`
2. 在浏览器中打开 http://localhost:5000
3. 根据上表输入对应的目录，点击"开始安全分析"

---

## 📋 项目概述

ASCP（Acoustic Side-Channel Research Platform）是一个基于机器学习的键盘声音识别研究平台。该项目旨在通过分析键盘敲击声音来识别输入内容，从而揭示键盘输入中潜在的声学侧信道安全风险，推动更安全的输入设备设计与隐私保护技术发展。

### 🎯 项目目标
- 研究键盘声音泄露的安全风险
- 评估不同环境下的键盘输入重构准确率
- 提供直观的可视化分析工具
- 推动键盘安全设计的改进

### 🎹 支持的键盘品牌
本项目已针对多个主流键盘品牌进行了优化，包括：
- HP、Dell、Lenovo 笔记本键盘
- Mac 全尺寸键盘
- MacBook Air 内置键盘（多个年份版本）

每个品牌都有专门训练的模型，以适应不同键盘的声学特征差异。

## 🌐 可视化界面使用指南

### 1. 启动Web界面

#### 方法一：使用启动脚本（推荐）
```bash
python start_web_interface.py
```

启动脚本会自动：
- ✅ 检查所有必要的依赖
- ✅ 验证项目文件完整性
- ✅ 启动Flask服务器
- ✅ 自动打开浏览器

#### 方法二：直接运行服务器
```bash
python web_server.py
```

然后手动在浏览器中访问：`http://localhost:5000`

### 2. 界面功能介绍

#### 🏠 首页（Landing Page）
- **项目介绍**：展示ASCP平台的研究目标和安全警告
- **技术指标**：显示AI技术、准确率和实时分析能力
- **开始研究**：点击进入主应用界面

#### 🔬 主应用界面

##### 配置区域
- **模型目录路径**：
  - 输入品牌专用模型目录（如：`hp_models`、`dell_models`等）
  - 留空使用默认模型（默认模型就是test797对应的MacBook Air的键盘）
- **测试数据目录**：
  - 输入对应品牌的测试目录（如：`hp_test`、`dell_test`等）
  - 或输入 `test797`（另一年份的MacBook Air数据）
- **多候选预测**：启用后会分析多个可能的输入序列，提供更全面的风险评估

##### 实时监控
- **进度条**：显示测试执行进度
- **系统日志**：实时显示处理过程和状态信息
- **状态指示器**：显示系统就绪状态

##### 结果展示
- **指标卡片**：
  - 字符重构准确率
  - 序列完全匹配率
  - 多候选命中率（如启用）
  - Top-5和Top-10准确率

- **详细分析结果**：
  - 每个测试文件的预测结果
  - 期望序列 vs 重构序列对比
  - 候选序列排名（如启用多候选）
  - 置信度分数

### 3. 使用流程

1. **准备测试数据**
   - 将WAV格式的键盘录音文件放入对应品牌的测试目录
   - 文件命名应包含实际输入的数字序列（如：`123456.wav`）

2. **配置参数**
   - **根据键盘品牌选择模型**：
     - HP键盘：输入 `hp_models`
     - Dell键盘：输入 `dell_models`
     - Lenovo键盘：输入 `lenovo_models`
     - Mac全键盘：输入 `mac_models`
     - MacBook Air：输入 `macair_models`
     - 使用默认模型：留空
   - **选择对应的测试目录**：
     - 对应品牌：输入 `品牌_test`
     - test797数据集：输入 `test797`
   - 选择是否启用多候选预测

3. **执行测试**
   - 点击"开始安全分析"按钮
   - 观察实时进度和日志输出
   - 等待分析完成

4. **查看结果**
   - 分析整体性能指标
   - 查看每个文件的详细预测结果
   - 评估声学侧信道风险等级

## 🛠️ 技术架构

### 后端架构
```
├── web_server.py           # Flask Web服务器
├── keystroke_recognition.py # 核心识别引擎
├── audio_processing.py      # 音频处理模块
├── feature_extraction.py    # 特征提取模块
├── keystroke_model.py       # 机器学习模型
├── data_manager.py         # 数据管理
└── config_manager.py       # 配置管理
```

### 前端技术
- **界面框架**：纯HTML/CSS/JavaScript
- **设计风格**：现代化响应式设计
- **交互方式**：实时流式更新，无需刷新页面
- **视觉效果**：渐变背景、动画效果、专业图表

### API接口

#### 状态检查
```
GET /api/status
```
返回系统运行状态信息

#### 模型测试
```
POST /api/test_model
Content-Type: application/json

{
  "modelDir": "models/",
  "testDir": "test/",
  "useMultiple": true,
  "numCandidates": 10
}
```
执行模型测试，返回流式响应

#### 目录验证
```
POST /api/validate_directory
Content-Type: application/json

{
  "directory": "path/to/dir",
  "type": "model"  // 或 "test"
}
```
验证目录是否包含有效文件

## 📊 核心功能

### 1. 音频处理
- 自动检测键盘敲击声音
- 智能分割按键片段
- 噪声过滤和信号增强

### 2. 特征提取
- MFCC（梅尔频率倒谱系数）
- 频谱特征分析
- 时域能量特征

### 3. 机器学习模型
- 随机森林（Random Forest）
- 梯度提升（Gradient Boosting）
- 深度学习模型支持

### 4. 多候选预测
- 生成多个可能的输入序列
- 计算各候选序列的置信度
- 评估Top-K准确率

## 🔧 安装要求

### 基础依赖
```bash
pip install flask flask-cors
pip install numpy scipy librosa
pip install scikit-learn
pip install matplotlib
```

### 可选依赖
```bash
pip install tensorflow  # 深度学习支持
pip install keras      # 神经网络模型
```

## 📁 项目结构

```
ASCP/
├── model_test_interface.html  # Web界面
├── web_server.py             # Flask服务器
├── start_web_interface.py    # 启动脚本
├── keystroke_recognition.py  # 核心算法
├── config.json              # 配置文件
├── models/                  # 默认模型目录
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   └── scaler.pkl
├── hp_models/               # HP键盘专用模型
├── hp_test/                 # HP键盘测试数据
├── dell_models/             # Dell键盘专用模型
├── dell_test/               # Dell键盘测试数据
├── lenovo_models/           # Lenovo键盘专用模型
├── lenovo_test/             # Lenovo键盘测试数据
├── mac_models/              # Mac全键盘专用模型
├── mac_test/                # Mac全键盘测试数据
├── macair_models/           # MacBook Air专用模型
├── macair_test/             # MacBook Air测试数据
└── test797/                 # 另一年份MacBook Air测试数据
    ├── 123456.wav
    ├── 987654.wav
    └── ...
```

## ⚠️ 安全声明

本项目仅用于安全研究和教育目的，旨在：
- 提高对键盘声音泄露风险的认知
- 推动更安全的输入设备设计
- 促进隐私保护技术的发展

**请勿将本项目用于任何非法或未经授权的活动。**

## 🎹 键盘品牌数据说明

本项目包含多个键盘品牌的专用模型和测试数据，每个品牌都有独立训练的模型以提高识别准确率：

### 品牌目录对照表

| 键盘品牌 | 模型目录 | 测试目录 | 说明 |
|---------|----------|----------|------|
| HP | `hp_models` | `hp_test` | HP品牌键盘 |
| Dell | `dell_models` | `dell_test` | Dell品牌键盘 |
| Lenovo | `lenovo_models` | `lenovo_test` | Lenovo/ThinkPad键盘 |
| Mac全键盘 | `mac_models` | `mac_test` | Mac外接全尺寸键盘 |
| MacBook Air | `macair_models` | `macair_test` | MacBook Air内置键盘 |
| MacBook Air (另一年份) | 默认模型 | `test797` | 不同年份的MacBook Air键盘数据 |

### 🚀 快速使用参考

| 场景 | 模型目录输入 | 测试目录输入 |
|------|-------------|-------------|
| 测试HP笔记本键盘 | `hp_models` | `hp_test` |
| 测试Dell笔记本键盘 | `dell_models` | `dell_test` |
| 测试ThinkPad键盘 | `lenovo_models` | `lenovo_test` |
| 测试Mac机械键盘 | `mac_models` | `mac_test` |
| 测试MacBook Air | `macair_models` | `macair_test` |
| 测试test797数据集 | 留空 | `test797` |
| 测试未知品牌键盘 | 留空 | 您的测试目录 |

### 使用示例

#### 1. 测试HP键盘录音
- **模型目录路径**：`hp_models`
- **测试数据目录**：`hp_test`

#### 2. 测试MacBook Air键盘录音（使用专用模型）
- **模型目录路径**：`macair_models`
- **测试数据目录**：`macair_test`

#### 3. 测试另一年份的MacBook Air（test797）
- **模型目录路径**：留空（使用默认模型）
- **测试数据目录**：`test797`

### 📌 重要说明
- 不同品牌和型号的键盘具有不同的声学特征
- 使用对应品牌的专用模型可以获得更高的识别准确率
- 如果测试数据来自未知键盘，建议先尝试默认模型
- `test797` 是另一年份MacBook Air的测试数据集，使用默认模型效果较好
- **注意**：部分目录命名可能不够直观（如`mac`指全键盘而非MacBook），请参考上表确认

## 🚀 快速开始

1. **克隆项目**
   ```bash
   git clone [项目地址]
   cd ASCP
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **准备数据**
   - 录制键盘声音样本用于训练
   - 准备测试音频文件

4. **训练模型**（如需要）
   ```bash
   python main_enhanced.py
   # 选择 3 - 训练模型
   ```

5. **启动Web界面**
   ```bash
   python start_web_interface.py
   ```

6. **开始测试**
   - 在浏览器中访问 http://localhost:5000
   - 根据键盘品牌选择对应的模型和测试目录
   - 配置参数并点击开始分析

## 📈 性能指标

典型测试结果：
- **字符准确率**：85-95%
- **序列匹配率**：70-85%
- **Top-5准确率**：90-98%
- **处理速度**：实时分析

### 不同键盘品牌的性能差异
- **MacBook Air**：通常具有最高的识别率（键盘结构统一）
- **Mac全键盘**：机械结构声音特征明显，识别率较高
- **HP/Dell/Lenovo**：因型号差异，识别率可能有所变化
- 使用对应品牌的专用模型可显著提升准确率

## ❓ 常见问题

### Q: 如何选择正确的模型？
**A:** 根据您测试数据的键盘品牌选择：
- 如果是HP笔记本键盘录音，使用 `hp_models`
- 如果是MacBook Air键盘录音，使用 `macair_models`
- 如果不确定键盘品牌，可以先尝试留空使用默认模型

### Q: test797 是什么？
**A:** test797 是另一个年份的MacBook Air键盘测试数据集。由于键盘设计在不同年份可能有细微变化，这个数据集使用默认模型效果更好。具体地，其意为 800 个音频文件中筛选出的 797 个音频。

### Q: 为什么要为不同品牌训练不同的模型？
**A:** 不同品牌和型号的键盘在按键机械结构、材质、间距等方面存在差异，这些差异会反映在声音特征上。使用专门训练的模型可以更准确地识别特定键盘的声音模式。

### Q: 如果我的键盘品牌不在列表中怎么办？
**A:** 您可以：
1. 先尝试使用默认模型（留空模型路径）
2. 如果效果不理想，可以收集该键盘的训练数据，训练新的专用模型
3. 尝试使用声学特征相似品牌的模型（如薄膜键盘可以互相尝试）

### Q: 目录命名有什么规律？
**A:** 目录命名遵循 `品牌_models` 和 `品牌_test` 的格式，但有一些特殊情况：
- `mac` 指的是Mac全尺寸外接键盘（不是MacBook）
- `macair` 指的是MacBook Air内置键盘
- `test797` 是特殊的测试集，实际上也是MacBook Air的数据（不同年份）

### Q: 为什么test797使用默认模型？
**A:** test797数据集来自不同年份的MacBook Air，由于键盘设计的细微变化，使用通用的默认模型反而能获得更好的泛化效果。

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出改进建议。请确保：
- 遵守项目的安全研究目的
- 提供清晰的问题描述或改进说明
- 遵循现有的代码风格

### 添加新键盘品牌支持
如果您想为新的键盘品牌添加支持：
1. 收集该品牌键盘的训练数据（建议至少100个样本）
2. 创建 `品牌_models` 和 `品牌_test` 目录
3. 使用主程序训练专用模型
4. 在文档中更新品牌列表

## 📄 许可证

本项目仅供学术研究和安全教育使用。使用者需自行承担相关责任。

---

**注意**：本项目的目的是提高对键盘声学侧信道攻击的认识，推动更安全的输入设备设计。请勿用于任何非法或未经授权的活动。
