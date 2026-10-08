# GitHub 发布检查记录

检查日期：2026-10-08。GitHub 仓库为 https://github.com/Gucci233/FlexPart；本记录覆盖代码发布检查，不修改 Hugging Face 仓库。

## 已修复

| 问题 | 修改 |
| --- | --- |
| README 缺少项目效果展示 | 从论文 arXiv 版本的原图 PDF 导出总览图、架构图及人工点/框提示的复杂拓扑生成结果图，保存在 `assets/` 并添加准确图注 |
| 权重发布状态与 README 不一致 | 添加 Hugging Face 链接并说明当前发布的是 transformer 权重 |
| Hugging Face 文件不能直接组成 pipeline | 新增 `scripts/prepare_weights.py`，装配 TripoSG 的 VAE、DINOv2、图像处理器和 scheduler；新增推理配置并在装配时检查所有权重名称和形状 |
| VAE 导入不存在的模块 | 从 PartCrafter 提取原始、无提示调制的 `DiTBlock`，保留上游许可证和版权说明；不能直接替换成 FlexPart 的条件 DiTBlock |
| transformer 包初始化导入不存在的类 | 改为导出当前实现的 `FlexPartDiTModel` |
| 代码包含作者机器的绝对路径 | 所有 `/mnt/...` 和 `/data/gqw/...` 路径改为相对路径或命令参数；脚本导入路径根据 `__file__` 定位项目根目录 |
| 推理忽略用户提示文件、读取固定测试数据 | 移除固定测试数据，读取用户传入的数值 NumPy 数组 |
| README 中的模型路径和 tag 参数不存在 | 推理入口新增 `--model_path`、`--tag` 等参数并同步文档 |
| 点提示错误激活框条件，未提供 mask 时尝试转换 None | 正确设置有效条件标记；无 mask 时直接传 None |
| 提示尺寸、数量、坐标无校验 | 校验数组形状、数值、部件数、图像边界、空 mask；读取提示时禁用 pickle |
| 移除背景的调用类型错误，并可能使图像与提示坐标错位 | 改为保留原始画布尺寸的背景移除；仅在 `--rmbg` 时加载 RMBG；修正 demo 对已 sigmoid 输出再次 sigmoid 的处理 |
| 提取失败时导出伪造退化网格 | 对空部件明确报错 |
| 预处理标注要求并未生成的 `render_OBB.png` | 将 OBB 可视化设为可选；补存评估所需的 `normalized_rotated_scene.glb` |
| 预处理扫描子目录但只在根目录查找原 mesh | 支持唯一 stem 的递归查找；重复 stem 明确报错；修复文件名含点时的截断 |
| 断点续跑漏查关键输出、标注顺序不稳定 | 检查关键输出并将最终标注按来源文件排序 |
| 旧预处理脚本不使用 `--input` 扫描 mesh | 使用用户输入目录，延迟加载 RMBG |
| 评估入口使用本机或空 JSON 路径 | 新增必填的标注路径和生成结果目录参数 |
| 训练启动器覆盖 W&B key、参数未引用 | 移除 key 覆盖，引用参数，允许环境变量配置 GPU 数量，定位到仓库根目录 |
| 从零训练分支重复初始化提示模块 | 构造时禁用自动添加，再显式添加一次 |
| 依赖表遗漏直接导入包 | 显式补充 accelerate、safetensors、pandas，删除重复条目 |
| 安装脚本缺少无 root 使用方式 | 支持 `SKIP_SYSTEM_DEPS=1`，添加失败即停止和目录定位 |
| 大权重和训练输出可能被加入 Git | 添加 `.gitignore`；本地约 6.16 GB 权重文件保留在磁盘但被忽略；添加文本换行规则 |

## 已完成验证

- 38 个 Python 文件通过 AST 语法检查。
- 仓库内部模块导入目标静态检查没有发现缺失文件。
- 两个 Shell 脚本通过 `bash -n`，检查不会安装依赖或启动训练。
- 8 项 unittest 回归测试通过，覆盖提示坐标、无效输入、缺失文件、pickle 拒绝、CLI 帮助、必需提示参数及预处理标注生成。
- README 与数据说明的本地文件和图片链接有效，代码块成对。
- 示例点提示通过原始 2048×2048 图片尺寸校验。
- 读取本地 safetensors 文件头，确认 21 个 transformer block、2048 隐藏宽度、64 latent 通道和 32 部件 embedding 容量与推理配置一致；没有将完整权重载入内存。
- Hugging Face API 确认 `gucci233/FlexPart` 为公开、非 gated 仓库，当前仅有 README、model_index 和单个 safetensors 权重等文件。
- 源码、YAML、Shell 文件中已无原作者机器的 `/mnt/...` 或 `/data/gqw/...` 路径。

运行输入回归测试：

```bash
python -m unittest discover -s tests -v
```

## 发布前仍需实测或确认

1. **完整装配与 CUDA 生成未实测。** 当前检查环境没有 torch、diffusers、accelerate 和 CUDA 推理条件，未执行装配脚本中的全部 654 个 tensor 校验、模型加载、Gradio 启动、mesh 提取和渲染。需在 Linux CUDA 环境按 README 运行装配和示例推理。
2. **推理配置的非权重设置。** 全局注意力 block 采用两份训练 YAML 中的偶数 block 设置。权重文件头无法验证这项设置、训练数据或生成质量。若发布的 checkpoint 使用其他设置，更新 `configs/flexpart_inference.json`。
3. **训练与评估的论文一致性。** 未执行两阶段训练。评估脚本保留其原有采样、阈值和 ICP 对齐逻辑；预处理会过滤可见 mask 很小的训练部件，但导出的 source scene 保留来源几何。基准发布前需核对这些设置与论文协议。
4. **依赖锁定。** requirements 中部分包仍未固定版本。完成干净环境实测后，应记录实际版本或生成锁定文件，以验证未来安装的可复现性。
5. **上游许可证。** 目录中的 MIT License 不能替代第三方代码的原始条款。恢复的 DiTBlock 保留上游 Hunyuan 许可证，另附 NOTICE；发布前还需核对已有改编代码是否完整保留其来源和条款。Hugging Face 权重当前标注 Apache-2.0，需确认其适用范围和基础模型条款。
6. **论文链接。** GitHub 仓库入口已补充。公开论文 URL 尚未提供，README 没有虚构论文编号或录用信息；公开后补上真实入口。

原论文目录只读，未修改论文源文件或 PDF。示例点是手工选择的演示输入，没有宣称对应论文结果。
