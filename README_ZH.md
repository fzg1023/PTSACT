# 声学CT温度场重建：使用说明

本材料包对应稿件MEAS-D-26-12759，包含合成数据生成器、Siddon投影矩阵构建与反演代码、训练权重、模型配置和评估程序。材料按合理请求由通讯作者提供；使用及转交范围请与作者确认。本包未附开放源代码许可证。

## 包含的材料

| 内容 | 位置 |
|---|---|
| 运行入口 | `run.py` |
| 数据生成、反演、推理与指标计算 | `repro/` |
| 物理模型和网络定义 | `vendor/` |
| 训练所得模型对应的配置 | `configs/` |
| 合成数据、权重与参考结果 | `assets/` |
| 测试 | `tests/` |
| 数值验证记录 | `VALIDATION.json` |
| 来源映射与逐文件校验和 | `SOURCE_MAP.json`、`MANIFEST_SHA256.csv` |

主基准包含4800个训练样本、1200个验证样本和1200个独立生成的测试样本，以及三个训练种子的模型。另提供直接网络、双输入参考、消融、RBF和TV对照材料，以及工厂几何高温、全温域合成验证所需的数据与模型。

本包不包含真实工厂CSV、实测声路衍生表、系统界面图片、论文和审稿文件。程序支持合成数据生成与已有模型评估，不包含全部模型的从零重训流程，也不覆盖论文所有图表及真实工厂回放。

## 环境与快速运行

原数值验证使用Python 3.9.2，依赖版本见`requirements.txt`。在兼容环境中安装依赖：

```text
python -m pip install -r requirements.txt
```

进入解压后的材料包目录，依次执行：

```text
python run.py verify
python run.py normalization --output outputs/normalization_check
python run.py generate --count-per-family 3 --output outputs/generator_check
python run.py evaluate --suite main --output outputs/main_test
```

默认使用CPU。已有兼容的PyTorch/CUDA环境时，可在评估命令中加`--device cuda`。所有输出应指定为`outputs/`下尚不存在的子目录；重复运行时更换目录名。

完整主评估使用1200个测试样本和三个种子的模型。参考结果为Stage 1平均归一化RMSE约1.9153%，两阶段方法约0.5497%；两阶段三种子均值的总体标准差约0.0007个百分点。这不是样本间标准差，也不是将三个模型输出平均后计算的集成结果。

## 数据生成与其他评估

重新生成全部1200个测试样本，并导出粗细网格投影矩阵、传感器坐标和声路顺序：

```text
python run.py generate --count-per-family 400 --output outputs/generated_test
```

分别评估对照、消融和工厂几何合成数据：

```text
python run.py evaluate --suite baselines --output outputs/baselines
python run.py evaluate --suite ablations --output outputs/ablations
python run.py evaluate --suite factory --output outputs/factory
python run.py evaluate --suite all --device cuda --output outputs/all_evaluations
```

`--limit-per-family 3`仅用于少量样本的运行检查；完整评估不加此选项。TV默认使用包内1200个已收敛参考解重算指标。单独重新求解一个样本可运行：

```text
python run.py tv-solve --family single_peak --index 0 --output outputs/tv_example
```

## 不提供真值的模型推理

输入NPZ应包含`T_stage1`数组，形状为`(N,32,32)`；或包含以秒为单位的`tof`数组，形状为`(N,120)`。TOF的传感器布局和声路顺序须与本包主基准一致。两种数组同时存在时优先使用`T_stage1`。推理不要求或读取`T_true`。

```text
python run.py predict --input my_measurements.npz --seed 42 --output outputs/prediction
```

归一化与残差恢复使用模型保存的固定常数。历史准备程序扫描过训练和验证数据，但仅用训练数据重算得到相同数值；`normalization`命令核对此事实，不使用测试真值确定常数。误差评估另行读取真值。

## 指标和校验

主基准的RMSE百分比以各类真值平均温度为分母，再对峰型等权汇总；主基准及受控对照的PSNR使用固定参考温差。工厂几何合成评估保留其原有指标定义，详见`FACTORY.md`，不与真实工厂声路误差混用。

不同设备及推理批次可能引起小幅浮点差异，数值验证记录见`VALIDATION.json`。`verify`命令检查文件是否与逐文件校验和一致，压缩包另附SHA-256校验文件。运行产生的结果保存在`outputs/`，不修改包内数据和模型。
