# CPIR-T2 有限近似误差模型

本目录给出 CPIR-T2 的有限、离散时间、经典 Markov 回归实例。模型采用总变差距离、Dobrushin 收缩系数和逐步核缺陷，检查几何级数误差界；另含非收缩线性界与“目标不可测”反例。

运行：

```powershell
python models/cpir_approximate/cpir_approximate.py
python -m unittest tests/test_cpir_approximate.py
```

该工件验证公式实现和反例责任，不构成自然系统的经验确认。标准收缩与有限时扰动结果来自既有 Markov/粗粒化理论；DCS 的工作是把它接入 CPIR 的类型化合同。
