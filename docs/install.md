# 安装指南

## 环境配置

pip环境：

```bash
# pip env
python -m venv .venv
.venv/Scripts/activate.bat

pip install requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
```

conda环境：

```bash
# conda env
conda create -n opt_env python=3.9

pip install requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
```

## 算法运行

```bash
cd src
python ALGO.py
```



## 参考资料

- [cplex教程](https://www.bilibili.com/video/BV1ot411X79Z/) | [docplex教程](https://brucehan.top/2020/02/02/docplex/)
- [cplex安装包](https://ibm.ent.box.com/s/wjuh81fmorssmwwoe4eg2spfl9xrakvn)
