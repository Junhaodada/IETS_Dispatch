# Dispatch paper复现

> Real-time dispatch of integrated electricity and thermal system incorporating storages via a stochastic dynamic programming with imitation learning

环境配置


```bash
# pip env
python -m venv .venv
.venv/Scripts/activate.bat
pip install requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

# conda env
conda create -n opt_env python=3.7
conda config --set show_channel_urls yes

# .condarc 中添加以下内容
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  deepmodeling: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/
conda clean -i
pip install requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
```

源码结构

- src/
  - ADP.py
    - 包含算法1和算法2
  - MILP.py
    - 包含整数规划求解函数
  - utils.py
    - 工具函数



参考资料

- [cplex教程](https://www.bilibili.com/video/BV1ot411X79Z/)

- [cplex安装包](https://ibm.ent.box.com/s/wjuh81fmorssmwwoe4eg2spfl9xrakvn)
