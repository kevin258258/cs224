# cs224n-A3

## 安装

首先，使用 Python 3.10 创建一个新的 conda 环境：

```bash
conda create -n cs224n-A3 python=3.10
```

激活环境：

```bash
conda activate cs224n-A3
```

使用 pip 安装所有依赖：

```bash
pip install -r requirements.txt
```

## 运行测试

运行测试前请先进入 `tests/` 目录：

```bash
cd tests
pytest
```

如果你只想运行某个特定测试（例如 `test_forward`）：

```bash
pytest test_student.py::test_forward
```
