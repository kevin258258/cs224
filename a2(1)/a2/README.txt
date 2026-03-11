欢迎来到作业 2！

本次作业将使用 PyTorch。如果你不熟悉 PyTorch，或想复习 PyTorch 基础内容，可以在 Canvas 的 Course Videos 下查看 PyTorch 复习课视频。

如果你想继续使用作业 1 的 `cs224n` 环境，请确保已安装 `local_env.yml` 中列出的全部依赖。可按以下步骤执行：

# 1. 激活旧环境：

    conda activate cs224n

# 2. 安装 docopt

    conda install docopt

# 3. 安装 pytorch、torchvision 和 tqdm

    conda install pytorch torchvision -c pytorch
    conda install -c anaconda tqdm


如果你希望为本次作业新建一个环境，请执行：

# 1. 使用 `local_env.yml` 中的依赖创建环境（根据电脑性能，这一步可能需要一些时间）：

    conda env create -f local_env.yml

# 2. 激活新环境：

    conda activate cs224n_a2


# 如需退出当前已激活的环境，请使用

    conda deactivate

## Windows 下没有 `zip` 命令
如果你使用 Windows，在运行 `collect_submission.sh` 时可能会报错。这是因为 Windows 默认不提供 `zip` 命令。遇到该错误时，你可以尝试[这里](https://superuser.com/questions/201371/create-zip-folder-from-the-command-line-windows)提到的其他方法；或者手动把 Python 文件打包为 zip。
