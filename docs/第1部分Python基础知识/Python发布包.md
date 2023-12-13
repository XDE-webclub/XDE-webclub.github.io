---
sidebar_position: 24
title: Python发布包(选修)
---

## Python发布包

### 打包

这里Python 3.12 以前的老项目可以使用distutils模块，更推荐使用setuptools模块，setuptools最常用的功能有：

- 依赖包安装与版本管理
- python库的打包分发
- c/c++ 拓展
- python环境限制与生成脚本

整个打包过程最重要的就是__setup.py__，它指定了重要的配置信息。setup.py的内容如下(示例)：

```python
from setuptools import setup,Extension

setup(
    ext_modules=[
    Extension(
    name = 'spam', # 包名称
    sources=['spammodule.cpp'],
    )]
)
```

通过setup函数的这些参数packages、include_package_data（其实就是MANIFEST.in文件）、exclude_package_data、package_data、data_files来指定需要打包的文件。

包含的文件如下：

- py_modules 和 packages 参数中所有 Python 源文件
- ext_modules or libraries 参数中提到的所有 C 源文件
- scripts 参数指定的脚本
- package_data 和 data_files 参数指定的所有文件
- setup.cfg 和 setup.py
- 类似于readme的文件（如README、README.txt、 README.rst、README.md）
- MANIFEST.in 中指定的所有文件（当运行python setup.py sdist时，会查阅MANIFEST.in文件，并且将里面约定的文件打包到最后的包里。什么要，什么不要）

打包命令说明：

1. 源码包source dist（简称sdist）：就是我们熟悉的 .zip 、.tar.gz 等后缀文件。就是一个压缩包，里面包含了所需要的的所有源码文件以及一些静态文件（txt文本、css、图片等）。

```python
python setup.py sdist --formats=gztar
```

2. 二进制包binary dist（简称bdist）：格式是wheel（.whl后缀），它的前身是egg。wheel本质也还是一个压缩包，可以像像zip一样解压缩。与源码包相比，二进制包的特点是不用再编译，也就是安装更快！在使用wheel之前，需要先安装wheel模块

```python
# 先安装wheel模块
pip install wheel

python setup.py bdist --formats=rpm
# 等价于
python setup.py build_rpm
```

3. 开发方式安装包，该命名不会真正的安装包，而是在系统环境中创建一个软链接指向包实际所在目录。这边在修改包之后不用再安装就能生效，便于调试。

```python
pip install -e .
等价于
python setup.py develop
```

4. 构建扩展，如用 C/C++, Cython 等编写的扩展，在调试时通常加 --inplace 参数，表示原地编译，即生成的扩展与源文件在同样的位置。

```python
python setup.py build_ext --inplace
```

5. 构建一个 wheel 分发包，egg 包是过时的，whl 包是新的标准

```python
python setup.py bdist_wheel
```

6. 构建一个 egg 分发包，经常用来替代基于 bdist 生成的模式

```python
python setup.py bdist_egg
```

7. 安装到库

```python
python setup.py install
#等价于
python setup.py build
python setup.py install

#python setup.py install包括两步：python setup.py build python setup.py install。
#这两步可分开执行， 也可只执行python setup.py install, 因为python setup.py install总是会先build后install.


#根据生成的文件等价于
pip install  xxx.zip
# 或
pip install xxx.whl
# 或.... xxx.egg
```

### 发布

如果我们需要包被全世界的同好通过 pip install 直接安装的话，需要将包上传到 pypi 网站。首先注册 pypi，获得用户名和密码。

上传 tar 包

`python setup.py sdist upload`

上传 whl 包

`python setup.py bdist_wheel upload`

如果要更安全和方便地上传包就使用 twine 上传。

安装 twine

`pip install twine`

上传所有包

`twine upload dist/*`

如果嫌每次输入用户名和密码麻烦可以配置到文件中。

编辑用户目录下的 .pypirc 文件，输入

```bash
[pypi]
username=your_username
password=your_password
```
