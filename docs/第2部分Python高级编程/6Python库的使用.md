
## Python库的使用

库、包、模块的包含关系为：多个模块组成为包、多个包组成为库。

在实际开发中不做严格区分。

### 内置库os

Python的内置os模块，是与操作系统进行交互的模块，主要有如下功能：





#### 文件路径操作
- os.remove(path) 或 os.unlink(path) ：删除指定路径的文件。路径可以是全名，也可以是当前工作目录下的路径。
- os.removedirs：删除文件，并删除中间路径中的空文件夹
- os.chdir(path)：将当前工作目录改变为指定的路径
- os.getcwd()：返回当前的工作目录
- os.curdir：表示当前目录的符号
- os.rename(old, new)：重命名文件
- os.renames(old, new)：重命名文件，如果中间路径的文件夹不存在，则创建文件夹
- os.listdir(path)：返回给定目录下的所有文件夹和文件名，不包括 '.' 和 '..' 以及子文件夹下的目录。（'.' 和 '..' 分别指当前目录和父目录）
- os.mkdir(name)：产生新文件夹
- os.makedirs(name)：产生新文件夹，如果中间路径的文件夹不存在，则创建文件夹


导入该模块：


```python 
import os
```


产生文件：


```python 
f = open('test.file', 'w')
f.close()
print('test.file' in os.listdir(os.curdir))
```


重命名文件:


```python 
os.rename("test.file", "test.new.file")
print("test.file" in os.listdir(os.curdir))
print("test.new.file" in os.listdir(os.curdir))
```


```python 
# 删除文件
os.remove("test.new.file")
```



#### 系统常量
- windows 为 \r\n
- unix为 \n




```python 
os.linesep
```


```python 
# 当前操作系统的路径分隔符：
os.sep
```


当前操作系统的环境变量中的分隔符（';' 或 ':'）：
- windows 为 ;
- unix 为:



```python 
os.pathsep
```


os.environ 是一个存储所有环境变量的值的字典，可以修改。


```python 
os.environ
```


#### os.path 模块


```python 
import os.path
```


- os.path.isfile(path) ：检测一个路径是否为普通文件
- os.path.isdir(path)：检测一个路径是否为文件夹
- os.path.exists(path)：检测路径是否存在
- os.path.isabs(path)：检测路径是否为绝对路径

windows系统：


```python 
print(os.path.isfile("C:/Windows"))
print(os.path.isdir("C:/Windows"))
print(os.path.exists("C:/Windows"))
print(os.path.isabs("C:/Windows"))
```


unix系统：


```python 
print(os.path.isfile("/Users"))
print(os.path.isdir("/Users"))
print(os.path.exists("/Users"))
print(os.path.isabs("/Users"))
```


#### split 和 join
- os.path.split(path)：拆分一个路径为 (head, tail) 两部分
- os.path.join(a, *p)：使用系统的路径分隔符，将各个部分合成一个路径


```python 
head, tail = os.path.split("c:/tem/b.txt")
print(head, tail)
```


```python 
a = "c:/tem"
b = "b.txt"
os.path.join(a, b)
```


```python 
def get_files(dir_path):
    '''
    列出文件夹下的所有文件
    :param dir_path: 父文件夹路径
    :return: 
    '''
    for parent, dirname, filenames in os.walk(dir_path):
        for filename in filenames:
            print("parent is:", parent)
            print("filename is:", filename)
            print("full name of the file is:", os.path.join(parent, filename))
```


列出当前文件夹的所有文件：


```python 
dir = os.curdir
get_files(dir)
```


#### Byte Code 编译
Python, Java 等语言先将代码编译为 byte code（不是机器码），然后再处理：
> .py -> .pyc -> interpreter

eval(statement, glob, local)

使用 eval 函数动态执行代码，返回执行的值。

exec(statement, glob, local)

使用 exec 可以添加修改原有的变量:



```python 
a = 1
exec('b = a + 10')
print(b)
```


```python 
local = dict(a=2)
glob = {}
exec("b = a+1", glob, local)

print(local)
```


compile 函数生成 byte code：
compile(str, filename, mode)


```python 
a = 1
b = compile('a+2', '', 'eval')
print(eval(b))
```


```python 
a = 1
c = compile("b=a+4", "", 'exec')
exec(c)
print(b)
```


```python 
# abstract syntax trees
import ast

tree = ast.parse('a+10', '', 'eval')
ast.dump(tree)
```


```python 
a = 1
c = compile(tree, '', 'eval')
d = eval(c)
print(d)
```


```python 
# 安全的使用方法 literal_eval ，只支持基本值的操作：
b = ast.literal_eval('[10.0, 2, True, "foo"]')
print(b)
```


### 第三方库

#### 第三方模块使用的基本流程

第三方模块使用的基本流程 以opencv为例
 
- 下载 pip install opencv-python
- 导入 import cv2
- 使用 模块名.方法名 示例 ： cv2.imread('./img/cat.jpg')

对于复杂的模块来说，使用help()方法、dir()方法不能很好的满足我们的需求。如果是新手需要搭配官方文档，查阅使用实例。

这里需要注意的是：opencv模块的下载名、导入名均不是opencv。

事实上模块名、下载名与导入名也并非一种强制的规则。

建议在下载模块之前先通过搜索引擎搜索。

更多是后续的开发者出于习惯会将名称统一。例子是pandas模块。

- 下载 pip install pandas
- 导入 import pandas
- 使用 模块名.方法名 示例 ： pandas.read_csv("./cat.csv")

在国内下载模块往往较慢，我们可以通过豆瓣、清华镜像站下载第三方模块。以下载scikit-learn模块为例

- python -m pip install scikit-learn==1.3.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

粘贴至终端，windows电脑可以通过win+R 输入CMD

MAC可以直接搜索终端打开。

#### 第三方模块的版本问题

第三方模块与系统模块一样，都是自定义好的一系列模块，这些模块也自然存在一些版本差异。

在使用的过程之中很可能因为版本的不匹配、方法的弃用导致示例的代码失效。

我们可以通过3个方式来解决：

1.升级至最新版本或安装指定的版本

- 安装指定的版本示例: pip install pandas==2.0.2
- 升级至最新版本示例: pip install --upgrade pandas

2.积极的查询官方文档。可在 https://pypi.org/ 上搜索对应模块，知名度较高的模块都会有系统的官方文档。

3.更换其他模块


#### 第三方模块OpenCV 


```python 
# 导入必要的包
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# 导入opencv
import cv2

# 使用opencv的imread方法，打开图片
img = cv2.imread('./img/cat.jpg')
# 检查类型，会发现自动转成了Numpy 数组的形式
type(img)
img

# 如果打开一张不存在的图片，不会报错，但是会返回空类型
img_wrong = cv2.imread('./img/wrong.jpg')
type(img_wrong)
img_wrong

plt.imshow(img)
# 为什么会显示的这么奇怪？

# （OpenCV和matplotlib 默认的RBG顺序不一样）
# matplotlib: R G B
# opencv: B G R
# 需要调整顺序

# 将OpenCV BGR 转换成RGB，cv2.COLOR_可以看到更多转换形式
img_fixed = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# 算法参考：RGB取均值、RGB按阈值取值、按色彩心理学公式取值R*0.299 + G*0.587 + B*0.114 = Gray

plt.imshow(img_fixed)
# 显示正常了

# 另外，我们再读取图片时也可以以灰度模式读取
img_gray = cv2.imread('./img/cat.jpg',cv2.IMREAD_GRAYSCALE)
# 显示这个灰度图
plt.imshow(img_gray,cmap="gray")

# 使用resize缩放（打开函数帮助）
img_resize = cv2.resize(img_fixed,(1000,300))
# 显示缩放后的图片
plt.imshow(img_resize)

# 翻转图片：0表示垂直翻转、1表示水平翻转，-1表示水平垂直都翻转
img_flip = cv2.flip(img_fixed,-1)

plt.imshow(img_flip)
```




### Python调用C

Python的底层是C写的（实际上大部分高级编程语言都是C写的）因此互相调用的逻辑主要是：数据类型转换、编译库的链接、接收返回值。

这个过程涉及到反复的调试，所以先从调试开始讲。

#### Visual Studio Code 和 Visual Studio的调试

##### Visual Studio Code

先看我们熟悉的Visual Studio Code ，以下简称VScode

点击“行号”前的位置，就可以给代码行打上红色的“断点”。

```Python
def mynameis(x):
    print('my name is ',end='')
    print(x,end='')# 断点
    print("!")


print(1)# 断点
mynameis('a')
print(2)# 断点
mynameis('b')
print(3)
```

接着点击刚刚的调试按钮，点击运行和调试，接着根据你的文件类型选择，譬如py文件就选择Python File. 然后可以看到代码上方有6个按钮。他们分别是：

> 1、continue（继续）
> 执行到下一断点，如果函数内容的子函数也有断点，会跳到子函数的断点处

> 2、step over（单步跳过）
> 一行一行的往下走，把没有断点的子函数当作一步，如果子函数内有断点，会跳到子函数的断点处，从断点处开始一行一行执行

> 3、step into（单步调试/单步执行）
> 一行一行往下走，如果这一行上有子函数，且无论子函数内有无断点，都会跳到子函数的第一行，从第一行开始，一行一行执行

> 4、step out（单步跳出）
> 执行到下一断点，如果遇到子函数，且子函数内没有断点，直接跳出子函数。如果子函数内有断点，会在执行完断点后再跳出子函数

> 5、Restart（重启）
> 从头开始，重新运行调试代码

> 6、stop（停止）
> 停止运行调试代码

接着打上断点，感受一下这几个按钮的功能吧。

##### Visual Studio

都是微软开发的软件，大同小异。

```C
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <windows.h>
using namespace std;

#include "test.h"


//定义一个全局结构体,作用域到文件末尾
struct Person {
    int age;
    char* name;
};

void test20() {
    //使用全局的结构体定义结构体变量p
    char x[] = "我是谁";
    struct Person p = { 10 ,x };

    printf("%d,%s\n", p.age, p.name);
}

int main(int argc, const char* argv[])
{
    //定义局部结构体名为Person,会屏蔽全局结构体
    //局部结构体作用域,从定义开始到“}”块结束
    struct Person {
        int age;
    };
    // 使用局部结构体类型
    struct Person pp;
    pp.age = 50;
    //pp.name = "zbz"; 会报错，没有name

    test20(); // 10 , 我是谁

    int a = 1;
    return 0;
}
```

我们先在红色区域（数字1）打上断点

再在绿色区域（数字2）点击调试

最后蓝色区域找到这个6个按钮

前面2个分别是stop（停止）和Restart（重启）

后面的1、2、3、4则依次对应着：continue（继续）、step over（单步跳过）、step into（单步调试/单步执行）和step out（单步跳出）

#### 代码的互相调用

##### 在Python中调用C（原生的Python.h）

python+c/c++混合编程如：

> 原生的Python.h

> cython

> pybind11：pytorch也采用该方法

> ctypes、cffi、SWIG、Boost.Pytho 等

但不论是哪个方法，大致的流程都是：转换数据类型->编译代码->生成编译后的文件（.pyd .pyc .pyo .so .dll 等）

```language
冷知识：

python的import不止能导入.py后缀结尾的文件

pyc是由py文件经过编译后生成的二进制文件，py文件变成pyc文件后，加载的速度有所提高，并且可以实现源码隐藏。

pyo是优化编译后的程序，也可以提高加载速度，针对嵌入式系统，把需要的模块编译成pyo文件可以减少容量。

.so和.dll分别是Linux和window的动态库

这些都可以被import导入，所以我们只需要编译C代码，然后import导入即可。

```

##### 环境设置

- 首先我们找到python的安装路径，通过文件搜索找到Python.h的文件夹路径
- 【设我的Python路径为C:\Python】
- 那么Python.h的文件位置就是：C:\Python\include 简称H路径
- python310_d.lib的位置就是：C:\Python\libs 简称L路径
- 接着右击【项目】，点击属性

- 最后在上方选择所有配置、所有平台。点击VC++目录，选择包含目录最右边的下拉三角，输入刚刚复制的**H路径**即可

- 接着再来载入python310_d.lib库，打开L路径查看里面有无python310_d.lib这个文件，【注意，310是python版本号，不同版本对应不同文件名】如果没有，则复制python310.lib，然后重命名。
- 还是打开刚刚的属性，依次设置。
- 库目录填【文件夹路径】

- 附加依赖项填【文件路径】

##### 代码编写

- 新建一个文件名，根据官方文档的说法，以C语言为例，如果一个模块叫 spam，则对应实现它的文件名叫 spammodule.c；如果这个模块名字非常长，比如 spammify，则这个模块的文件可以直接叫 spammify.c

这里我调整了一下官方文档给的示例，添加了一些注释。让新手更易读。

当然原生的方法总是最底层但是最麻烦的方法，如果使用诸如Python中的ctypes模块则流程会简化。此处可以查阅相关文档。

```C
#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject* spam_system(PyObject* self, PyObject* args)
{
    /*
     self 参数指向模块对象；对于方法则指向对象实例。

     args 参数是指向一个 Python 的 tuple 对象的指针，其中包含参数。
     每个 tuple 项对应一个调用参数。 这些参数也全都是 Python 对象
     要在我们的 C 函数中使用它们就需要先  将其转换为 C 值。
    */

    const char* command;
    int sts;
    //PyArg_ParseTuple() 会检查参数类型并将其转换为 C 值。 
    //它使用模板字符串确定需要的参数类型以及存储被转换的值的 C 变量类型。
    //在所有参数都有正确类型且组成部分按顺序放在传递进来的地址里时，返回真(非零)。
    //其在传入无效参数时返回假(零)。在后续例子里，还会抛出特定异常，使得调用的函数可以理解返回 NULL(也就是例子里所见)。
    // "s" 是一个参数，将 Unicode 对象转换为指向字符串的 C 指针。具体可以参考 https://docs.python.org/3/c-api/arg.html
    if (PyArg_ParseTuple(args, "s", &command)) {

        // system 是C的库函数，从属于stdlib标准库,【片面】的说：
        // 返回值是0表示成功 
        // 返回值是其他表示执行失败
        // 至于为什么是片面的，原因会在下个阶段解释。
        sts = system(command);

        //PyLong_FromLong返回一个表示 Python 整数对象的 PyObject 子类型。
        return PyLong_FromLong(sts);
    }
    else {
        return NULL;
    }
}

// 构造方法
static PyMethodDef SpamMethods[] = {
    {"system",  spam_system, METH_VARARGS,
     "Execute a shell command."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

// 调用构造方法
static struct PyModuleDef spammodule = {
    PyModuleDef_HEAD_INIT,
    "spam",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SpamMethods
};
// 初始化
PyMODINIT_FUNC
PyInit_spam(void)
{
    return PyModule_Create(&spammodule);
}

int
main(int argc, char* argv[])
{
    wchar_t* program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }
    /* Add a built-in module, before Py_Initialize */
    if (PyImport_AppendInittab("spam", PyInit_spam) == -1) {
        fprintf(stderr, "Error: could not extend in-built modules table\n");
        exit(1);
    }
    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(program);
    /* Initialize the Python interpreter.  Required.
       If this step fails, it will be a fatal error. */
    Py_Initialize();
    /* Optionally import the module; alternatively,
       import can be deferred until the embedded script
       imports it. */
    PyObject* pmodule = PyImport_ImportModule("spam");
    if (!pmodule) {
        PyErr_Print();
        fprintf(stderr, "Error: could not import module 'spam'\n");
    }

    PyMem_RawFree(program);
    return 0;
}

```


### Python发布包

#### 打包

这里Python 3.12 以前的老项目可以使用distutils模块，更推荐使用setuptools模块，setuptools最常用的功能有：

- 依赖包安装与版本管理
- python库的打包分发
- c/c++ 拓展
- python环境限制与生成脚本

整个打包过程最重要的就是**setup.py**，它指定了重要的配置信息。setup.py的内容如下(示例)：

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
#### 发布

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


