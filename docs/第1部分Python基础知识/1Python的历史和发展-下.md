
### 安装Vscode


[Vscdoe官网下载网址](https://code.visualstudio.com/download)

- VSCode（Visual Studio Code）是一款通用跨平台的编辑器

它不会运行程序，它需要安装相应的语言包才能运行程序。它可以编辑任何语言的程序，支持几乎所有主流的开发语言的语法高亮、智能代码补全等。安装过程全部勾选。


- 注意与Visual Studio区别

Visual Studio是一个集成的开发环境。




#### Vscode安装插件


Vscode左侧菜单通常为：文件、搜索、源代码管理、调试、应用商店等等（不同版本显示不同）

逐一打开，找到应用商店，在其中搜索插件名称即可下载。推荐下载插件：

- 简体中文包：包名：`Chinese (Simplified) Language Pack for Visual Studio Code`
- Python包：包名：`Python`



#### 拓展：Vscode的个性化设置

- 主题颜色

在设置中点击主题颜色，可以选择自己喜欢的主题颜色。

- 保存时代码自动格式化

安装成功后可以在Vscode的设置中搜索`format on save`，勾选即可。


测试：在Vscode中新建一个xxx.py文件，输入以下内容：
```python
print("hello world")
```

运行方式1：点击右上角的三角形运行按钮

运行方式2.在编辑器中输入`python xxx.py`运行

运行方式3：在编辑器中输入`python -m xxx.py`运行

可以看到输出结果为`hello world`。

python xxx.py和python -m xxx.py是两种加载py文件的方式:
1叫做直接运行
2把模块当作脚本来启动(注意：但是__name__的值为'main' )

