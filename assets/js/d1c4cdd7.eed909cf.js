"use strict";(self.webpackChunkjiangmiemie=self.webpackChunkjiangmiemie||[]).push([[3625],{9108:(n,e,s)=>{s.r(e),s.d(e,{assets:()=>o,contentTitle:()=>l,default:()=>d,frontMatter:()=>r,metadata:()=>a,toc:()=>h});var i=s(5893),t=s(1151);const r={sidebar_position:21,title:"Python\u5185\u7f6e\u5e93\u7684\u4f7f\u7528"},l=void 0,a={id:"\u7b2c2\u90e8\u5206Python\u57fa\u7840\u77e5\u8bc6/Python\u5185\u7f6e\u5e93\u7684\u4f7f\u7528",title:"Python\u5185\u7f6e\u5e93\u7684\u4f7f\u7528",description:"Python\u5185\u7f6e\u5e93\u7684\u4f7f\u7528",source:"@site/docs/\u7b2c2\u90e8\u5206Python\u57fa\u7840\u77e5\u8bc6/Python\u5185\u7f6e\u5e93\u7684\u4f7f\u7528.md",sourceDirName:"\u7b2c2\u90e8\u5206Python\u57fa\u7840\u77e5\u8bc6",slug:"/\u7b2c2\u90e8\u5206Python\u57fa\u7840\u77e5\u8bc6/Python\u5185\u7f6e\u5e93\u7684\u4f7f\u7528",permalink:"/docs/\u7b2c2\u90e8\u5206Python\u57fa\u7840\u77e5\u8bc6/Python\u5185\u7f6e\u5e93\u7684\u4f7f\u7528",draft:!1,unlisted:!1,tags:[],version:"current",sidebarPosition:21,frontMatter:{sidebar_position:21,title:"Python\u5185\u7f6e\u5e93\u7684\u4f7f\u7528"},sidebar:"tutorialSidebar",previous:{title:"\u534f\u7a0b(\u9009\u4fee)",permalink:"/docs/\u7b2c2\u90e8\u5206Python\u57fa\u7840\u77e5\u8bc6/\u534f\u7a0b"},next:{title:"\u7b2c\u4e09\u65b9\u5e93",permalink:"/docs/\u7b2c2\u90e8\u5206Python\u57fa\u7840\u77e5\u8bc6/\u7b2c\u4e09\u65b9\u5e93"}},o={},h=[{value:"Python\u5185\u7f6e\u5e93\u7684\u4f7f\u7528",id:"python\u5185\u7f6e\u5e93\u7684\u4f7f\u7528",level:2},{value:"\u6587\u4ef6\u8def\u5f84\u64cd\u4f5c",id:"\u6587\u4ef6\u8def\u5f84\u64cd\u4f5c",level:3},{value:"\u7cfb\u7edf\u5e38\u91cf",id:"\u7cfb\u7edf\u5e38\u91cf",level:3},{value:"os.path \u6a21\u5757",id:"ospath-\u6a21\u5757",level:3},{value:"split \u548c join",id:"split-\u548c-join",level:3},{value:"Byte Code \u7f16\u8bd1",id:"byte-code-\u7f16\u8bd1",level:3}];function c(n){const e={blockquote:"blockquote",code:"code",h2:"h2",h3:"h3",li:"li",p:"p",pre:"pre",ul:"ul",...(0,t.a)(),...n.components};return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(e.h2,{id:"python\u5185\u7f6e\u5e93\u7684\u4f7f\u7528",children:"Python\u5185\u7f6e\u5e93\u7684\u4f7f\u7528"}),"\n",(0,i.jsx)(e.p,{children:"\u5e93\u3001\u5305\u3001\u6a21\u5757\u7684\u5305\u542b\u5173\u7cfb\u4e3a\uff1a\u591a\u4e2a\u6a21\u5757\u7ec4\u6210\u4e3a\u5305\u3001\u591a\u4e2a\u5305\u7ec4\u6210\u4e3a\u5e93\u3002"}),"\n",(0,i.jsx)(e.p,{children:"\u5728\u5b9e\u9645\u5f00\u53d1\u4e2d\u4e0d\u505a\u4e25\u683c\u533a\u5206\u3002"}),"\n",(0,i.jsx)(e.p,{children:"Python\u6807\u51c6\u5e93\uff1aPython\u5185\u7f6e\u7684\u5e93\uff0c\u4e0d\u9700\u8981\u5b89\u88c5\uff0c\u76f4\u63a5\u5bfc\u5165\u5373\u53ef\u4f7f\u7528\u3002"}),"\n",(0,i.jsx)(e.p,{children:"\u4ee5Python\u7684\u5185\u7f6eos\u6a21\u5757\u4e3a\u4f8b\uff0c\u662f\u4e0e\u64cd\u4f5c\u7cfb\u7edf\u8fdb\u884c\u4ea4\u4e92\u7684\u6a21\u5757\uff0c\u4e3b\u8981\u6709\u5982\u4e0b\u529f\u80fd\uff1a"}),"\n",(0,i.jsx)(e.h3,{id:"\u6587\u4ef6\u8def\u5f84\u64cd\u4f5c",children:"\u6587\u4ef6\u8def\u5f84\u64cd\u4f5c"}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsx)(e.li,{children:"os.remove(path) \u6216 os.unlink(path) \uff1a\u5220\u9664\u6307\u5b9a\u8def\u5f84\u7684\u6587\u4ef6\u3002\u8def\u5f84\u53ef\u4ee5\u662f\u5168\u540d\uff0c\u4e5f\u53ef\u4ee5\u662f\u5f53\u524d\u5de5\u4f5c\u76ee\u5f55\u4e0b\u7684\u8def\u5f84\u3002"}),"\n",(0,i.jsx)(e.li,{children:"os.removedirs\uff1a\u5220\u9664\u6587\u4ef6\uff0c\u5e76\u5220\u9664\u4e2d\u95f4\u8def\u5f84\u4e2d\u7684\u7a7a\u6587\u4ef6\u5939"}),"\n",(0,i.jsx)(e.li,{children:"os.chdir(path)\uff1a\u5c06\u5f53\u524d\u5de5\u4f5c\u76ee\u5f55\u6539\u53d8\u4e3a\u6307\u5b9a\u7684\u8def\u5f84"}),"\n",(0,i.jsx)(e.li,{children:"os.getcwd()\uff1a\u8fd4\u56de\u5f53\u524d\u7684\u5de5\u4f5c\u76ee\u5f55"}),"\n",(0,i.jsx)(e.li,{children:"os.curdir\uff1a\u8868\u793a\u5f53\u524d\u76ee\u5f55\u7684\u7b26\u53f7"}),"\n",(0,i.jsx)(e.li,{children:"os.rename(old, new)\uff1a\u91cd\u547d\u540d\u6587\u4ef6"}),"\n",(0,i.jsx)(e.li,{children:"os.renames(old, new)\uff1a\u91cd\u547d\u540d\u6587\u4ef6\uff0c\u5982\u679c\u4e2d\u95f4\u8def\u5f84\u7684\u6587\u4ef6\u5939\u4e0d\u5b58\u5728\uff0c\u5219\u521b\u5efa\u6587\u4ef6\u5939"}),"\n",(0,i.jsx)(e.li,{children:"os.listdir(path)\uff1a\u8fd4\u56de\u7ed9\u5b9a\u76ee\u5f55\u4e0b\u7684\u6240\u6709\u6587\u4ef6\u5939\u548c\u6587\u4ef6\u540d\uff0c\u4e0d\u5305\u62ec '.' \u548c '..' \u4ee5\u53ca\u5b50\u6587\u4ef6\u5939\u4e0b\u7684\u76ee\u5f55\u3002\uff08'.' \u548c '..' \u5206\u522b\u6307\u5f53\u524d\u76ee\u5f55\u548c\u7236\u76ee\u5f55\uff09"}),"\n",(0,i.jsx)(e.li,{children:"os.mkdir(name)\uff1a\u4ea7\u751f\u65b0\u6587\u4ef6\u5939"}),"\n",(0,i.jsx)(e.li,{children:"os.makedirs(name)\uff1a\u4ea7\u751f\u65b0\u6587\u4ef6\u5939\uff0c\u5982\u679c\u4e2d\u95f4\u8def\u5f84\u7684\u6587\u4ef6\u5939\u4e0d\u5b58\u5728\uff0c\u5219\u521b\u5efa\u6587\u4ef6\u5939"}),"\n"]}),"\n",(0,i.jsx)(e.p,{children:"\u5bfc\u5165\u8be5\u6a21\u5757\uff1a"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:"import os\n"})}),"\n",(0,i.jsx)(e.p,{children:"\u4ea7\u751f\u6587\u4ef6\uff1a"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:"f = open('test.file', 'w')\nf.close()\nprint('test.file' in os.listdir(os.curdir))\n"})}),"\n",(0,i.jsx)(e.p,{children:"\u91cd\u547d\u540d\u6587\u4ef6:"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:'os.rename("test.file", "test.new.file")\nprint("test.file" in os.listdir(os.curdir))\nprint("test.new.file" in os.listdir(os.curdir))\n'})}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:'# \u5220\u9664\u6587\u4ef6\nos.remove("test.new.file")\n'})}),"\n",(0,i.jsx)(e.h3,{id:"\u7cfb\u7edf\u5e38\u91cf",children:"\u7cfb\u7edf\u5e38\u91cf"}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsx)(e.li,{children:"windows \u4e3a \\r\\n"}),"\n",(0,i.jsx)(e.li,{children:"unix\u4e3a \\n"}),"\n"]}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:"os.linesep\n"})}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:"# \u5f53\u524d\u64cd\u4f5c\u7cfb\u7edf\u7684\u8def\u5f84\u5206\u9694\u7b26\uff1a\nos.sep\n"})}),"\n",(0,i.jsx)(e.p,{children:"\u5f53\u524d\u64cd\u4f5c\u7cfb\u7edf\u7684\u73af\u5883\u53d8\u91cf\u4e2d\u7684\u5206\u9694\u7b26\uff08';' \u6216 ':'\uff09\uff1a"}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsx)(e.li,{children:"windows \u4e3a ;"}),"\n",(0,i.jsx)(e.li,{children:"unix \u4e3a:"}),"\n"]}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:"os.pathsep\n"})}),"\n",(0,i.jsx)(e.p,{children:"os.environ \u662f\u4e00\u4e2a\u5b58\u50a8\u6240\u6709\u73af\u5883\u53d8\u91cf\u7684\u503c\u7684\u5b57\u5178\uff0c\u53ef\u4ee5\u4fee\u6539\u3002"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:"os.environ\n"})}),"\n",(0,i.jsx)(e.h3,{id:"ospath-\u6a21\u5757",children:"os.path \u6a21\u5757"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:"import os.path\n"})}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsx)(e.li,{children:"os.path.isfile(path) \uff1a\u68c0\u6d4b\u4e00\u4e2a\u8def\u5f84\u662f\u5426\u4e3a\u666e\u901a\u6587\u4ef6"}),"\n",(0,i.jsx)(e.li,{children:"os.path.isdir(path)\uff1a\u68c0\u6d4b\u4e00\u4e2a\u8def\u5f84\u662f\u5426\u4e3a\u6587\u4ef6\u5939"}),"\n",(0,i.jsx)(e.li,{children:"os.path.exists(path)\uff1a\u68c0\u6d4b\u8def\u5f84\u662f\u5426\u5b58\u5728"}),"\n",(0,i.jsx)(e.li,{children:"os.path.isabs(path)\uff1a\u68c0\u6d4b\u8def\u5f84\u662f\u5426\u4e3a\u7edd\u5bf9\u8def\u5f84"}),"\n"]}),"\n",(0,i.jsx)(e.p,{children:"windows\u7cfb\u7edf\uff1a"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:'print(os.path.isfile("C:/Windows"))\nprint(os.path.isdir("C:/Windows"))\nprint(os.path.exists("C:/Windows"))\nprint(os.path.isabs("C:/Windows"))\n'})}),"\n",(0,i.jsx)(e.p,{children:"unix\u7cfb\u7edf\uff1a"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:'print(os.path.isfile("/Users"))\nprint(os.path.isdir("/Users"))\nprint(os.path.exists("/Users"))\nprint(os.path.isabs("/Users"))\n'})}),"\n",(0,i.jsx)(e.h3,{id:"split-\u548c-join",children:"split \u548c join"}),"\n",(0,i.jsxs)(e.ul,{children:["\n",(0,i.jsx)(e.li,{children:"os.path.split(path)\uff1a\u62c6\u5206\u4e00\u4e2a\u8def\u5f84\u4e3a (head, tail) \u4e24\u90e8\u5206"}),"\n",(0,i.jsx)(e.li,{children:"os.path.join(a, *p)\uff1a\u4f7f\u7528\u7cfb\u7edf\u7684\u8def\u5f84\u5206\u9694\u7b26\uff0c\u5c06\u5404\u4e2a\u90e8\u5206\u5408\u6210\u4e00\u4e2a\u8def\u5f84"}),"\n"]}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:'head, tail = os.path.split("c:/tem/b.txt")\nprint(head, tail)\n'})}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:'a = "c:/tem"\nb = "b.txt"\nos.path.join(a, b)\n'})}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:"def get_files(dir_path):\n    '''\n    \u5217\u51fa\u6587\u4ef6\u5939\u4e0b\u7684\u6240\u6709\u6587\u4ef6\n    :param dir_path: \u7236\u6587\u4ef6\u5939\u8def\u5f84\n    :return: \n    '''\n    for parent, dirname, filenames in os.walk(dir_path):\n        for filename in filenames:\n            print(\"parent is:\", parent)\n            print(\"filename is:\", filename)\n            print(\"full name of the file is:\", os.path.join(parent, filename))\n"})}),"\n",(0,i.jsx)(e.p,{children:"\u5217\u51fa\u5f53\u524d\u6587\u4ef6\u5939\u7684\u6240\u6709\u6587\u4ef6\uff1a"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:"dir = os.curdir\nget_files(dir)\n"})}),"\n",(0,i.jsx)(e.h3,{id:"byte-code-\u7f16\u8bd1",children:"Byte Code \u7f16\u8bd1"}),"\n",(0,i.jsx)(e.p,{children:"Python, Java \u7b49\u8bed\u8a00\u5148\u5c06\u4ee3\u7801\u7f16\u8bd1\u4e3a byte code\uff08\u4e0d\u662f\u673a\u5668\u7801\uff09\uff0c\u7136\u540e\u518d\u5904\u7406\uff1a"}),"\n",(0,i.jsxs)(e.blockquote,{children:["\n",(0,i.jsx)(e.p,{children:".py -> .pyc -> interpreter"}),"\n"]}),"\n",(0,i.jsx)(e.p,{children:"eval(statement, glob, local)"}),"\n",(0,i.jsx)(e.p,{children:"\u4f7f\u7528 eval \u51fd\u6570\u52a8\u6001\u6267\u884c\u4ee3\u7801\uff0c\u8fd4\u56de\u6267\u884c\u7684\u503c\u3002"}),"\n",(0,i.jsx)(e.p,{children:"exec(statement, glob, local)"}),"\n",(0,i.jsx)(e.p,{children:"\u4f7f\u7528 exec \u53ef\u4ee5\u6dfb\u52a0\u4fee\u6539\u539f\u6709\u7684\u53d8\u91cf:"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:"a = 1\nexec('b = a + 10')\nprint(b)\n"})}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:'local = dict(a=2)\nglob = {}\nexec("b = a+1", glob, local)\n\nprint(local)\n'})}),"\n",(0,i.jsx)(e.p,{children:"compile \u51fd\u6570\u751f\u6210 byte code\uff1a\ncompile(str, filename, mode)"}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:"a = 1\nb = compile('a+2', '', 'eval')\nprint(eval(b))\n"})}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:'a = 1\nc = compile("b=a+4", "", \'exec\')\nexec(c)\nprint(b)\n'})}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:"# abstract syntax trees\nimport ast\n\ntree = ast.parse('a+10', '', 'eval')\nast.dump(tree)\n"})}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:"a = 1\nc = compile(tree, '', 'eval')\nd = eval(c)\nprint(d)\n"})}),"\n",(0,i.jsx)(e.pre,{children:(0,i.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:"# \u5b89\u5168\u7684\u4f7f\u7528\u65b9\u6cd5 literal_eval \uff0c\u53ea\u652f\u6301\u57fa\u672c\u503c\u7684\u64cd\u4f5c\uff1a\nb = ast.literal_eval('[10.0, 2, True, \"foo\"]')\nprint(b)\n"})})]})}function d(n={}){const{wrapper:e}={...(0,t.a)(),...n.components};return e?(0,i.jsx)(e,{...n,children:(0,i.jsx)(c,{...n})}):c(n)}},1151:(n,e,s)=>{s.d(e,{Z:()=>a,a:()=>l});var i=s(7294);const t={},r=i.createContext(t);function l(n){const e=i.useContext(r);return i.useMemo((function(){return"function"==typeof n?n(e):{...e,...n}}),[e,n])}function a(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(t):n.components||t:l(n.components),i.createElement(r.Provider,{value:e},n.children)}}}]);