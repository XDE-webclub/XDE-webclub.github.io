"use strict";(self.webpackChunkjiangmiemie=self.webpackChunkjiangmiemie||[]).push([[4971],{6608:(n,e,s)=>{s.r(e),s.d(e,{assets:()=>o,contentTitle:()=>p,default:()=>c,frontMatter:()=>l,metadata:()=>r,toc:()=>h});var t=s(4848),i=s(8453);const l={sidebar_position:24,title:"Python\u53d1\u5e03\u5305(\u9009\u4fee)"},p=void 0,r={id:"\u7b2c1\u90e8\u5206Python\u57fa\u7840\u77e5\u8bc6/Python\u53d1\u5e03\u5305",title:"Python\u53d1\u5e03\u5305(\u9009\u4fee)",description:"Python\u53d1\u5e03\u5305",source:"@site/docs/\u7b2c1\u90e8\u5206Python\u57fa\u7840\u77e5\u8bc6/Python\u53d1\u5e03\u5305.md",sourceDirName:"\u7b2c1\u90e8\u5206Python\u57fa\u7840\u77e5\u8bc6",slug:"/\u7b2c1\u90e8\u5206Python\u57fa\u7840\u77e5\u8bc6/Python\u53d1\u5e03\u5305",permalink:"/docs/\u7b2c1\u90e8\u5206Python\u57fa\u7840\u77e5\u8bc6/Python\u53d1\u5e03\u5305",draft:!1,unlisted:!1,tags:[],version:"current",sidebarPosition:24,frontMatter:{sidebar_position:24,title:"Python\u53d1\u5e03\u5305(\u9009\u4fee)"},sidebar:"tutorialSidebar",previous:{title:"Python\u8c03\u7528C(\u9009\u4fee)",permalink:"/docs/\u7b2c1\u90e8\u5206Python\u57fa\u7840\u77e5\u8bc6/Python\u8c03\u7528C"},next:{title:"\u6570\u636e\u79d1\u5b66\u5165\u95e8",permalink:"/docs/\u7b2c1\u90e8\u5206Python\u57fa\u7840\u77e5\u8bc6/\u6570\u636e\u79d1\u5b66\u5165\u95e8"}},o={},h=[{value:"Python\u53d1\u5e03\u5305",id:"python\u53d1\u5e03\u5305",level:2},{value:"\u6253\u5305",id:"\u6253\u5305",level:3},{value:"\u53d1\u5e03",id:"\u53d1\u5e03",level:3}];function d(n){const e={code:"code",h2:"h2",h3:"h3",li:"li",ol:"ol",p:"p",pre:"pre",ul:"ul",...(0,i.R)(),...n.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(e.h2,{id:"python\u53d1\u5e03\u5305",children:"Python\u53d1\u5e03\u5305"}),"\n",(0,t.jsx)(e.h3,{id:"\u6253\u5305",children:"\u6253\u5305"}),"\n",(0,t.jsx)(e.p,{children:"\u8fd9\u91ccPython 3.12 \u4ee5\u524d\u7684\u8001\u9879\u76ee\u53ef\u4ee5\u4f7f\u7528distutils\u6a21\u5757\uff0c\u66f4\u63a8\u8350\u4f7f\u7528setuptools\u6a21\u5757\uff0csetuptools\u6700\u5e38\u7528\u7684\u529f\u80fd\u6709\uff1a"}),"\n",(0,t.jsxs)(e.ul,{children:["\n",(0,t.jsx)(e.li,{children:"\u4f9d\u8d56\u5305\u5b89\u88c5\u4e0e\u7248\u672c\u7ba1\u7406"}),"\n",(0,t.jsx)(e.li,{children:"python\u5e93\u7684\u6253\u5305\u5206\u53d1"}),"\n",(0,t.jsx)(e.li,{children:"c/c++ \u62d3\u5c55"}),"\n",(0,t.jsx)(e.li,{children:"python\u73af\u5883\u9650\u5236\u4e0e\u751f\u6210\u811a\u672c"}),"\n"]}),"\n",(0,t.jsx)(e.p,{children:"\u6574\u4e2a\u6253\u5305\u8fc7\u7a0b\u6700\u91cd\u8981\u7684\u5c31\u662f__setup.py__\uff0c\u5b83\u6307\u5b9a\u4e86\u91cd\u8981\u7684\u914d\u7f6e\u4fe1\u606f\u3002setup.py\u7684\u5185\u5bb9\u5982\u4e0b(\u793a\u4f8b)\uff1a"}),"\n",(0,t.jsx)(e.pre,{children:(0,t.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:"from setuptools import setup,Extension\n\nsetup(\n    ext_modules=[\n    Extension(\n    name = 'spam', # \u5305\u540d\u79f0\n    sources=['spammodule.cpp'],\n    )]\n)\n"})}),"\n",(0,t.jsx)(e.p,{children:"\u901a\u8fc7setup\u51fd\u6570\u7684\u8fd9\u4e9b\u53c2\u6570packages\u3001include_package_data\uff08\u5176\u5b9e\u5c31\u662fMANIFEST.in\u6587\u4ef6\uff09\u3001exclude_package_data\u3001package_data\u3001data_files\u6765\u6307\u5b9a\u9700\u8981\u6253\u5305\u7684\u6587\u4ef6\u3002"}),"\n",(0,t.jsx)(e.p,{children:"\u5305\u542b\u7684\u6587\u4ef6\u5982\u4e0b\uff1a"}),"\n",(0,t.jsxs)(e.ul,{children:["\n",(0,t.jsx)(e.li,{children:"py_modules \u548c packages \u53c2\u6570\u4e2d\u6240\u6709 Python \u6e90\u6587\u4ef6"}),"\n",(0,t.jsx)(e.li,{children:"ext_modules or libraries \u53c2\u6570\u4e2d\u63d0\u5230\u7684\u6240\u6709 C \u6e90\u6587\u4ef6"}),"\n",(0,t.jsx)(e.li,{children:"scripts \u53c2\u6570\u6307\u5b9a\u7684\u811a\u672c"}),"\n",(0,t.jsx)(e.li,{children:"package_data \u548c data_files \u53c2\u6570\u6307\u5b9a\u7684\u6240\u6709\u6587\u4ef6"}),"\n",(0,t.jsx)(e.li,{children:"setup.cfg \u548c setup.py"}),"\n",(0,t.jsx)(e.li,{children:"\u7c7b\u4f3c\u4e8ereadme\u7684\u6587\u4ef6\uff08\u5982README\u3001README.txt\u3001 README.rst\u3001README.md\uff09"}),"\n",(0,t.jsx)(e.li,{children:"MANIFEST.in \u4e2d\u6307\u5b9a\u7684\u6240\u6709\u6587\u4ef6\uff08\u5f53\u8fd0\u884cpython setup.py sdist\u65f6\uff0c\u4f1a\u67e5\u9605MANIFEST.in\u6587\u4ef6\uff0c\u5e76\u4e14\u5c06\u91cc\u9762\u7ea6\u5b9a\u7684\u6587\u4ef6\u6253\u5305\u5230\u6700\u540e\u7684\u5305\u91cc\u3002\u4ec0\u4e48\u8981\uff0c\u4ec0\u4e48\u4e0d\u8981\uff09"}),"\n"]}),"\n",(0,t.jsx)(e.p,{children:"\u6253\u5305\u547d\u4ee4\u8bf4\u660e\uff1a"}),"\n",(0,t.jsxs)(e.ol,{children:["\n",(0,t.jsx)(e.li,{children:"\u6e90\u7801\u5305source dist\uff08\u7b80\u79f0sdist\uff09\uff1a\u5c31\u662f\u6211\u4eec\u719f\u6089\u7684 .zip \u3001.tar.gz \u7b49\u540e\u7f00\u6587\u4ef6\u3002\u5c31\u662f\u4e00\u4e2a\u538b\u7f29\u5305\uff0c\u91cc\u9762\u5305\u542b\u4e86\u6240\u9700\u8981\u7684\u7684\u6240\u6709\u6e90\u7801\u6587\u4ef6\u4ee5\u53ca\u4e00\u4e9b\u9759\u6001\u6587\u4ef6\uff08txt\u6587\u672c\u3001css\u3001\u56fe\u7247\u7b49\uff09\u3002"}),"\n"]}),"\n",(0,t.jsx)(e.pre,{children:(0,t.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:"python setup.py sdist --formats=gztar\n"})}),"\n",(0,t.jsxs)(e.ol,{start:"2",children:["\n",(0,t.jsx)(e.li,{children:"\u4e8c\u8fdb\u5236\u5305binary dist\uff08\u7b80\u79f0bdist\uff09\uff1a\u683c\u5f0f\u662fwheel\uff08.whl\u540e\u7f00\uff09\uff0c\u5b83\u7684\u524d\u8eab\u662fegg\u3002wheel\u672c\u8d28\u4e5f\u8fd8\u662f\u4e00\u4e2a\u538b\u7f29\u5305\uff0c\u53ef\u4ee5\u50cf\u50cfzip\u4e00\u6837\u89e3\u538b\u7f29\u3002\u4e0e\u6e90\u7801\u5305\u76f8\u6bd4\uff0c\u4e8c\u8fdb\u5236\u5305\u7684\u7279\u70b9\u662f\u4e0d\u7528\u518d\u7f16\u8bd1\uff0c\u4e5f\u5c31\u662f\u5b89\u88c5\u66f4\u5feb\uff01\u5728\u4f7f\u7528wheel\u4e4b\u524d\uff0c\u9700\u8981\u5148\u5b89\u88c5wheel\u6a21\u5757"}),"\n"]}),"\n",(0,t.jsx)(e.pre,{children:(0,t.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:"# \u5148\u5b89\u88c5wheel\u6a21\u5757\npip install wheel\n\npython setup.py bdist --formats=rpm\n# \u7b49\u4ef7\u4e8e\npython setup.py build_rpm\n"})}),"\n",(0,t.jsxs)(e.ol,{start:"3",children:["\n",(0,t.jsx)(e.li,{children:"\u5f00\u53d1\u65b9\u5f0f\u5b89\u88c5\u5305\uff0c\u8be5\u547d\u540d\u4e0d\u4f1a\u771f\u6b63\u7684\u5b89\u88c5\u5305\uff0c\u800c\u662f\u5728\u7cfb\u7edf\u73af\u5883\u4e2d\u521b\u5efa\u4e00\u4e2a\u8f6f\u94fe\u63a5\u6307\u5411\u5305\u5b9e\u9645\u6240\u5728\u76ee\u5f55\u3002\u8fd9\u8fb9\u5728\u4fee\u6539\u5305\u4e4b\u540e\u4e0d\u7528\u518d\u5b89\u88c5\u5c31\u80fd\u751f\u6548\uff0c\u4fbf\u4e8e\u8c03\u8bd5\u3002"}),"\n"]}),"\n",(0,t.jsx)(e.pre,{children:(0,t.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:"pip install -e .\n\u7b49\u4ef7\u4e8e\npython setup.py develop\n"})}),"\n",(0,t.jsxs)(e.ol,{start:"4",children:["\n",(0,t.jsx)(e.li,{children:"\u6784\u5efa\u6269\u5c55\uff0c\u5982\u7528 C/C++, Cython \u7b49\u7f16\u5199\u7684\u6269\u5c55\uff0c\u5728\u8c03\u8bd5\u65f6\u901a\u5e38\u52a0 --inplace \u53c2\u6570\uff0c\u8868\u793a\u539f\u5730\u7f16\u8bd1\uff0c\u5373\u751f\u6210\u7684\u6269\u5c55\u4e0e\u6e90\u6587\u4ef6\u5728\u540c\u6837\u7684\u4f4d\u7f6e\u3002"}),"\n"]}),"\n",(0,t.jsx)(e.pre,{children:(0,t.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:"python setup.py build_ext --inplace\n"})}),"\n",(0,t.jsxs)(e.ol,{start:"5",children:["\n",(0,t.jsx)(e.li,{children:"\u6784\u5efa\u4e00\u4e2a wheel \u5206\u53d1\u5305\uff0cegg \u5305\u662f\u8fc7\u65f6\u7684\uff0cwhl \u5305\u662f\u65b0\u7684\u6807\u51c6"}),"\n"]}),"\n",(0,t.jsx)(e.pre,{children:(0,t.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:"python setup.py bdist_wheel\n"})}),"\n",(0,t.jsxs)(e.ol,{start:"6",children:["\n",(0,t.jsx)(e.li,{children:"\u6784\u5efa\u4e00\u4e2a egg \u5206\u53d1\u5305\uff0c\u7ecf\u5e38\u7528\u6765\u66ff\u4ee3\u57fa\u4e8e bdist \u751f\u6210\u7684\u6a21\u5f0f"}),"\n"]}),"\n",(0,t.jsx)(e.pre,{children:(0,t.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:"python setup.py bdist_egg\n"})}),"\n",(0,t.jsxs)(e.ol,{start:"7",children:["\n",(0,t.jsx)(e.li,{children:"\u5b89\u88c5\u5230\u5e93"}),"\n"]}),"\n",(0,t.jsx)(e.pre,{children:(0,t.jsx)(e.code,{className:"language-python",metastring:"showLineNumbers",children:"python setup.py install\n#\u7b49\u4ef7\u4e8e\npython setup.py build\npython setup.py install\n\n#python setup.py install\u5305\u62ec\u4e24\u6b65\uff1apython setup.py build python setup.py install\u3002\n#\u8fd9\u4e24\u6b65\u53ef\u5206\u5f00\u6267\u884c\uff0c \u4e5f\u53ef\u53ea\u6267\u884cpython setup.py install, \u56e0\u4e3apython setup.py install\u603b\u662f\u4f1a\u5148build\u540einstall.\n\n\n#\u6839\u636e\u751f\u6210\u7684\u6587\u4ef6\u7b49\u4ef7\u4e8e\npip install  xxx.zip\n# \u6216\npip install xxx.whl\n# \u6216.... xxx.egg\n"})}),"\n",(0,t.jsx)(e.h3,{id:"\u53d1\u5e03",children:"\u53d1\u5e03"}),"\n",(0,t.jsx)(e.p,{children:"\u5982\u679c\u6211\u4eec\u9700\u8981\u5305\u88ab\u5168\u4e16\u754c\u7684\u540c\u597d\u901a\u8fc7 pip install \u76f4\u63a5\u5b89\u88c5\u7684\u8bdd\uff0c\u9700\u8981\u5c06\u5305\u4e0a\u4f20\u5230 pypi \u7f51\u7ad9\u3002\u9996\u5148\u6ce8\u518c pypi\uff0c\u83b7\u5f97\u7528\u6237\u540d\u548c\u5bc6\u7801\u3002"}),"\n",(0,t.jsx)(e.p,{children:"\u4e0a\u4f20 tar \u5305"}),"\n",(0,t.jsx)(e.p,{children:(0,t.jsx)(e.code,{children:"python setup.py sdist upload"})}),"\n",(0,t.jsx)(e.p,{children:"\u4e0a\u4f20 whl \u5305"}),"\n",(0,t.jsx)(e.p,{children:(0,t.jsx)(e.code,{children:"python setup.py bdist_wheel upload"})}),"\n",(0,t.jsx)(e.p,{children:"\u5982\u679c\u8981\u66f4\u5b89\u5168\u548c\u65b9\u4fbf\u5730\u4e0a\u4f20\u5305\u5c31\u4f7f\u7528 twine \u4e0a\u4f20\u3002"}),"\n",(0,t.jsx)(e.p,{children:"\u5b89\u88c5 twine"}),"\n",(0,t.jsx)(e.p,{children:(0,t.jsx)(e.code,{children:"pip install twine"})}),"\n",(0,t.jsx)(e.p,{children:"\u4e0a\u4f20\u6240\u6709\u5305"}),"\n",(0,t.jsx)(e.p,{children:(0,t.jsx)(e.code,{children:"twine upload dist/*"})}),"\n",(0,t.jsx)(e.p,{children:"\u5982\u679c\u5acc\u6bcf\u6b21\u8f93\u5165\u7528\u6237\u540d\u548c\u5bc6\u7801\u9ebb\u70e6\u53ef\u4ee5\u914d\u7f6e\u5230\u6587\u4ef6\u4e2d\u3002"}),"\n",(0,t.jsx)(e.p,{children:"\u7f16\u8f91\u7528\u6237\u76ee\u5f55\u4e0b\u7684 .pypirc \u6587\u4ef6\uff0c\u8f93\u5165"}),"\n",(0,t.jsx)(e.pre,{children:(0,t.jsx)(e.code,{className:"language-bash",children:"[pypi]\nusername=your_username\npassword=your_password\n"})})]})}function c(n={}){const{wrapper:e}={...(0,i.R)(),...n.components};return e?(0,t.jsx)(e,{...n,children:(0,t.jsx)(d,{...n})}):d(n)}},8453:(n,e,s)=>{s.d(e,{R:()=>p,x:()=>r});var t=s(6540);const i={},l=t.createContext(i);function p(n){const e=t.useContext(l);return t.useMemo((function(){return"function"==typeof n?n(e):{...e,...n}}),[e,n])}function r(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(i):n.components||i:p(n.components),t.createElement(l.Provider,{value:e},n.children)}}}]);