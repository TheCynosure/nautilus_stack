(window.webpackJsonp=window.webpackJsonp||[]).push([[15],{152:function(e,t,n){"use strict";n.r(t),n.d(t,"frontMatter",(function(){return o})),n.d(t,"metadata",(function(){return i})),n.d(t,"rightToc",(function(){return l})),n.d(t,"default",(function(){return u}));var a=n(2),r=n(9),c=(n(0),n(170)),o={id:"install_nautilus",title:"Installing Nautilus"},i={id:"install_nautilus",title:"Installing Nautilus",description:"__The Nautilus Stack is built using Catkin, you should already have a catkin workspace created__, if you do not follow this [tutorial to create it first](http://wiki.ros.org/catkin/Tutorials/create_a_workspace).",source:"@site/docs/install_nautilus.md",permalink:"/nautilus_docs/docs/install_nautilus",editUrl:"https://github.com/facebook/docusaurus/edit/master/website/docs/install_nautilus.md",sidebar:"sidebar",previous:{title:"Getting Started",permalink:"/nautilus_docs/docs/getting_started"},next:{title:"Run Nautilus without Autonomous Loop Closure",permalink:"/nautilus_docs/docs/run_nautilus"}},l=[],s={rightToc:l};function u(e){var t=e.components,n=Object(r.a)(e,["components"]);return Object(c.b)("wrapper",Object(a.a)({},s,n,{components:t,mdxType:"MDXLayout"}),Object(c.b)("p",null,Object(c.b)("strong",{parentName:"p"},"The Nautilus Stack is built using Catkin, you should already have a catkin workspace created"),", if you do not follow this ",Object(c.b)("a",Object(a.a)({parentName:"p"},{href:"http://wiki.ros.org/catkin/Tutorials/create_a_workspace"}),"tutorial to create it first"),"."),Object(c.b)("p",null,"The stack should be placed into the src folder in your catkin workspace."),Object(c.b)("pre",null,Object(c.b)("code",Object(a.a)({parentName:"pre"},{}),"cd <catkin_workspace_path>/src\n")),Object(c.b)("p",null,"First clone the stack and all it's submodules:"),Object(c.b)("pre",null,Object(c.b)("code",Object(a.a)({parentName:"pre"},{}),"git clone --recurse-submodules https://github.com/TheCynosure/nautilus_stack.git\n")),Object(c.b)("p",null,"Next we need to install all the dependencies:"),Object(c.b)("pre",null,Object(c.b)("code",Object(a.a)({parentName:"pre"},{}),"cd nautilus_stack/nautilus\nsudo ./install_ubuntu.sh\n")),Object(c.b)("p",null,"Now we need to build the stack:"),Object(c.b)("pre",null,Object(c.b)("code",Object(a.a)({parentName:"pre"},{}),"cd <catkin_workspace_path>\ncatkin_make\n")),Object(c.b)("div",{className:"admonition admonition-note alert alert--secondary"},Object(c.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-heading"}),Object(c.b)("h5",{parentName:"div"},Object(c.b)("span",Object(a.a)({parentName:"h5"},{className:"admonition-icon"}),Object(c.b)("svg",Object(a.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(c.b)("path",Object(a.a)({parentName:"svg"},{fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"})))),"note")),Object(c.b)("div",Object(a.a)({parentName:"div"},{className:"admonition-content"}),Object(c.b)("p",{parentName:"div"},"For faster performance check that you are building the Release version of nautilus. In the ",Object(c.b)("inlineCode",{parentName:"p"},"CMakeLists.txt")," inside ",Object(c.b)("inlineCode",{parentName:"p"},"nautilus_stack/nautilus")," check the ",Object(c.b)("inlineCode",{parentName:"p"},"CMAKE_BUILD_TYPE")," is to set to be ",Object(c.b)("inlineCode",{parentName:"p"},"Release"),"."))),Object(c.b)("p",null,"Now finally make sure to source the ",Object(c.b)("inlineCode",{parentName:"p"},"setup.bash")," file to get all the relavent environment variables:"),Object(c.b)("pre",null,Object(c.b)("code",Object(a.a)({parentName:"pre"},{}),"source devel/setup.bash\n")))}u.isMDXComponent=!0},170:function(e,t,n){"use strict";n.d(t,"a",(function(){return p})),n.d(t,"b",(function(){return m}));var a=n(0),r=n.n(a);function c(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){c(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,a,r=function(e,t){if(null==e)return{};var n,a,r={},c=Object.keys(e);for(a=0;a<c.length;a++)n=c[a],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var c=Object.getOwnPropertySymbols(e);for(a=0;a<c.length;a++)n=c[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var s=r.a.createContext({}),u=function(e){var t=r.a.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):i({},t,{},e)),n},p=function(e){var t=u(e.components);return r.a.createElement(s.Provider,{value:t},e.children)},b={inlineCode:"code",wrapper:function(e){var t=e.children;return r.a.createElement(r.a.Fragment,{},t)}},d=Object(a.forwardRef)((function(e,t){var n=e.components,a=e.mdxType,c=e.originalType,o=e.parentName,s=l(e,["components","mdxType","originalType","parentName"]),p=u(n),d=a,m=p["".concat(o,".").concat(d)]||p[d]||b[d]||c;return n?r.a.createElement(m,i({ref:t},s,{components:n})):r.a.createElement(m,i({ref:t},s))}));function m(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var c=n.length,o=new Array(c);o[0]=d;var i={};for(var l in t)hasOwnProperty.call(t,l)&&(i[l]=t[l]);i.originalType=e,i.mdxType="string"==typeof e?e:a,o[1]=i;for(var s=2;s<c;s++)o[s]=n[s];return r.a.createElement.apply(null,o)}return r.a.createElement.apply(null,n)}d.displayName="MDXCreateElement"}}]);