---
id: install_nautilus
title: Installing Nautilus
---

__The Nautilus Stack is built using Catkin, you should already have a catkin workspace created__, if you do not follow this [tutorial to create it first](http://wiki.ros.org/catkin/Tutorials/create_a_workspace).

The stack should be placed into the src folder in your catkin workspace.

```
cd <catkin_workspace_path>/src
```

First clone the stack and all it's submodules:

```
git clone --recurse-submodules https://github.com/TheCynosure/nautilus_stack.git
```

Next we need to install all the dependencies:

```
cd nautilus_stack/nautilus
sudo ./install_ubuntu.sh
```

Now we need to build the stack:

```
cd <catkin_workspace_path>
catkin_make
```
:::note
For faster performance check that you are building the Release version of nautilus. In the `CMakeLists.txt` inside ```nautilus_stack/nautilus``` check the `CMAKE_BUILD_TYPE` is to set to be `Release`.
:::

Now finally make sure to source the `setup.bash` file to get all the relavent environment variables:

```
source devel/setup.bash
```
