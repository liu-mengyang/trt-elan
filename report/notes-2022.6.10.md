# Notes-2022.6.10

暂时跳过pad OP，直接优化后面的ELAB。

## 遇到的问题

1. reformat不支持

   ![image-20220610233623824](C:\Users\Ethan\AppData\Roaming\Typora\typora-user-images\image-20220610233623824.png)

​		似乎是dynamic shape的问题，因为我将pad OP skip掉了，dynamic shape可能出问题。

**解决方案**

1. 暂时使用合理的固定尺寸，比如针对64x64输入，固定尺寸到80x80
