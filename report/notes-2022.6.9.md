# Notes-2022.6.9

## 遇到的问题

1. 从ONNX通过trtexec转换到TensorRT时，遭遇不支持MOD_13 OP的情况，发现现在的tensorrt确实还不支持

2. 从ONNX通过trtexec转换到TensorRT时，遭遇不支持PAD_52 OP的情况，发现tensorrt已经支持PAD OP，思考有可能是trtexec的问题

   ![image-20220610200347722](C:\Users\Ethan\AppData\Roaming\Typora\typora-user-images\image-20220610200347722.png)

## 解决方案

1. 进行graph surgeon，对于MOD_13 OP `a % b` 使用 `a - a // b * b`代替，即使用一个SUB和一个DIV代替

2. 抛弃trtexec，改用手动转换方案：并没有效果。

   书写reflect padding的plugin。或者手动生成元素使用矩阵拼接替换pad操作。

