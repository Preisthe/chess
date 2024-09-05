# chess

A Chinese chess recognition system
对 https://github.com/haodong2000/chess_vision 的复现

## Files

params.py: 参数、标签、棋子类别
crop.py: 裁剪 chess.jpg, 生成旋转后的数据集
CnnModel.py: 定义模型、模型训练函数和测试函数
train.py: 加载数据集并训练模型
chessboard.py: 加载 validation 文件夹中的测试集测试，或 test 文件夹中的图片测试

## Usage

```bash
# run following commands in the root working directory
python crop.py # 生成图片集
python train.py # 开始训练
python chessboard.py # 测试模型，测试前请在程序中修改对应的模型路径
```
