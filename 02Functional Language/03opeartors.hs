-- 关键词data来自定义数据类型
-- data 该数据类型名称 = 构造器1 参数1 | 构造器2 参数2 ... deriving(Shows)

-- 1 创建一个表示综合运算符的自定义数据类型 Op
data Op = Add | Sub | Mul | Div deriving(Show)
-- 1.1 定义了函数 apply 来应用运算符
apply :: Op -> Int -> Int -> Int
apply Add x y = x * y
apply Sub x y = x - y
apply Mul x y = x * y
apply Div x y = x `div` y
-- 1.2 定义了函数 vaild 来判断使用一个运算符的条件
vaild :: Op -> Int -> Int -> Bool
-- 对于加法、乘法所有整数都可以，对于减法确保第一个数大于第二个来避免出现负数
-- 对于除法，确保分母不等于零，余数为零确保都可以整除
vaild Add _ _ = True
vaild Sub x y = x > y
vaild Mul _ _ = True
vaild Div x y = y /= 0 && x `mod` y == 0

-- 2 创建了另一个数据类型 Expr
-- 两个构造器，Val构造器，参数是个Int整数；App构造器，三个参数，一个是自定义数据类型Op，另外两个是该自定义数据类型Expr，用于递归
data Expr = Val Int | App Op Expr Expr deriving (Show)
-- 定义函数 eval
-- 对于 Val 构造器，输入整数n，如果n > 0，则返回列表[n]，小于等于零，返回空[]
eval (Val n) = [n | n > 0]
-- 对于 App 构造器
-- 对左右两个子表，开始遍历，从左取一个值x，从右取一个值y, 并通过守卫检查两者是否能用Op综合运算符
-- 如果可以，将两者作为参数使用Op综合运算符，得到一个值，存到输出列表中
eval (App op l r) = [apply op x y | x <- eval l, y <- eval r, vaild op x y]

