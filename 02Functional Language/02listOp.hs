-- 计算例表中每个元素的平方
-- 方法1： 使用之前的方法
squares :: [Int] -> [Int]
squares l = [x^2 | x <- l]


-- 定义函数
-- 实现递归遍历列表，并将每个元素求平方
-- 首先先声明函数，并指出参数为int列表，返回一个int列表
squaresCond :: [Int] -> [Int]
squaresCond l = 
    -- 判空操作：如果列表为空，直接返回空列表
    -- 作用：防止原始列表就为空，另一个是递归到最后为[]空时跳出递归
    if null l then []
    -- 如果列表不为空，开始递归操作
    else 
        -- 分割当前递归下的列表，头部元素和剩余元素列表
        let x = head l
            xs = tail l
        -- 头部元素执行平方操作，剩余元素列表开始递归
        in
            x^2 : squaresCond xs

-- 也可以如下实现
squaresRec :: [Int] -> [Int]
-- 判空操作：如果列表为空，直接返回空列表，作用见上
squaresRec [] = []
-- 如果列表不为空，开始递归操作
-- 将列表分解为头部元素 x 和剩余元素列表 xs
-- 然后头部元素取平方，剩余列表列表 xs 执行递归，调用 squaresRec 函数
squaresRec (x : l) = x^2 : squaresRec l
