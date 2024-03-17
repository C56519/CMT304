-- 一、计算例表中每个元素的平方
-- 方法1： 使用之前的方法
squares :: [Int] -> [Int]
squares l = [x^2 | x <- l]


-- 方法二：定义函数
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

-- 方法三：也可以如下实现递归
squaresRec :: [Int] -> [Int]
-- 判空操作：如果列表为空，直接返回空列表，作用见上
squaresRec [] = []
-- 如果列表不为空，开始递归操作
-- 将列表分解为头部元素 x 和剩余元素列表 xs
-- 然后头部元素取平方，剩余列表列表 xs 执行递归，调用 squaresRec 函数
squaresRec (x : l) = x^2 : squaresRec l



-- 二、过滤列表中的奇数
-- 方法一：使用守卫的方法
odds :: [Int] -> [Int]
odds l = [x | x <- l, odd x]

-- 方法二：递归
oddsCond :: [Int] -> [Int]
oddsCond l =
    -- 判空
    if null l then []
    else
        -- 分割
        let
            x = head l
            xs = tail l
        in
            -- 如果是奇数，头部元素输出，剩余列表递归
            if odd x then x : oddsCond xs
            -- 如果不是奇数，直接递归剩余列表
            else oddsCond xs

-- 方法三：另一种递归
oddsRec :: [Int] -> [Int]
-- 判空
oddsRec [] = []
-- 分割
oddsRec (x : xs)
    -- 守卫来判断是否是奇数
    -- 如果是，首元素输出，剩余列表递归 
    | odd x = x : oddsRec xs
    -- 如果不是，剩余列表递归
    | otherwise = oddsRec xs