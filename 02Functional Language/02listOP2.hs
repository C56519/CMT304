-- 1 输入两个整数，一个是start，一个是finish，然后创建一个连续的整数列表
creatListFromTo :: Int -> Int -> [Int]
creatListFromTo x y
    -- 守卫：如果x > y，处理错误情况
    | x > y = []
    -- 如果x <= y，输出头部元素x，然后从x的下一个整数开始到y结束递归
    | x <= y = x : creatListFromTo (x + 1) y


-- 2 求一个数的阶乘
computingFactorial :: Int -> Int
-- 学习一种函数中定义辅助函数的方法，使用where关键词
-- 这里的 fact 辅助函数接受两个参数，起始数字为1，结束数字为x
computingFactorial x = fact 1 x
    where
        fact :: Int -> Int -> Int
        fact m n
            -- 守卫，处理错误情况
            | m > n = 1
            -- 如果m<=n，递归计算阶乘，m乘以从m的下一个整数开始到y结束递归
            | m <= n = m * fact (m + 1) n


-- 3 将两个列表组合成元组列表
-- 注意：也可以直接使用库函数 zip list1 list2
listZip :: [a] -> [b] -> [(a, b)]
-- 处理错误情况，空列表
-- 另一个作用是当一个列表长度不够时终止递归
listZip [] l2 = []
listZip l1 [] = []
-- 将它们的头部元素组合成一个元组 (x,y)，并将该元组添加到结果列表的头部。然后递归地对剩余的列表部分调用 zip 函数，继续进行组合。
listZip (x : xs) (y : ys) = (x, y) : listZip xs ys


-- 4 按值查找列表中的元素，返回所有一样元素的索引列表
-- 该函数接受一个列表和一个值，Eq a ，是一个类型约束，表示类型 a 必须支持相等性比较
searchValue :: Eq a => [a] -> a -> [Int]
searchValue l v = srch l v 0
    where
        -- 函数中定义了一个辅助函数srch，参数：列表，搜寻的值，索引计数器(从0开始计数)
        srch :: Eq a => [a] -> a -> Int -> [Int]
        -- 如果别表为空，直接返回空列表
        srch [] v i = []
        -- 如果不为空，分割列表为当前头部元素和剩余列表
        srch (x : xs) v i
            -- 如果当前头部元素是要找的值，将其索引添加到输出列表中，然后在剩余列表中继续递归，找找还有没有
            | x == v = i : srch xs v (i + 1)
            -- 其他情况，在剩余列表中继续递归，继续找
            | otherwise = srch xs v (i + 1)

-- 5 结合zip库函数使用列表推导来按照值查找
searchComp :: Eq a => [a] -> a -> [Int]
-- 首先将索引列表[0..]与输入列表zip匹配
-- 然后抽取匹配后的列表的每个元素，如果该元素值等于v，那么输出该元素的索引到输出列表中
searchComp l v = [ i | (i, x) <- zip [0..] l, x == v ]


-- 6 map
--  直接定义一个不带参数的addOne函数，表示其使用map并将函数设置为默认执行一个匿名函数，将输入的每个元素+1
addOne :: [Int] -> [Int]
addOne = map (\x -> x + 1)

-- 7 filter
--  直接定义一个不带参数的函数，表示其使用filter并将断言函数设置为默认执行一个匿名函数，判断该元素是否大于零，输出所有大于零的元素列表
positiveFilter :: [Int] -> [Int]
positiveFilter = filter (\x -> x > 0)

-- 8 fold
-- 对列表每个元素执行累加，返回累加值；如果列表为空，默认返回 0
sumList :: [Int] -> Int
sumList = foldr (+) 0

--对列表每个元素执行累乘，返回累乘值；如果列表为空，默认返回 1
productList:: [Int] -> Int
productList = foldr (*) 1

-- 列表降维(推平)，将二维列表降维拼接成一维列表
concatenateList :: [[a]] -> [a]
concatenateList = foldr (++) []