main :: IO ()

-- 1 函数，提取列表中所有奇数
getOddsInList :: [Int] -> [Int]
getOddsInList l = [x | x <- l, odd x]
-- 2 函数：列表所有元素求平方
getSquaresInList :: [Int] -> [Int]
getSquaresInList l = [x^x | x <- l]

main = do
    -- 打印字符串
    putStrLn("求 1 到 n 中所有奇数的平方和：n =")
    -- 获取输入字符串
    inputString <- getLine
    -- 变量num: 将输入中的整数赋值给num
    let num = read inputString :: Int

    -- 2 函数二，提取列表中所有奇数每个求平方，再求和
    print(sum(getSquaresInList(getOddsInList [1..num])))


