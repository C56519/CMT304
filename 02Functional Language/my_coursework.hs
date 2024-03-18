-- 1 定义变量
-- 表示一个二元图片，即由整数构成的二维列表
type BinaryImage = [[Int]]
-- 坐标：定义一个类型别名，表示一个坐标，即由两个整数构成的元组
type Coord = (Int, Int)
-- 是否被访问：定义一个类型别名，表示一个访问记录，即由布尔值构成的二维列表
type Visited = [[Bool]]

-- 主函数: 查找最大连通组件的数量
nlcc :: BinaryImage -> Int -> Int
nlcc l v = maximum $ map (\coord -> snd $ exploreNeighbors coord visited) allCoords
    where
        -- 1 信息准备
        -- 获取二元图片的行数和列数
        rowsNum = length l
        colsNum = length (head l)
        --创建列表，存储所有坐标
        allCoords = [(x, y) | x <- [0..rowsNum-1], y <- [0..colsNum-1]]
        -- 初始化访问列表
        visited = [[False | _ <- [0..colsNum - 1]] | _ <- [0..rowsNum - 1]]


        --startExplore :: Coord -> (Visited, Int)
        {-startExplore (x, y)
            | l !! x !! y == v && not (visited !! x !! y) = exploreNeighbors (0, 0) visited
            | otherwise = 0
        -}

        -- 判断坐标是否超出图像边界
        ifInBounds :: Coord -> Bool
        ifInBounds (x, y) = x >= 0 && y >= 0 && x < rowsNum && y < colsNum

        -- 获取该像素的邻居像素坐标
        getNeighbors :: Coord -> [Coord]
        getNeighbors (x, y) = filter ifInBounds [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

        -- 更新访问列表
        markVisited :: Int -> Int -> Visited -> Visited
        markVisited x y vl =
          -- 使用分解列表的方式获取目标行前和目标行之后的所有行
          let (beforeRows, targetRow:afterRows) = splitAt x vl
              -- 替换目标行中的目标列元素
              modifiedRow = take y targetRow ++ [True] ++ drop (y + 1) targetRow
          in beforeRows ++ [modifiedRow] ++ afterRows

        exploreNeighbors :: Coord -> Visited -> (Visited, Int)
        exploreNeighbors (x, y) visited
            -- 如果不在边界内，或被访问过，或与搜索值v不匹配 返回0
            | not (ifInBounds (x, y)) || visited !! x !! y || l !! x !! y /= v    = (visited, 0)
            -- 否则，探索周边邻居，最后将结果累加
            | otherwise =
                let visited' = markVisited x y visited
                    neighbors = getNeighbors (x, y)
                    {-
                                        regionResult = map  (`exploreNeighbors` visited') neighbors
                    total = 1 + sum (map snd regionResult)
                in (visited', total)
                    -}
                                -- 对每个邻居执行递归探索
                    foldFunc (accVisited, accSum) neighbor =
                        let (newVisited, result) = exploreNeighbors neighbor accVisited
                        in (newVisited, accSum + result)  -- 累加结果并更新visited状态
                    (finalVisited, total) = foldl foldFunc (visited', 0) neighbors
                in (finalVisited, total + 1)  -- 包括当前像素