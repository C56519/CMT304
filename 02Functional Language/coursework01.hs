-- 1 Define some type synonyms.
-- 1.1 A binary picture, meaning a two-dimensional list formed by integers.
type BinaryImage = [[Int]]
-- 1.2 The coordinates of a pixel.
type Coord = (Int, Int)
-- 1.3 A visit record list, recording whether each pixel has been visited.
type Visited = [[Bool]]
-- 工具函数
-- 判断是否在边界内
ifInBounds :: Coord -> Bool
ifInBounds (x, y) = x >= 0 && y >= 0 && x < rowsNum && y < colsNum
-- 深度优先搜索
-- 使用坐标而非整个图进行DFS
dfs :: BinaryImage -> Coord -> Int -> [[Bool]] -> (Int, [[Bool]])
dfs image (x, y) v visited
  | not (ifInBounds (x, y)) || visited !! x !! y || image !! x !! y /= v = (0, visited)
  | otherwise = let
      newVisited = (take x visited) ++ [take y (visited !! x) ++ [True] ++ drop (y + 1) (visited !! x)] ++ (drop (x + 1) visited)
      directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
      (count, finalVisited) = foldl (\(c, vst) (dx, dy) -> let (nc, nvst) = dfs image (x + dx, y + dy) v vst in (c + nc, nvst)) (1, newVisited) directions
    in (count, finalVisited)

-- 主函数，开始DFS并找到最大连通组件
nlcc :: BinaryImage -> Int -> Int
nlcc l v =
  let
    rows = length l
    cols = if rows == 0 then 0 else length (head l)
    visited = replicate rows (replicate cols False)
    allCoords = [(x, y) | x <- [0..rows - 1], y <- [0..cols - 1]]
    
    -- 对每个未访问的点执行DFS，寻找最大连通组件
    dfsAll :: [(Int, Int)] -> [[Bool]] -> Int -> Int
    dfsAll [] _ maxCount = maxCount
    dfsAll ((x, y):cs) visited maxCount =
      if visited !! x !! y || l !! x !! y /= v then dfsAll cs visited maxCount
      else
        let (count, newVisited) = dfs l (x, y) v visited
        in dfsAll cs newVisited (max maxCount count)

  in dfsAll allCoords visited 0