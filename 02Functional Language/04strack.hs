data Stack a = Stack [a] deriving (Eq, Ord)
-- 创建空栈
empty :: Stack a -> Stack a
empty _ = Stack []
-- 向栈顶压入一个元素，返回
push :: a -> Stack a -> Stack a
push x (Stack s) = Stack (x : s)
-- 从栈顶弹出一个元素
pop :: Stack a -> (Maybe a, Stack a)
-- (1) 如果栈空，返回一个空值和原始栈
pop (Stack []) = (Nothing, Stack [])
-- 如果非空，返回栈顶元素以及更新后的栈
pop (Stack (x : xs)) = (Just x, Stack xs)

-- 图数据结构
type BinaryImage = [[Int]]
type Coord = (Int, Int)
type Node = (Coord, Int)
type Edge = (Node, [Node])
-- 自定义图：每个坐标映射到其邻接坐标的列表
newtype Graph = Graph [Edge] deriving (Show)

-- 从二维数组构建图
buildGraph :: BinaryImage -> Graph
buildGraph image =
    let rowsNum = length image
        colsNum = length (head image)
        allCoords :: [(Int, Int)]
        allCoords = [(x, y) | x <- [0..rowsNum - 1], y <- [0..colsNum - 1]]

        -- 工具函数
        -- 判断是否在边界内
        ifInBounds :: Coord -> Bool
        ifInBounds (x, y) = x >= 0 && y >= 0 && x < rowsNum && y < colsNum
        -- 获取邻居坐标
        getNeighbors :: Coord -> [Coord]
        getNeighbors (x, y) = filter ifInBounds [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

        -- 构建图
        buildEdge ::  Coord -> Edge
        buildEdge coord =
            let value = image !! fst coord !! snd coord
                node = (coord, value)
                neighbors = [(nc, image !! fst nc !! snd nc) | nc <- getNeighbors coord, ifInBounds nc]
            in (node, neighbors)
        edges = map buildEdge allCoords
    in Graph edges

-- 深度优先搜索
dfs :: Graph -> Coord -> [Coord] -> Int -> (Int, [Coord])
dfs (Graph edges) coord visited v =
    -- 如果当前坐标未访问过且值等于目标值v，则进行深度优先搜索
    if notElem coord visited && (getValue coord edges == Just v) then
        -- 对于每个邻居，如果它的值等于v且未被访问过，继续递归搜索
        let visited' = coord : visited
            neighbors = getNeighbors coord edges
            -- 定义foldl的折叠函数
            foldFunc (accSize, accVisited) neighbor =
                if notElem neighbor accVisited && (getValue neighbor edges == Just v) then
                    -- 递归调用dfs函数，并将搜索结果累加到accSize和accVisited中
                    let (size, newVisited) = dfs (Graph edges) neighbor accVisited v
                    in (accSize + size, newVisited)
                else (accSize, accVisited)
            -- 对所有未访问过的邻居执行折叠操作，计算连通区域的大小和更新访问列表
            (totalSize, finalVisited) = foldl foldFunc (1, visited') neighbors  -- 包含当前节点
        in (totalSize, finalVisited)
    -- 如果当前坐标已经访问过或者值不等于目标值v，则返回0和原始访问列表
    else (0, visited)
dfsSearch node = 
    visited = []
    let



nlcc :: BinaryImage -> Int -> Int
nlcc l v = 
    let
        graph = buildGraph l
        -- DFS搜索，请完善
        result = map dfsSearch graph

