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
-- 像素坐标
type BinaryImage = [[Int]]
type Coord = (Int, Int)
type Node = (Coord, Int)
type Edge = [Node]

-- 自定义图：每个坐标映射到其邻接坐标的列表
data Graph = Graph [Node] [Edge] deriving (Show)

-- 创建一个空图
createEmptyGraph :: Graph
createEmptyGraph = Graph [] []

ifInBounds :: Coord -> Bool
ifInBounds (x, y) = x >= 0 && y >= 0 && x < rowsNum && y < colsNum

getNeighbors :: Coord -> [Coord]
getNeighbors (x, y) = filter ifInBounds [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]


-- 从二维数组构建图
buildGraph :: BinaryImage -> Graph
buildGraph image =
    let rowsNum = length image
        colsNum = length (head image)
        allcoords :: [(Int, Int)]
        allcoords = [(x, y) | x <- [0..rowsNum - 1], y <- [0..colsNum - 1]]

        -- 工具函数
        -- 判断是否在边界内
        ifInBounds :: Coord -> Bool
        ifInBounds (x, y) = x >= 0 && y >= 0 && x < rowsNum && y < colsNum
        -- 获取邻居坐标
        getNeighbors :: Coord -> [Coord]
        getNeighbors (x, y) = filter ifInBounds [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

        -- 构建图
        -- nodes 列表
        nodes = [(coord, image !! fst coord !! snd coord) | coord <- allCoords]
        -- edge 列表
        edges = []
    in foldr (\(a, b) g -> addEdge a b g) createEmptyGraph edges