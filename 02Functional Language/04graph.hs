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

