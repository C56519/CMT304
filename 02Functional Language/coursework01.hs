import Data.Array
import Data.List (foldl')

type BinaryImage = [[Int]] -- 二维列表表示的图像
type ImageArray = Array (Int, Int) Int -- 数组表示的图像
type Visited = Array (Int, Int) Bool
type Coord = (Int, Int)

-- 将二维列表转换为数组
createImageArray :: BinaryImage -> ImageArray
createImageArray l = listArray ((0, 0), (rows - 1, cols - 1)) (concat l)
  where
    rows = length l
    cols = length (head l)

-- 使用数组重写的主函数
nlcc :: BinaryImage -> Int -> Int
nlcc l v = maximum $ map (\coord -> snd $ exploreNeighbors coord visited) allCoords
  where
    imageArray = createImageArray l
    ((_, _), (xMax, yMax)) = bounds imageArray
    
    allCoords = range $ bounds imageArray
    visited = listArray (bounds imageArray) (repeat False)

    ifInBounds :: Coord -> Bool
    ifInBounds (x, y) = x >= 0 && y >= 0 && x <= xMax && y <= yMax

    getNeighbors :: Coord -> [Coord]
    getNeighbors (x, y) = filter ifInBounds [(x-1, y), (x+1, y), (x, y-1), (x, y + 1)]

    updateVisitList :: Coord -> Visited -> Visited
    updateVisitList (x, y) vl = vl // [((x, y), True)]

    exploreNeighbors :: Coord -> Visited -> (Visited, Int)
    exploreNeighbors (x, y) vis
      | not (ifInBounds (x, y)) = (vis, 0)
      | vis ! (x, y) = (vis, 0)
      | imageArray ! (x, y) /= v = (vis, 0)
      | otherwise =
          let
            vis' = updateVisitList (x, y) vis
            neighbors = getNeighbors (x, y)
            computingResult (vi, sum) coord = 
              let (newVi, result) = exploreNeighbors coord vi
              in (newVi, sum + result)
            (finalVis, total) = foldl' computingResult (vis', 1) neighbors
          in (finalVis, total)