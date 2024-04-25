import Data.Array
-- 1 Define some type synonyms.
-- 1.1 A binary picture, meaning a two-dimensional list formed by integers.
type BinaryImage = [[Int]]
-- 1.2 The coordinates of a pixel.
type Coord = (Int, Int)
-- 1.3 Images represented by arrays.
type ImageArray = Array Coord Int
-- 1.4 A visit record list, recording whether each pixel has been visited.
type Visited = Array Coord Bool


-- 2 Optimisation function
-- Transforming a two-dimensional list into an array.
-- Before optimising, it just used binary list: type Visited = [[Bool]]
createImageArray :: BinaryImage -> ImageArray
createImageArray list = listArray ((0, 0), (rows - 1, cols - 1)) (concat list)
  where
    rows = length list
    cols = length (head list)


-- 3 Main Function
-- Function: Finds the maximum number of connected components of a binary image.
-- Arguments: l: a list of binary images.   v: A value to find (0 or 1).
-- Return: the maximum number of connected components for this image.

-- Workflow:
-- Use the map function to apply the exploreNeighbors helper function to each coordinate.
-- This helper function accepts two arguments, the coordinates and an arrary of pixel visits, and returns a tuple.
-- Then it takes the second value of the tuple which is the number of connections for this pixel.
-- So we have the number of connections for all the pixels.
-- Finally we take the maximum value to get the maximum number of connections for the image.
nlcc :: BinaryImage -> Int -> Int
nlcc l v = maximum $ map (\coord -> snd $ exploreNeighbors coord visited) allCoords
    where
        -- 3.1 Define local variables.
        -- (1) Get the number of rows and columns of this binary image.
        rowsNum = length l
        colsNum = length (head l)
        -- (2) Create a list to store the coordinates of all pixels.
        allCoords = [(x, y) | x <- [0..rowsNum - 1], y <- [0..colsNum - 1]]
        -- (3) Create an visit array and set the default values all to false.
        imageArray = createImageArray l
        visited = listArray (bounds imageArray) (repeat False)

        -- 3.2 Define helper functions.
        -- 3.2.1 Determine if the coordinate is beyond the image boundary.
        ifInBounds :: Coord -> Bool
        ifInBounds (x, y) = x >= 0 && y >= 0 && x < rowsNum && y < colsNum

        -- 3.2.2 Get the coordinates of the neighbours of this pixel.
        getNeighbors :: Coord -> [Coord]
        getNeighbors (x, y) = filter ifInBounds [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

        -- 3.2.3 Updating the visit list.
        -- It seems that haskell probably discourage modifying existing values in a list, and might instead promote creating new values.
        -- So I have tried another approach to achieve similar function to keep the list update.
        -- Basically, the idea is that every time an element is visited, it would replace the value at the specified position, 
        -- and keep the value at the rest of the list unchanged, thus creating a new list to keep the latest status of the list.
        -- The old code is as followed:
        --      let (beforeRows, thisRow : afterRows) = splitAt x vl
        --          updatedThisRow = take y thisRow ++ [True] ++ drop (y + 1) thisRow
        --          in beforeRows ++ [updatedThisRow] ++ afterRows
        -- But this is very performance intensive, so I import the Data.Array to directly modify the values in the array,
        -- to mark pixels which have already been visited.

        updateVisitList :: Coord -> Visited -> Visited
        updateVisitList (x, y) oldVisitList =
          oldVisitList // [((x, y), True)]

        -- 3.2.4 Explore neighbors and calculate the number of connected components for this pixel.
        -- Using the depth-first search algorithm(DFS).
        -- Parameters: coordinates of this pixel and a visit list.
        exploreNeighbors :: Coord -> Visited -> (Visited, Int)
        exploreNeighbors (x, y) visited
            -- If the current pixel is not in the boundaries, or has been visited, or is not the search value v, then skip.
            | not (ifInBounds (x, y)) || visited ! (x, y) || imageArray ! (x, y) /= v    = (visited, 0)
            -- Otherwise, explore neighbors.
            | otherwise =
                let
                    -- Updating the visit list.
                    visited' = updateVisitList (x, y) visited
                    -- Finding Neighbors.
                    neighbors = getNeighbors (x, y)
                    -- Wrap function: complete recursive exploration of neighbours and accumulate results
                    computingResult (thisVisited, thisSum) neighbor =
                        -- Recursive Exploration
                        let (newVisited, result) = exploreNeighbors neighbor thisVisited
                        in (newVisited, thisSum + result)
                    -- Running the explore function to get the accumulation result
                    (finalVisited, total) = foldl computingResult (visited', 0) neighbors
                -- Counting the final result with the current pixel
                in (finalVisited, total + 1)