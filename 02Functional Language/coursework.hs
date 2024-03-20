-- 1 Define some type synonyms.
-- 1.1 A binary picture, meaning a two-dimensional list formed by integers.
type BinaryImage = [[Int]]
-- 1.2 The coordinates of a pixel.
type Coord = (Int, Int)
-- 1.3 A visit record list, recording whether each pixel has been visited.
type Visited = [[Bool]]

-- 2 Main Functions.
-- Function: Finds the maximum number of connected components of a binary image.
-- Arguments: l: a list of binary images.   v: A value to find (0 or 1).
-- Return: the maximum number of connected components for this image.

-- Workflow:
-- Use the map function to apply the exploreNeighbors helper function to each coordinate.
-- This helper function accepts two arguments, the coordinates and a list of pixel visits, and returns a tuple.
-- Then it takes the second value of the tuple which is the number of connections for this pixel.
-- So we have the number of connections for all the pixels.
-- Finally we take the maximum value to get the maximum number of connections for the image.
nlcc :: BinaryImage -> Int -> Int
nlcc l v = maximum $ map (\coord -> snd $ exploreNeighbors coord visited) allCoords
    where
        -- 2.1 Define local variables.
        -- (1) Get the number of rows and columns of this binary image.
        rowsNum = length l
        colsNum = length (head l)
        -- (2) Create a list to store the coordinates of all pixels.
        allCoords = [(x, y) | x <- [0..rowsNum - 1], y <- [0..colsNum - 1]]
        -- (3) Create an visit list and set the default values all to false.
        visited = [[False | _ <- [0..colsNum - 1]] | _ <- [0..rowsNum - 1]]

        -- 2.2 Define helper functions.
        -- 2.2.1 Determine if the coordinate is beyond the image boundary.
        ifInBounds :: Coord -> Bool
        ifInBounds (x, y) = x >= 0 && y >= 0 && x < rowsNum && y < colsNum

        -- 2.2.2 Get the coordinates of the neighbours of this pixel.
        getNeighbors :: Coord -> [Coord]
        getNeighbors (x, y) = filter ifInBounds [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

        -- 2.2.3 Updating the visit list.
        -- It seems that haskell probably discourage modifying existing values in a list, and might instead promote creating new values.
        -- So I tried another approach to achieve similar function to keep the list update.
        -- Basically, the idea is that every time an element is visited, it would replace the value at the specified position, 
        -- and keep the value at the rest of the list unchanged, thus creating a new list to keep the latest status of the list.
        -- Parameters: x, y: coordinates     vl: visit list
        -- Return: New visit list
        updateVisitList :: Int -> Int -> Visited -> Visited
        updateVisitList x y vl =
          -- (1) Split the list into three parts.
          -- The current line, the part of the list before the current line, and the part of the list after the current line.
          let (beforeRows, thisRow : afterRows) = splitAt x vl
              -- (2) Process the current row list according to three parts.
              -- Get the element before the current column, get the element after the current column.
              -- And change the value of the element in the current column to True.
              -- Finally append the three parts into a row list.
              updatedThisRow = take y thisRow ++ [True] ++ drop (y + 1) thisRow
            -- (3) Append the three row list into a new visit list.
          in beforeRows ++ [updatedThisRow] ++ afterRows

        -- 2.2.4 Explore neighbors and calculate the number of connected components for this pixel.
        -- Parameters: coordinates of this pixel and a visit list.
        exploreNeighbors :: Coord -> Visited -> (Visited, Int)
        exploreNeighbors (x, y) visited
            -- If the current pixel is not in the boundaries, or has been visited, or is not the search value v, then skip.
            | not (ifInBounds (x, y)) || visited !! x !! y || l !! x !! y /= v    = (visited, 0)
            -- Otherwise, explore neighbors.
            | otherwise =
                let
                    -- Updating the visit list.
                    visited' = updateVisitList x y visited
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