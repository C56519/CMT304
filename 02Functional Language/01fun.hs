doubleX :: Int -> Int
doubleX x =
    x * 2

doubleXY :: Int -> Int -> Int
doubleXY x y =
    doubleX x + doubleX y

checkifdoubleX :: Int -> Int
checkifdoubleX x =
    if x > 100
        then x
        else x * 2