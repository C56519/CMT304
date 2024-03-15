doubleX x =
    x * 2

doubleXY x y =
    doubleX x + doubleX y

checkifdoubleX x =
    if x > 100
        then x
        else x * 2