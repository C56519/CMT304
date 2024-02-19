[toc]

# Rules

Answer set programming is a programme about setting rules

```asp
large(C) :- size(C,S1), size(uk,S2), S1 > S2.
```

1. 结构
   - head:      large(C)
   - :-      if
   - body:      size(C,S1), size(uk,S2), S1 > S2.
     - ,      逗号 相当于 and
2. 翻译成英语：
   - A country C is large: if the population size of C is S1, the population size of the UK is S2, and S1 > S2.
3. 注：这不是命令，这是有关large(C)的声明式解释，也算一种 知识表示