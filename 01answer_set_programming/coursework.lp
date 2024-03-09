% 1 首先，考虑所有可能的分配方案
% 1.1 元素和属性分析
% (1) 过程：每个请求req -> 成员mem
% (2) 属性：对于每一个请求，每个成员都需要声明对该req领域的熟悉程度，四选一(EXPERT专家, KNOWLEDGEABLE知识渊博, FAMILIAR熟悉, INEXPERT不熟悉)

% 2 增加限制条件
% 2.1 确保没有分配req给声明了“INEXPERT不熟悉”的成员
% 2.2 确保每个成员分配了“FAMILIAR熟悉”的请求数量不超过 k
% 2.3 尽可能将请求分配给声明了“EXPERT专家”的成员
% 2.4 保持所有成员被分配工作量的大致平衡 -> 任务量差距不超过 m
% 2.5 成员数量总数为 n

% 定义专业知识等级
% exp(expert).
% exp(knowledgeable).
% exp(familiar).
% exp(inexpert).
% test: input
#const k = 3.
#const m = 2.
% rule1: 确保没有分配req给声明了“INEXPERT不熟悉”的成员
% 如果 Mem 对请求 Req 的 Exp 不是 inexpert，那么就将Mem 分配给请求 Req。
assign(Mem, Req) :- bid(Mem, Req, Exp), Exp != inexpert.

% rule2: 确保每个成员分配了“FAMILIAR熟悉”的请求数量不超过 k
% 如果成员 Mem 对熟悉程度为 familiar 的请求的分配数量超过了 k，那么成员 Mem 将不再被分配竞标价格为 familiar 的请求。
% 对于每一个成员 Mem，如果该成员有bid()，则该成员被认为是一个成员
member(Mem) :- bid(Mem, _, _).
:- member(Mem), #count { Req: assign(Mem, Req), Req = familiar } > k.

% rule3: 尽可能将请求分配给声明了“EXPERT专家”的成员
#maximize { 1@2, Mem, Req : assign(Mem, Req), bid(Mem, Req, expert)}.

% rule4: 保持所有成员被分配工作量的大致平衡 -> 任务量差距不超过 m
:- member(Mem1), member(Mem2), Mem1 != Mem2,
    X = #count { Req : assign(Mem1, Req) },
    Y = #count { Req : assign(Mem2, Req) },
    abs(X - Y) > m.



bid(mem1, req1, expert).
bid(mem1, req2, inexpert).
bid(mem1, req3, expert).
bid(mem1, req4, knowledgeable).
bid(mem1, req5, expert).
bid(mem1, req6, inexpert).
bid(mem1, req7, familiar).
bid(mem1, req8, knowledgeable).

bid(mem2, req1, familiar).
bid(mem2, req2, knowledgeable).
bid(mem2, req3, familiar).
bid(mem2, req4, inexpert).
bid(mem2, req5, inexpert).
bid(mem2, req6, knowledgeable).
bid(mem2, req7, expert).
bid(mem2, req8, inexpert).

bid(mem3, req1, inexpert).
bid(mem3, req2, expert).
bid(mem3, req3, inexpert).
bid(mem3, req4, knowledgeable).
bid(mem3, req5, inexpert).
bid(mem3, req6, knowledgeable).
bid(mem3, req7, knowledgeable).
bid(mem3, req8, familiar).

bid(mem4, req1, familiar).
bid(mem4, req2, knowledgeable).
bid(mem4, req3, familiar).
bid(mem4, req4, familiar).
bid(mem4, req5, knowledgeable).
bid(mem4, req6, inexpert).
bid(mem4, req7, inexpert).
bid(mem4, req8, inexpert).

bid(mem5, req1, inexpert).
bid(mem5, req2, expert).
bid(mem5, req3, expert).
bid(mem5, req4, inexpert).
bid(mem5, req5, knowledgeable).
bid(mem5, req6, inexpert).
bid(mem5, req7, familiar).
bid(mem5, req8, familiar).

bid(mem6, req1, familiar).
bid(mem6, req2, expert).
bid(mem6, req3, familiar).
bid(mem6, req4, familiar).
bid(mem6, req5, knowledgeable).
bid(mem6, req6, inexpert).
bid(mem6, req7, expert).
bid(mem6, req8, expert).