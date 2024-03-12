#const k=2.
#const m=2.

member(P) :- bid(P, _, _).
request(R) :- bid(_, R, _).

% 1 确保每个req分发到一个mem
1{ assign(P, R) : member(P) }1 :- request(R).
% 2 确保inexpert的mem不分发req
:- assign(P, R), bid(P, R, inexpert).
% 3 确保每个成员分配了“FAMILIAR熟悉”的请求数量不超过 k
:- member(P), #count { R : assign(P, R), bid(P, R, familiar) } > k.

% 4 尽可能将请求分配给声明了“EXPERT专家”的成员，并按照熟悉度来分配
priority(expert, 3; knowledgeable, 2; familiar, 1).
% 计算每个请求的最大优先级
max_priority_for_request(R, M) :- request(R), M = #max { L, P : bid(P, R, E), priority(E, L) }.
% 确保只有最大优先级的成员被分配到请求
:- assign(P, R), bid(P, R, E), priority(E, L), max_priority_for_request(R, M), L < M.

% 5 保证所有成员被分配工作量的大致平衡 -> 任务量差距不超过 m
member_assigned(P, N) :- member(P), N = #count { R: assign(P, R) }.
:- member(X), member(Y), X != Y, member_assigned(X, N1), member_assigned(Y, N2), |N1-N2| > m.

#show assign/2.

% inputs
bid(mem1, req1, expert).
bid(mem2, req1, familiar).
bid(mem3, req1, inexpert).
bid(mem4, req1, familiar).
bid(mem5, req1, inexpert).
bid(mem6, req1, familiar).

bid(mem1, req2, expert).
bid(mem2, req2, inexpert).
bid(mem3, req2, knowledgeable).
bid(mem4, req2, knowledgeable).
bid(mem5, req2, expert).
bid(mem6, req2, expert).

bid(mem1, req3, expert).
bid(mem2, req3, familiar).
bid(mem3, req3, expert).
bid(mem4, req3, familiar).
bid(mem5, req3, inexpert).
bid(mem6, req3, familiar).

bid(mem1, req4, knowledgeable).
bid(mem2, req4, inexpert).
bid(mem3, req4, knowledgeable).
bid(mem4, req4, familiar).
bid(mem5, req4, knowledgeable).
bid(mem6, req4, familiar).

bid(mem1, req5, expert).
bid(mem2, req5, expert).
bid(mem3, req5, inexpert).
bid(mem4, req5, knowledgeable).
bid(mem5, req5, knowledgeable).
bid(mem6, req7, expert).

bid(mem1, req6, inexpert).
bid(mem2, req6, knowledgeable).
bid(mem3, req6, knowledgeable).
bid(mem4, req6, inexpert).
bid(mem5, req6, inexpert).
bid(mem6, req6, inexpert).

bid(mem1, req7, expert).
bid(mem2, req7, familiar).
bid(mem3, req7, expert).
bid(mem4, req7, knowledgeable).
bid(mem5, req7, inexpert).
bid(mem6, req7, familiar).

bid(mem1, req8, inexpert).
bid(mem2, req8, familiar).
bid(mem3, req8, knowledgeable).
bid(mem4, req8, inexpert).
bid(mem5, req8, familiar).
bid(mem6, req8, expert).