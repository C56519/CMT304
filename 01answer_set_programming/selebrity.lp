% 通过一份聚会人员名单，找到聚会中所有人都认识的名人，并且名人只认识名人，且至少存在一位名人

% 名单
knows(adam,  dan;adam,alice;adam,peter;adam,eva;adam,hakan;adam,anna;adam,agatha).
knows(dan,   adam;dan,alice;dan,peter;dan,hakan;dan,anna;dan,agatha).
knows(eva,   alice;eva,peter;eva,adam;eva,hakan;eva,anna;eva,agatha).
knows(alice, peter;alice,adam;alice,hakan;alice,anna;alice,agatha).
knows(peter, alice;peter,adam;peter,hakan;peter,anna;peter,agatha).
knows(hakan, anna;hakan,agatha).
knows(anna,  hakan;anna,agatha).
knows(agatha,  hakan;agatha,anna).

% 1. 得到所有人的列表，并统计所有人数量
person(X) :- knows(X, _).
person(X) :- knows(_, X).
num_p(N) :- N = #count {P : person(P)}.

% 2. rule1: 规定什么是 名人
% 对于名人C，首先是个人，然后对于N个人，其他N-1个人中，每个人P都认识C
celebrity(C) :- person(C), num_p(N), N-1 {knows(P, C) : person(P)} N-1.

% rule2: 名人只认识名人
% 如果C是名人，并且是个人，然后其他任何人P都不是名人，但是名人C认识P，那么规则不成立
:- celebrity(C), person(C), not celebrity(P), knows(C, P).

#show celebrity/1.