% family relationship

% 平行关系
married(X, Y) :- married(Y, X).
% stronger
% (X, Y) => XX的丈夫是YY
married(X, Y) :- husband(X, Y).
married(X, Y) :- wife(X, Y).
father(X, Z) :- parent(X, Y), husband(Y, Z).
mother(X, Z) :- parent(X, Y), wife(Y, Z). 
husband(Y, Z) :- mother(X, Y), married(Y, Z).
wife(Y, Z) :- father(X, Y), married(Y, Z).

% 上级关系
parent(X, Y) :- father(X, Y).
parent(X, Y) :- mother(X, Y).
mother(X, Z) :- father(X, Y), married(Y, Z).
father(X, Z) :- mother(X, Y), married(Y, Z).
parent(X, Z) :- parent(X, Y), married(Y, Z).
% 描述隔代
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- ancestor(X, Y), ancestor(Y, Z).

% input
father(bill, john).
mother(john, marry).
married(marry, matt).
married(john, jone).