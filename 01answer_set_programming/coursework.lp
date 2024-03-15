#const n = 2.
#const k = 3.     % Maximum number of "familiar" category submissions reviewed by each member.
#const m = 1.     % The workload differences among members do not exceed the difference k.

% All members and submissions.
member(P) :- bid(P, _, _).
request(R) :- bid(_, R, _).

% Generate all possible assignments between members and requests
n{ assign(P, R) : member(P) }n :- request(R).
:- assign(P, R), bid(P, R, inexpert).
:- member(P), #count { R : assign(P, R), bid(P, R, familiar) } > k.

member_assigned(P, N) :- member(P), N = #count { R: assign(P, R) }.
% (2) For two different members, it is not allowed that the absolute difference in the number of requests assigned to them is greater than m.
:- member(X), member(Y), X != Y,
    member_assigned(X, N1),
    member_assigned(Y, N2), |N1-N2| > m.

#maximize {1@3, P, R : assign(P, R), bid(P, R, expert)}.
#maximize {1@2, P, R : assign(P, R), bid(P, R, knowledgeable)}.
#maximize {1@1, P, R : assign(P, R), bid(P, R, familiar)}.

#show assign/2.

% inputs
bid(mem1, req1, expert).
bid(mem2, req1, familiar).
bid(mem3, req1, inexpert).

bid(mem1, req2, expert).
bid(mem2, req2, inexpert).
bid(mem3, req2, knowledgeable).

bid(mem1, req3, expert).
bid(mem2, req3, familiar).
bid(mem3, req3, expert).

bid(mem1, req4, knowledgeable).
bid(mem2, req4, inexpert).
bid(mem3, req4, knowledgeable).

bid(mem1, req5, expert).
bid(mem2, req5, expert).
bid(mem3, req5, inexpert).

bid(mem1, req6, inexpert).
bid(mem2, req6, knowledgeable).
bid(mem3, req6, knowledgeable).

bid(mem1, req7, expert).
bid(mem2, req7, familiar).
bid(mem3, req7, expert).

bid(mem1, req8, inexpert).
bid(mem2, req8, familiar).
bid(mem3, req8, knowledgeable).
