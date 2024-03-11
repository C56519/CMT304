member(P) :- bid(P,_,_).
bid(P, R, 4) :- bid(P, R, expert).
bid(P, R, 3) :- bid(P, R, knowledgeable).
bid(P, R, 2) :- bid(P, R, familiar).
bid(P, R, 1) :- bid(P, R, inexpert).

assign(P, R) :- member(P), bid(P, R, E), E != 1.
1 { assign(P, R) : member(P) } 1 :- bid(P, R, _).



#show assign/2.
% input
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