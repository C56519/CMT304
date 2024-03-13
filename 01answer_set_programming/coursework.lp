% CMT304 Logic Programming Coursework
% Student Name: Wenxuan Chen
% Student ID: 23084626

% Define constants
#const n = 2.     % The number of members of the technical board to assign to each request.
#const k = 2.     % Maximum number of "familiar" category submissions reviewed by each member.
#const m = 1.     % The workload differences among members do not exceed the difference k.

% All members and submissions.
member(P) :- bid(P, _, _).
request(R) :- bid(_, R, _).

% Rule 1 (Optimise): Ensure that each req(submission) is distributed to a mem(member).
% Approach: Specify that each submission can only be allocated to one individual from the group of members.
n{ assign(P, R) : member(P) }n :- request(R).

% Rule 2: No member of the technical board is required to review a submission that is placed in the "inexpert" category.
% Approach: The assignment is cancelled, if
% (1) A member is assigned a request.
% (2) And the member's expertise for the request is "inexpert".
:- assign(P, R), bid(P, R, inexpert).

% Rule 3: No member of the technical board is required to review more than k submissions from the "familiar" category.
% Approach:
% (1) For each member P, calculate the number of assigned requests with category "familiar".
% (2) If this number exceeds the pre-defined k, the rule is activated and the allocation is prevented.
:- member(P), #count { R : assign(P, R), bid(P, R, familiar) } > k.

% Rule 4: The total number of cases when a submission is assigned to a member who placed it in the "expert" category is as large as possible.
% Approach: (Optimise added)
% (1) 
% (2) 
#maximize {1@3, P, R : assign(P, R), bid(P, R, expert)}.
#maximize {1@2, P, R : assign(P, R), bid(P, R, knowledgeable)}.
#maximize {1@1, P, R : assign(P, R), bid(P, R, familiar)}.

% Rule 5: The workloads of the technical board members are approximately equal, that is, do not differ by more than m.
% Approach:
% (1) For each member, compute the number of requests assigned to him.
member_assigned(P, N) :- member(P), N = #count { R: assign(P, R) }.
% (2) For two different members, it is not allowed that the absolute difference in the number of requests assigned to them is greater than m.
:- member(X), member(Y), X != Y,
    member_assigned(X, N1),
    member_assigned(Y, N2), |N1-N2| > m.

%#show assign/2.
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
bid(mem6, req5, expert).

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