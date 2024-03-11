person(marina; willem; bob; tina; bert; jane; alyssa).
tables(1..3).

1{seating(People, Table) : tables(Table)}1 :- person(People).
2{seating(People, Table) : person(People)}3 :- tables(Table).
:- seating(marina, Table), seating(willem, Table).