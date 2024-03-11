person(dana; cody, abby, bess).

% Dana likes Cody.
likes(dana, cody).
% Abby does not like Dana.
:- likes(abby, dana).
% Dana does not like Abby.
:- likes(dana, abby).
% Bess likes Cody or Dana.
1{likes(bess, cody); likes(bess, dana)}.
% Abby likes everyone that Bess likes.
likes(abby, Person) :- likes(bess, Person).
% Cody likes everyone who likes her.
likes(cody, Person) :- likes(Person, cody).
% Nobody likes herself.
:- likes(Person, Person).

% 对于每个人 X，选择至少1个人，至多 4 个人 Y 作为 X 的喜欢对象
1{like(X, Y) : person(Y)}4 :- person(X).
% #show指令控制显示哪些内容
% likes/2 谓词/元数
% 整体上是只显示 likes谓词中有两个参数的结果
#show likes/2.