clear all;
clear classes
x = sdpvar(2,2);
objective = x( 1 , 1 )+ x( 1 , 2)+ x( 2 , 1) + x( 2 , 2)
constraints1 = x( 1 , 1 ) >=2;
constraints2 = x( 1 , 2 ) >=2;
constraints3 = x( 2 , 1 ) >=2;
constraints4 = x( 2 , 2 ) >=2;
constraints = [constraints1,constraints2,constraints3,constraints4];
ops = sdpsettings('solver','cplex');
solvesdp(constraints,objective,ops)
double(x)

x
double(objective)
