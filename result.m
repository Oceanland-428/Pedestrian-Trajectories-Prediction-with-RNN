X_pred = zeros(1,220);
Y_pred = zeros(1,220);
X_real = zeros(1,220);
Y_real = zeros(1,220);
x = zeros (1,10);
y = zeros (1,10);
index = [52, 13, 3, 31, 64, 193, 17, 70, 77, 197]
index = index+1;
for i = index
    a = strcat('VarName',num2str(i));
    b =eval(a);
    X_pred(1,i) = b(1,1);
    Y_pred(1,i) = b(2,1);
    X_real(1,i) = b(3,1);
    Y_real(1,i) = b(4,1);
end

x_p = X_pred(find(X_pred~=0));
y_p = Y_pred(find(Y_pred~=0));
x_r = X_real(find(X_real~=0));
y_r = Y_real(find(Y_real~=0));
x_test = zeros(1,50);
y_test = zeros(1,50);
for j = 0:49
    c = strcat('e',num2str(j));
    d = eval(c);
    x_test(1,j+1)=d(1,1);
    y_test(1,j+1)=d(2,1);
end

a = [1,2,4,5];
b = [a(1:2) 3 a(3:end)];
x_final_pred = zeros (1,60)
y_final_pred = zeros (1,60)
x_final_real = zeros (1,60)
y_final_real = zeros (1,60)
for q = 1:10
    x_final_pred((6*(q-1)+1):(6*q))= [x_test((5*(q-1)+1):5*q),x_p(q)];
    y_final_pred((6*(q-1)+1):(6*q))= [y_test((5*(q-1)+1):5*q),y_p(q)];
    x_final_real((6*(q-1)+1):(6*q))= [x_test((5*(q-1)+1):5*q),x_r(q)];
    y_final_real((6*(q-1)+1):(6*q))= [y_test((5*(q-1)+1):5*q),y_r(q)];
end

figure (1)
for q = 1:10
    plot(x_final_pred((6*(q-1)+1):6*q),y_final_pred((6*(q-1)+1):6*q));
    hold on;
    plot(x_final_pred(6*q),y_final_pred(6*q),'o');
    hold on;
    plot(x_final_real((6*(q-1)+1):6*q),y_final_real((6*(q-1)+1):6*q));
    hold on;
    plot(x_final_real(6*q),y_final_real(6*q),'x');
    hold on;
end
axis([0 1 0 1])
