clc
clear all
close all

grayColor = [.7 .7 .7];
%Graph specifics
N = 20;%20
Aj = zeros(N,N);
for i = 2:N
    Aj(i-1,i) = 1;
    Aj(i,i-1) = 1;
end
Aj(1,N) = 1;
Aj(N,1) = 1;
I = eye(N);
D = diag(Aj*ones(N,1));
L = D - Aj;
plot_agent = N;
%defining the Ns GMMS
% the surface is between [-8,80] and [-60,60]
T = 10;
Ns = 12;
scatterness = 6;
pos = [-60 40;-20 40;20 40;60 40;-60 0;-20 0;20 0;60 0;-60 -40;-20 -40;20 -40;60 -40];
for k = 1:Ns
    mu(:,k) = [pos(k,1)+scatterness*randn;pos(k,2)+scatterness*randn];
    sigma(:,:,k) = 8*[randn randn;randn randn];
    sigma(:,:,k) = sigma(:,:,k) + sigma(:,:,k)';
    sigma(:,:,k) = sigma(:,:,k) + (max(abs(eig(sigma(:,:,k))))+15*rand)*eye(2,2);
    color(k,:) = rand(1,3);
end
figure(1)
for k = 1:Ns
    z = ellipsedata(sigma(:,:,k), mu(:,k), 100, 3, 1E-12);
    p1 = plot(z(:,1),z(:,2),'Color',grayColor,'LineWidth',.7);
    hold on
end
figure(2)
for k = 1:Ns
    z = ellipsedata(sigma(:,:,k), mu(:,k), 100, 3, 1E-12);
    plot(z(:,1),z(:,2),'Color',grayColor,'LineWidth',1)
    hold on
end
%creating the data points on the 2D map
figure(1)
M = 100;%1000
agent_data_size = M/N;
x = zeros(2,M);
data_points = zeros(2,Ns);
n_data_points = zeros(1,Ns);
for n = 1:M
    gmm = floor(rand()*Ns)+1;
    x(:,n) = mvnrnd(mu(:,gmm),sigma(:,:,gmm),1);
    data_points(:,gmm) = data_points(:,gmm)+x(:,n);
    n_data_points(1,gmm) = n_data_points(gmm)+1;
    figure(1)
    plot(x(1,n),x(2,n),'x','Color',color(gmm,:))
    hold on
    figure(2)
    plot(x(1,n),x(2,n),'x','Color',color(gmm,:))
    hold on
end
for k = 1:Ns
    data_points(:,k) = data_points(:,k)/n_data_points(1,k);
end
%Initial matrix dimensions
p_hat = zeros(1,Ns);
mu_hat = zeros(2,Ns,1);
sigma_hat = zeros(2,2,Ns,1);
gamma = zeros(Ns,M);
prod2 = zeros(4,M);
T_con = [8,15,30,50];
[~,len_T_con] = size(T_con);
plot_iter_1 = 1;
plot_iter_2 = 2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Consensus Convergence %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%E-step
for n = 1:M
    agent = floor((n-1)/agent_data_size)+1;
    nom1 = p_hat(agent,1)*mvnpdf(x(:,n),mu_hat(:,1,agent),sigma_hat(:,:,1,agent));
    den = 0;
    for j = 1:Ns
        den = den + p_hat(agent,j)*mvnpdf(x(:,n),mu_hat(:,j,agent),sigma_hat(:,:,j,agent));
    end
    gamma(1,n) = nom1/den;
end
%Initialization
Nsim = 50;
reference_value = zeros(N,1);
for i = 1:N
    reference_value(i,1) = sum(gamma((i-1)*agent_data_size+1:i*agent_data_size));
end
ravg = sum(reference_value)/N*ones(N,1);
lambda_eig = sort(eig(L));
lambda_2 = lambda_eig(2);
lambda_N = lambda_eig(end);
%Laplacian
dt_lap = 2/eig(lambda_N);
x_lap = reference_value;
e_lap(1,1) = log(norm(x_lap(:,1)-ravg));
%Delay
dt_delay = dt_lap;
d_delay = 5;
x_delay(:,1:d_delay) = zeros(N,d_delay);
x_delay(:,1+d_delay) = reference_value;
e_delay(1,1+d_delay) = log(norm(x_delay(:,1+d_delay)-ravg));
%NAG_C
dt_NAGC = 1/eig(lambda_N);
x_NAGC = reference_value;
y_NAGC = reference_value;
e_NAGC(1,1) = log(norm(x_NAGC(:,1)-ravg));
%NAG_SC
alpha_NAGSC = 1/eig(lambda_N);
beta_NAGSC = (sqrt(lambda_N)-sqrt(lambda_2))/(sqrt(lambda_N)+sqrt(lambda_2));
x_NAGSC(:,1) = reference_value;
x_NAGSC(:,2) = reference_value;
e_NAGSC(1,1) = log(norm(x_NAGSC(:,2)-ravg));
%TM
ro_tm = 1-sqrt(lambda_2/lambda_N);
alpha_tm = (1+ro_tm)/lambda_N;
beta_tm = ro_tm^2/(2-ro_tm);
gamma_tm = ro_tm^2/(2-ro_tm)/(1+ro_tm);
delta_tm = ro_tm^2/(1+ro_tm)^2;
zeta_tm(:,1) = reference_value;
zeta_tm(:,2) = reference_value;
%M-step
for k = 1:Nsim
    %Laplacian
    x_lap(:,k+1) = x_lap(:,k)-dt_lap*L*x_lap(:,k);
    e_lap(1,k+1) = log(norm(x_delay(:,k+1+del)-ravg));
    %Delay
    x_delay(:,k+1+d_delay) = x_delay(:,k+d_delay)-dt_lap*L*x_delay(:,k);
    e_delay(1,k+1) = log(norm(x_delay(:,k+1+del)-ravg));
    %NAG_C
    y_NAGC(:,k+1) = x_NAGC(:,k) - dt_NAGC*L*x_NAGC(:,k);
    x_NAGC(:,k+1) = y_NAGC(:,k) + (k+1)/(k+3)*(y_NAGC(:,k+1)-y_NAGC(:,k));
    e_NAGC(1,k+1) = log(norm(x_NAGC(:,k+1)-ravg));
    %NAG_SC
    y_NAGSC(:,k) = (1+beta_NAGSC)*x_NAGSC(:,k+1) - beta_NAGSC*x_NAGSC(:,k);
    x_NAGSC(:,k+2) = y_NAGSC(:,k) - alpha_NAGSC*L*y_NAGSC(:,k);
    e_NAGSC(1,k+1) = log(norm(x_NAGC(:,k+2)-ravg));
    %TM
    x_tm(:,k) = (1+delta_tm)*zeta_tm(:,k+1) - delta_tm*zeta_tm(:,k);
    y_tm(:,k) = (1+gamma_tm)*zeta_tm(:,k+1) - gamma_tm*zeta_tm(:,k);
    zeta_tm(:,k+2) = (1+beta_tm)*zeta_tm(:,k+1) - beta_tm*zeta_tm(:,k) - alpha_tm*L*y_tm(:,k);
    e_tm(1,k) = log(norm(x_tm(:,k)-ravg));
end

figure(4)
sim = 0:1:Nsim-1;
v1 = plot(sim,e_tm(1,1:Nsim),'b','LineWidth',2);
hold on
v2 = plot(sim,e_NAGSC(1,1:Nsim),'k','LineWidth',2);
hold on
v3 = plot(sim,e_NAGC(1,1:Nsim),'g','LineWidth',2);
hold on
v4 = plot(sim,e_delay(1,1:Nsim),'m','LineWidth',2);
hold on
v5 = plot(sim,e_lap(1,1:Nsim),'r','LineWidth',2);

set(gcf,'position',[100,100,500,250])
set(gca,'FontSize',20)
%legend([l2(1,1,4),l2(1,1,2),l2(1,1,1),l2(1,1,5),l2(1,1,3)],{'d=5','Algorithm in [15]','NAG-SC','Algorithm in [20]','TM'},'FontSize',15)
xlabel('$k$','fontsize',27,'interpreter','latex')
ylabel('$e(k)$','fontsize',27,'interpreter','latex')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Central consensus %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%running the EM algorithm
%initialization
for k = 1:Ns
    p_hat(1,k) = 1/Ns;
    mu_hat(:,k,1) = [pos(k,1);pos(k,2)];
    sigma_hat(:,:,k,1) = [20 0;0 20];
end
error_cen = zeros(T,1);
error_cen(1,1) = log(norm(data_points-mu_hat(:,:,1))/norm(data_points));
for t = 1:T
    %E-step
    for k = 1:Ns
        for n = 1:M
            nom = p_hat(1,k)*mvnpdf(x(:,n),mu_hat(:,k,1),sigma_hat(:,:,k,1));
            den = 0;
            for j = 1:Ns
                den = den + p_hat(1,j)*mvnpdf(x(:,n),mu_hat(:,j,1),sigma_hat(:,:,j,1));
            end
            gamma(k,n) = nom/den;
        end
    end
    %M-step
    for k = 1:Ns
        gamma_summation = sum(gamma(k,:));
        p_hat(1,k) = gamma_summation/M;
        mu_hat(:,k,1) = sum(gamma(k,:).*x,2)/gamma_summation;
        sigma_summation = zeros(2,2);
        for n = 1:M
            sigma_summation = sigma_summation + gamma(k,n)*(x(:,n)-mu_hat(:,k,1))*(x(:,n)-mu_hat(:,k,1))';
        end
        sigma_hat(:,:,k) = sigma_summation/gamma_summation;
        if all(eig(sigma_hat(:,:,k,1)) > .1)
        else
            sigma_hat(:,:,k,1) = sigma_hat(:,:,k,1) + (max(abs(eig(sigma_hat(:,:,k,1))))+.1)*eye(2,2);
        end
    end
    error_cen(t+1) = log(norm(data_points-mu_hat(:,:,1))/norm(data_points));
end
LL_cen = 0;
for n = 1:M
    LL_cat_cen = 0;
    for k = 1:Ns
        LL_cat_cen = LL_cat_cen+p_hat(1,k)*mvnpdf(x(:,n),mu_hat(:,k,1),sigma_hat(:,:,k,1));
    end
    LL_cen = LL_cen+log(LL_cat_cen);
end

for iter = 1:len_T_con
    T_con_lap = T_con(iter);
    T_con_tm = T_con(iter);
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Laplacian consensus %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %running the EM algorithm
    %initialization
    for k = 1:Ns
        for i = 1:N
            p_hat(i,k) = 1/Ns;
            mu_hat(:,k,i) = [pos(k,1);pos(k,2)];
            sigma_hat(:,:,k,i) = [20 0;0 20];
        end
    end
    error_lap = zeros(T,1);
    error_lap(1,1) = log(norm(data_points-mu_hat(:,:,plot_agent))/norm(data_points));
    for t = 1:T
        %E-step
        for k = 1:Ns
            for n = 1:M
                agent = floor((n-1)/agent_data_size)+1;
                nom1 = p_hat(agent,k)*mvnpdf(x(:,n),mu_hat(:,k,agent),sigma_hat(:,:,k,agent));
                den = 0;
                for j = 1:Ns
                    den = den + p_hat(agent,j)*mvnpdf(x(:,n),mu_hat(:,j,agent),sigma_hat(:,:,j,agent));
                end
                gamma(k,n) = nom1/den;
            end
        end
        %M-step
        for k = 1:Ns
            %pi(k)
            gamma_con_lap = con_lap(gamma(k,:),M,N,L,T_con_lap);
            p_hat(:,k) = gamma_con_lap/agent_data_size;
            %mu(k)
            nom2 = gamma(k,:).*x;
            for j = 1:2
                nom_con_lap = con_lap(nom2(j,:),M,N,L,T_con_lap);
                for i = 1:N
                    mu_hat(j,k,i) = nom_con_lap(i,1)/gamma_con_lap(i,1);
                end
            end
            %sigma(k)
            for n = 1:M
                agent = floor((n-1)/agent_data_size)+1;
                prod1 = (x(:,n)-mu_hat(:,k,agent))*(x(:,n)-mu_hat(:,k,agent))';
                prod2(:,n) = gamma(k,n)*[prod1(1,1);prod1(2,1);prod1(1,2);prod1(2,2)];
            end
            sigma11 = con_lap(prod2(1,:),M,N,L,T_con_lap);
            sigma21 = con_lap(prod2(2,:),M,N,L,T_con_lap);
            sigma12 = con_lap(prod2(3,:),M,N,L,T_con_lap);
            sigma22 = con_lap(prod2(4,:),M,N,L,T_con_lap);
            for i = 1:N
                sigma_hat(1,1,k,i) = sigma11(i,1)./gamma_con_lap(i,1);
                sigma_hat(2,1,k,i) = sigma21(i,1)./gamma_con_lap(i,1);
                sigma_hat(1,2,k,i) = sigma12(i,1)./gamma_con_lap(i,1);
                sigma_hat(2,2,k,i) = sigma22(i,1)./gamma_con_lap(i,1);
                if all(eig(sigma_hat(:,:,k,i)) > .1)
                else
                    sigma_hat(:,:,k,i) = sigma_hat(:,:,k,i) + (max(abs(eig(sigma_hat(:,:,k,i))))+.1)*eye(2,2);
                end
            end
        end
        error_lap(t+1) = log(norm(data_points-mu_hat(:,:,plot_agent))/norm(data_points));
    end
    if iter == plot_iter_1
        figure(1)
        for k = 1:Ns
            z = ellipsedata(sigma_hat(:,:,k,plot_agent), mu_hat(:,k,plot_agent), 100, 3, 1E-12);
            p2 = plot(z(:,1),z(:,2),'Color',color(k,:),'LineWidth',1);
            hold on
        end
    end
    if iter == plot_iter_2
        figure(2)
        for k = 1:Ns
            z = ellipsedata(sigma_hat(:,:,k,plot_agent), mu_hat(:,k,plot_agent), 100, 3, 1E-12);
            plot(z(:,1),z(:,2),'Color',color(k,:),'LineWidth',1)
            hold on
        end
    end
    LL_lap(iter) = 0;
    for n = 1:M
        LL_cat_lap = 0;
        for k = 1:Ns
            LL_cat_lap = LL_cat_lap+p_hat(plot_agent,k)*mvnpdf(x(:,n),mu_hat(:,k,plot_agent),sigma_hat(:,:,k,plot_agent));
        end
        LL_lap(iter) = LL_lap(iter)+log(LL_cat_lap);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TM consensus (unbounded) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %running the EM algorithm
    %initialization
    for k = 1:Ns
        for i = 1:N
            p_hat(i,k) = 1/Ns;
            mu_hat(:,k,i) = [pos(k,1);pos(k,2)];
            sigma_hat(:,:,k,i) = [20 0;0 20];
        end
    end
    error_tm = zeros(T,1);
    error_tm(1,1) = log(norm(data_points-mu_hat(:,:,plot_agent))/norm(data_points));
    for t = 1:T
        %E-step
        for k = 1:Ns
            for n = 1:M
                agent = floor((n-1)/agent_data_size)+1;
                nom1 = p_hat(agent,k)*mvnpdf(x(:,n),mu_hat(:,k,agent),sigma_hat(:,:,k,agent));
                den = 0;
                for j = 1:Ns
                    den = den + p_hat(agent,j)*mvnpdf(x(:,n),mu_hat(:,j,agent),sigma_hat(:,:,j,agent));
                end
                gamma(k,n) = nom1/den;
            end
        end
        %M-step
        for k = 1:Ns
            %pi(k)
            gamma_con_lap = con_tm(gamma(k,:),M,N,L,T_con_tm);
            p_hat(:,k) = gamma_con_lap/agent_data_size;
            %mu(k)
            nom2 = gamma(k,:).*x;
            for j = 1:2
                nom_con_lap = con_tm(nom2(j,:),M,N,L,T_con_tm);
                for i = 1:N
                    mu_hat(j,k,i) = nom_con_lap(i,1)/gamma_con_lap(i,1);
                end
            end
            %sigma(k)
            for n = 1:M
                agent = floor((n-1)/agent_data_size)+1;
                prod1 = (x(:,n)-mu_hat(:,k,agent))*(x(:,n)-mu_hat(:,k,agent))';
                prod2(:,n) = gamma(k,n)*[prod1(1,1);prod1(2,1);prod1(1,2);prod1(2,2)];
            end
            sigma11 = con_tm(prod2(1,:),M,N,L,T_con_tm);
            sigma21 = con_tm(prod2(2,:),M,N,L,T_con_tm);
            sigma12 = con_tm(prod2(3,:),M,N,L,T_con_tm);
            sigma22 = con_tm(prod2(4,:),M,N,L,T_con_tm);
            for i = 1:N
                sigma_hat(1,1,k,i) = sigma11(i,1)./gamma_con_lap(i,1);
                sigma_hat(2,1,k,i) = sigma21(i,1)./gamma_con_lap(i,1);
                sigma_hat(1,2,k,i) = sigma12(i,1)./gamma_con_lap(i,1);
                sigma_hat(2,2,k,i) = sigma22(i,1)./gamma_con_lap(i,1);
                if all(eig(sigma_hat(:,:,k,i)) > .1)
                else
                    sigma_hat(:,:,k,i) = sigma_hat(:,:,k,i) + (max(abs(eig(sigma_hat(:,:,k,i))))+.1)*eye(2,2);
                end
            end
        end
        error_tm(t+1) = log(norm(data_points-mu_hat(:,:,plot_agent))/norm(data_points));
    end
    if iter == plot_iter_1
        figure(1)
        for k = 1:Ns
            z = ellipsedata(sigma_hat(:,:,k,plot_agent), mu_hat(:,k,plot_agent), 100, 3, 1E-12);
            p3 = plot(z(:,1),z(:,2),'Color',color(k,:),'LineWidth',3);
            hold on
        end
    end
    if iter == plot_iter_2
        figure(2)
        for k = 1:Ns
            z = ellipsedata(sigma_hat(:,:,k,plot_agent), mu_hat(:,k,plot_agent), 100, 3, 1E-12);
            plot(z(:,1),z(:,2),'Color',color(k,:),'LineWidth',3)
            hold on
        end
    end
    LL_tm(iter) = 0;
    for n = 1:M
        LL_cat_tm = 0;
        for k = 1:Ns
            LL_cat_tm = LL_cat_tm+p_hat(plot_agent,k)*mvnpdf(x(:,n),mu_hat(:,k,plot_agent),sigma_hat(:,:,k,plot_agent));
        end
        LL_tm(iter) = LL_tm(iter)+log(LL_cat_tm);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TM consensus (bounded) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %running the EM algorithm
    %initialization
    for k = 1:Ns
        for i = 1:N
            p_hat(i,k) = 1/Ns;
            mu_hat(:,k,i) = [pos(k,1);pos(k,2)];
            sigma_hat(:,:,k,i) = [20 0;0 20];
        end
    end
    error_tm_b = zeros(T,1);
    error_tm_b(1,1) = log(norm(data_points-mu_hat(:,:,plot_agent))/norm(data_points));
    for t = 1:T
        %E-step
        for k = 1:Ns
            for n = 1:M
                agent = floor((n-1)/agent_data_size)+1;
                nom1 = p_hat(agent,k)*mvnpdf(x(:,n),mu_hat(:,k,agent),sigma_hat(:,:,k,agent));
                den = 0;
                for j = 1:Ns
                    den = den + p_hat(agent,j)*mvnpdf(x(:,n),mu_hat(:,j,agent),sigma_hat(:,:,j,agent));
                end
                gamma(k,n) = nom1/den;
            end
        end
        %M-step
        for k = 1:Ns
            %pi(k)
            gamma_con_lap = con_tm_b(gamma(k,:),M,N,L,T_con_tm);
            p_hat(:,k) = gamma_con_lap/agent_data_size;
            %mu(k)
            nom2 = gamma(k,:).*x;
            for j = 1:2
                nom_con_lap = con_tm_b(nom2(j,:),M,N,L,T_con_tm);
                for i = 1:N
                    mu_hat(j,k,i) = nom_con_lap(i,1)/gamma_con_lap(i,1);
                end
            end
            %sigma(k)
            for n = 1:M
                agent = floor((n-1)/agent_data_size)+1;
                prod1 = (x(:,n)-mu_hat(:,k,agent))*(x(:,n)-mu_hat(:,k,agent))';
                prod2(:,n) = gamma(k,n)*[prod1(1,1);prod1(2,1);prod1(1,2);prod1(2,2)];
            end
            sigma11 = con_tm_b(prod2(1,:),M,N,L,T_con_tm);
            sigma21 = con_tm_b(prod2(2,:),M,N,L,T_con_tm);
            sigma12 = con_tm_b(prod2(3,:),M,N,L,T_con_tm);
            sigma22 = con_tm_b(prod2(4,:),M,N,L,T_con_tm);
            for i = 1:N
                sigma_hat(1,1,k,i) = sigma11(i,1)./gamma_con_lap(i,1);
                sigma_hat(2,1,k,i) = sigma21(i,1)./gamma_con_lap(i,1);
                sigma_hat(1,2,k,i) = sigma12(i,1)./gamma_con_lap(i,1);
                sigma_hat(2,2,k,i) = sigma22(i,1)./gamma_con_lap(i,1);
                if all(eig(sigma_hat(:,:,k,i)) > .1)
                else
                    sigma_hat(:,:,k,i) = sigma_hat(:,:,k,i) + (max(abs(eig(sigma_hat(:,:,k,i))))+.1)*eye(2,2);
                end
            end
        end
        error_tm_b(t+1) = log(norm(data_points-mu_hat(:,:,plot_agent))/norm(data_points));
    end
    %{
    if iter == plot_iter
        figure(1)
        for k = 1:Ns
            z = ellipsedata(sigma_hat(:,:,k,plot_agent), mu_hat(:,k,plot_agent), 100, 3, 1E-12);
            plot(z(:,1),z(:,2),'Color',color(k,:),'LineWidth',3)
            hold on
        end
    end
    %}
    LL_tm_b(iter) = 0;
    for n = 1:M
        LL_cat_tm = 0;
        for k = 1:Ns
            LL_cat_tm = LL_cat_tm+p_hat(plot_agent,k)*mvnpdf(x(:,n),mu_hat(:,k,plot_agent),sigma_hat(:,:,k,plot_agent));
        end
        LL_tm_b(iter) = LL_tm_b(iter)+log(LL_cat_tm);
    end
end
%figure(2)
%plot(0:T,error_lap,'r')
%hold on
%plot(0:T,error_tm,'b')
figure(3)
l1 = yline(LL_cen,'--','Color','m','LineWidth',2);
hold on
l2 = plot(T_con,LL_tm,'.','Color','k','MarkerSize',30);
hold on
l3 = plot(T_con,LL_tm_b,'.','Color','b','MarkerSize',30);
hold on
l4 = plot(T_con,LL_lap,'.','Color','r','MarkerSize',30);
hold on
for iter = 1:len_T_con
    xline(T_con(iter),'-','Color',grayColor','LineWidth',.5)
end

figure(1)
xlabel('y','fontsize',20)
ylabel('x','fontsize',20)
set(gca,'FontSize',15)
axis([-100 100 -80 80])
xticks([-100 -80 -60 -40 -20 0 20 40 60 80 100])
yticks([-80 -60 -40 -20 0 20 40 60 80])
legend([p1(1,1),p2(1,1),p3(1,1)],{'True GMM', 'Distributed EM (Laplacian)', 'Distributed EM (TM)'},'FontSize',10)
box on

figure(2)
xlabel('y')
ylabel('x')
set(gca,'FontSize',15)
axis([-100 100 -80 80])
xticks([-100 -80 -60 -40 -20 0 20 40 60 80 100])
yticks([-80 -60 -40 -20 0 20 40 60 80])
box on

figure(3)
xlim([0 60])
ylim([LL_lap(1)-50 LL_cen+50])
xticks([0 8 15 30 50 60])
ax=gca;
ax.YAxis.Exponent = 3;
ytickformat('%,.1f')
set(gcf,'position',[100,100,750,250])
L = get(gca,'YLim');
set(gca,'YTick',linspace(L(1),L(2),3))
xlabel('$T_{consensus}$','interpreter','latex')
ylabel('Log-likelihood')
set(gca,'FontSize',15)
legend([l1(1,1),l2(1,1),l3(1,1),l4(1,1)],{'Central EM','Decentral EM (unbounded TM)','Decentral EM (bounded TM)','Decentral EM (Laplacian)'},'FontSize',15)
box on
