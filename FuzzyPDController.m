function FuzzyPDController()
clearvars; clc; close all;
m = 1.7;         % Quadrotor mass (kg)
g = 9.81;          % Gravity acceleration (m/s^2)

Ix = 4e-3;          % Moments of inertia (kg*m^2)
Iy = 4e-3;  
Iz = 8.4e-3;

k = 2.98e-5;       % Lift constant
b = 3.23e-7;       % Drag constant
l = 0.25;          % Arm length (m)

Jx = 5.7e-4;       % Drag coefficients
Jy = 5.7e-4;
Jz = 5.7e-4;
dt = 0.01;

% Initial state vector x0 (12 x 1)=  [x, y, z, vx, vy, vz, phi, theta, psi, phi_dot, theta_dot, psi_dot]

x0 = zeros(14, 1);
x0(3,1) = 1;

t0 = 0;         
Tf = 60;        

fis1 = readfis('delKp.fis');
fis2 = readfis('delKd.fis');

Kp_p = [0.1; 0.1; 0.1];
Kd_p = [1/3; 1/3; 1/3];

Kp_a = [0.1; 0.1; 0.1];
Kd_a = [1/3; 1/3; 1/3];

%- different eqn
opts = odeset('RelTol', 1e-3, 'AbsTol', 1e-5, 'MaxStep', 0.02);

% Function handle directly calls nested function 'f'
[t, X] = ode45(@f, [t0 Tf], x0, opts);

% Extract states
x = X(:,1);
y = X(:,2);
z = X(:,3);

phi = X(:,7);
theta = X(:,8);
psi = X(:,9);

phi_d_prev = X(:,13);
theta_d_prev = X(:,14);

xd = cos(pi*t/20);
yd = sin(pi*t/20);
zd = 2+ 0*t;
psi_d = zeros(size(t));

%- simulation result

figure('Name','3D Trajectory');
plot3(x, y, z, 'b', 'LineWidth', 2); hold on;
plot3(xd, yd, zd, 'r--', 'LineWidth', 2);
grid on;
xlabel('x (m)'); ylabel('y (m)'); zlabel('z (m)');
legend('Actual', 'Desired');
title('3D Trajectory Tracking');
axis equal; view(45, 30);

figure('Name','Position tracking');
subplot(3,1,1);
plot(t, x, 'b', t, xd, 'r--', 'LineWidth', 1.5); grid on;
ylabel('x (m)'); legend('Actual','Desired');
title('Position tracking');
subplot(3,1,2);
plot(t, y, 'b', t, yd, 'r--', 'LineWidth', 1.5); grid on;
ylabel('y (m)'); legend('Actual','Desired')
subplot(3,1,3);
plot(t, z, 'b', t, zd, 'r--', 'LineWidth', 1.5); grid on;
ylabel('z (m)'); xlabel('Time (s)');  legend('Actual','Desired');


figure('Name','Attitude tracking','Color','w');
subplot(3,1,1);
plot(t, phi, 'b', t, phi_d_prev, 'r--', 'LineWidth', 1.5); grid on;
ylabel('\phi (rad)'); legend('\phi','\phi_d');
title('Attitude tracking');
subplot(3,1,2);
plot(t, theta, 'b', t, theta_d_prev, 'r--', 'LineWidth', 1.5); grid on;
ylabel('\theta (rad)'); legend('\theta','\theta_d');
subplot(3,1,3);
plot(t, psi, 'b', t, psi_d, 'r--', 'LineWidth', 1.5); grid on;
ylabel('\psi (rad)'); xlabel('Time (s)'); legend('\psi','\psi_d');



%- controller
    function deq = f(t, X)

    % Reference Trajectory
    xd = cos(pi*t/20);
    yd = sin(pi*t/20);
    zd = 2+ 0*t;
    psi_d = 0;
    psi_d_dot = 0;

    xd_dot = -(pi/20)*sin(pi*t/20);
    yd_dot = (pi/20)*cos(pi*t/20);
    zd_dot = 0*t;

    %- redefine variable
    x = X(1); y = X(2); z = X(3);
    x_dot = X(4); y_dot = X(5); z_dot = X(6);

    phi = X(7); theta = X(8); psi = X(9);
    phi_dot = X(10); theta_dot = X(11); psi_dot = X(12);

    phi_d_prev = X(13);
    theta_d_prev = X(14);

    %- fuzzy pid position controller
    ex = xd - x;
    ex_dot = xd_dot - x_dot;
    ey = yd - y;
    ey_dot = yd_dot - y_dot;
    ez = zd - z;
    ez_dot = zd_dot - z_dot;

    ex = sat(ex,-5,5);
    ey = sat(ey,-5,5);
    ez = sat(ez,-5,5);

    ex_dot = sat(ex_dot,-5,5);
    ey_dot = sat(ey_dot,-5,5);
    ez_dot = sat(ez_dot,-5,5);

    delKpx = evalfis(fis1,[ex ex_dot]);
    delKdx = evalfis(fis2,[ex ex_dot]);

    delKpy = evalfis(fis1,[ey ey_dot]);
    delKdy = evalfis(fis2,[ey ey_dot]);

    delKpz = evalfis(fis1,[ez ez_dot]);
    delKdz = evalfis(fis2,[ez ez_dot]);

    %delKp = sat(delKp, 0, 10);
    %delKd = sat(delKd, 0, 10);

     ux = (Kp_p(1,1) + delKpx).*ex + (Kd_p(1,1) + delKdx).*ex_dot;
     uy = (Kp_p(2,1) + delKpy).*ey + (Kd_p(2,1) + delKdy).*ey_dot;
     uz = (Kp_p(3,1) + delKpz).*ez + (Kd_p(3,1) + delKdz).*ez_dot;

     T = m*sqrt(ux^2 + uy^2 + (uz + g)^2); 
    
     phi_d = asin((ux*sin(psi_d) - uy*cos(psi_d)) / sqrt(ux^2 + uy^2 + (uz + g)^2));

     theta_d = atan((ux*cos(psi_d) + uy*sin(psi_d)) / (uz + g));

     phi_d_dot   = (phi_d  - phi_d_prev)/dt;
     theta_d_dot = (theta_d - theta_d_prev)/dt;

     %--pid  attitude controller
     e_phi = phi_d - phi;
    e_theta = theta_d - theta;
    e_psi = psi_d - psi;

    e_phi_dot = phi_d_dot - phi_dot;
    e_theta_dot = theta_d_dot - theta_dot;
    e_psi_dot = psi_d_dot - psi_dot;

    tau_phi = Kp_a(1,1).*e_phi + Kd_a(1,1).*e_phi_dot;
    tau_theta = Kp_a(2,1).*e_theta + Kd_a(2,1).*e_theta_dot;
    tau_psi = Kp_a(3,1).*e_psi + Kd_a(3,1).*e_psi_dot;

    %- disturbance
    dis_x = 0.05*sin(t)*cos(0.5*t)^2;
    dis_y = 0.05*sin(t)*cos(0.5*t)^2;
    dis_z = 0.05*sin(t)*cos(0.5*t)^2;

    dis_phi = 0.02*sin(0.7*t);
    dis_theta = 0.02*sin(0.7*t);
    dis_psi = 0.02*sin(0.7*t);

    %% Derivative equations of attitudes
    phi_dot2 = theta_dot*psi_dot*(Iy - Iz)/Ix + tau_phi/Ix+ dis_x;
    theta_dot2 = phi_dot*psi_dot*(Iz - Ix)/Iy + tau_theta/Iy+ dis_y; 
    psi_dot2 = phi_dot*theta_dot*(Ix - Iy)/Iz + tau_psi/Iz+ dis_z;

    %% Derivative equations of positions
    x_dot2 = (T/m)*( cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi)) - (Jx/m)*x_dot+ dis_phi;
    y_dot2 = (T/m)*( cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi)) - (Jy/m)*y_dot+ dis_theta;
    z_dot2 = (T/m)*cos(phi)*cos(theta) - g - (Jz/m)*z_dot+ dis_psi;

    % Derivatives Vector Output
    deq = zeros(12,1);
    deq(1) = x_dot;
    deq(2) = y_dot;
    deq(3) = z_dot;

    deq(4) = x_dot2;
    deq(5) = y_dot2;
    deq(6) = z_dot2;

    deq(7) = phi_dot;
    deq(8) = theta_dot;
    deq(9) = psi_dot;

    deq(10) = phi_dot2;
    deq(11) = theta_dot2;
    deq(12) = psi_dot2;

    deq(13) = phi_d_dot;
    deq(14) = theta_d_dot;
end
end

function u = sat(x, a, b)
    u = min(max(x,a),b);
end