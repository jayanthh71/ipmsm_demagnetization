%% Parameters for IPMSM Outer Loop Controller Evaluation Example

% This example shows how to control the rotor angular velocity in an
% interior permanent magnet synchronous machine (IPMSM) based
% electrical-traction drive. For evaluation of the outer loop velocity
% controller, the inner loop current controller is replaced by a
% three-phase controlled current source. An ideal torque source provides
% the load. The Scopes subsystem contains scopes that allow you to see
% the simulation results. The Control subsystem includes a PI-based outer
% loop controller. During the three-seconds simulation, the angular
% velocity demand is -1000, 2000, 3000, 4000, and then 5000 rpm.

% Copyright 2018-2023 The MathWorks, Inc.

%% Machine Parameters
Pmax = 35000; % Maximum power                      [W]
Tmax = 205; % Maximum torque                     [N*m]
Ld = 0.00024368; % Stator d-axis inductance           [H]
Lq = 0.00029758; % Stator q-axis inductance           [H]
L0 = 0.00012184; % Stator zero-sequence inductance    [H]
Rs = 0.010087; % Stator resistance per phase        [Ohm]
psim = 0.04366; % Permanent magnet flux linkage      [Wb]
p = 8; % Number of pole pairs
Jm = 0.1234; % Rotor inertia                      [Kg*m^2]
Vnom = 325; % Nominal DC voltage                 [V]

%% Control Parameters
Ts = 1e-4; % Fundamental sample time            [s]
Tso = 1e-3; % Sample time for outer control loop [s]

Kp_omega = 3.5; % Proportional gain velocity controller
Ki_omega = 200; % Integrator gain velocity controller

%% Zero-Cancellation Transfer Functions
numd_omega = Tso / (Kp_omega / Ki_omega);
dend_omega = [1 (Tso - (Kp_omega / Ki_omega)) / (Kp_omega / Ki_omega)];

%% Current References
load IPMSM35kWCurrentReferences;
