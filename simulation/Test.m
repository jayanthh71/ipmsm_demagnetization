clc;

assignin('base', 'psim', input_psim);

%% ---- Build input time-series ----
t_sim = (0:100e-6:3)';
speed_signal = input_speed * ones(size(t_sim));
torque_signal = input_torque * ones(size(t_sim));

speed_ts = timeseries(speed_signal, t_sim);
torque_ts = timeseries(torque_signal, t_sim);

Test_1 = Simulink.SimulationData.Dataset;
Test_1 = addElement(Test_1, speed_ts, 'Speed');
Test_1 = addElement(Test_1, torque_ts, 'Torque');
save('inputs.mat', 'Test_1');

%% ---- Run simulation ----
simOut = sim('IPMSMOuterLoopControl', 'StopTime', num2str(3));
logs = simOut.simlog_IPMSMOuterLoopControl.IPMSM;

ia = logs.i_a.series.values;
iq = logs.i_q.series.values;
id = logs.i_d.series.values;
torque = logs.electrical_torque.series.values;
omega = logs.angular_velocity.series.values;
loss = logs.power_dissipated.series.values;

%% ---- Remove first 30% transient ----
N = length(ia);
idx = round(0.30 * N):N;
ia = ia(idx);
iq = iq(idx);
id = id(idx);
torque = torque(idx);
omega = omega(idx);
loss = loss(idx);

%% ---- Feature extraction ----
Ia_rms = rms(ia);
Iq_rms = rms(iq);
Id_rms = rms(id);
Peak_I = max(abs(ia));
MechPower = mean(torque .* omega);
CopperLoss = mean(loss);
T_Iq_ratio = mean(torque) / Iq_rms;
Id_Iq_ratio = rms(id ./ iq);

%% ---- Construct output struct ----
data = struct();
data.Speed = input_speed;
data.Torque = input_torque;
data.Ia_RMS = Ia_rms;
data.Iq_RMS = Iq_rms;
data.Id_RMS = Id_rms;
data.Peak_Current = Peak_I;
data.MechPower = MechPower;
data.CopperLoss = CopperLoss;
data.Torque_Iq_Ratio = T_Iq_ratio;
data.Id_Iq_Ratio = Id_Iq_ratio;
data.True_Flux_Linkage = input_psim;

%% ---- Save JSON ----
jsonStr = jsonencode(data);
fid = fopen('features.json', 'w');
fprintf(fid, jsonStr);
fclose(fid);
