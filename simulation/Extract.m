clc;

%% ---- Simulation parameters ----
psim_nom = 0.04366; % Wb

speed_vec = linspace(500, 5000, 7); % RPM
torque_vec = linspace(-20, -180, 7); % Nm
psim_vec = linspace(0.5 * psim_nom, psim_nom, 10);

runs_per_point = 2;

%% ---- Pre-allocate ----
total_samples = length(speed_vec) * length(torque_vec) * length(psim_vec) * runs_per_point;
data = zeros(total_samples, 12);
class_label = cell(total_samples, 1);
row = 1;

%% ---- Main loop ----
for p = 1:length(psim_vec)

    for s = 1:length(speed_vec)

        for t = 1:length(torque_vec)

            for r = 1:runs_per_point

                %% ---- Apply noise to all three inputs ----
                speed_offset = speed_vec(s) * 0.06 * (2 * rand - 1);
                noisy_speed = speed_vec(s) + speed_offset;
                torque_offset = torque_vec(t) * 0.06 * (2 * rand - 1);
                noisy_torque = torque_vec(t) + torque_offset;
                psim_offset = psim_vec(p) * 0.03 * (2 * rand - 1);
                noisy_psim = psim_vec(p) + psim_offset;
                noisy_psim = max(0.5 * psim_nom, min(psim_nom * 1.03, noisy_psim));

                %% ---- Compute demagnetisation percentage and class ----
                demag_pct = (1 - noisy_psim / psim_nom) * 100;

                if noisy_psim >= 0.90 * psim_nom
                    label = 'Healthy';
                elseif noisy_psim >= 0.80 * psim_nom
                    label = 'Mild';
                elseif noisy_psim >= 0.65 * psim_nom
                    label = 'Moderate';
                else
                    label = 'Severe';
                end

                %% ---- Assign to Simulink workspace ----
                assignin('base', 'psim', noisy_psim);

                %% ---- Build input timeseries ----
                t_sim = (0:100e-6:3)';
                speed_signal = noisy_speed * ones(size(t_sim));
                torque_signal = noisy_torque * ones(size(t_sim));

                speed_ts = timeseries(speed_signal, t_sim);
                torque_ts = timeseries(torque_signal, t_sim);

                Test_1 = Simulink.SimulationData.Dataset;
                Test_1 = addElement(Test_1, speed_ts, 'Speed');
                Test_1 = addElement(Test_1, torque_ts, 'Torque');
                save('inputs.mat', 'Test_1');

                %% ---- Run simulation ----
                simOut = sim('IPMSMOuterLoopControl', 'StopTime', '3');
                logs = simOut.simlog_IPMSMOuterLoopControl.IPMSM;

                ia = logs.i_a.series.values;
                iq = logs.i_q.series.values;
                id = logs.i_d.series.values;
                torque = logs.electrical_torque.series.values;
                omega = logs.angular_velocity.series.values;
                loss = logs.power_dissipated.series.values;

                %% ---- Remove first 30% transient ----
                N = length(ia);
                idx = round(0.3 * N):N;
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

                %% ---- Store row ----
                data(row, :) = [noisy_speed, noisy_torque, Ia_rms, Iq_rms, Id_rms, Peak_I, MechPower, CopperLoss, T_Iq_ratio, Id_Iq_ratio, noisy_psim, demag_pct];
                class_label{row} = label;
                fprintf('[%4d/%4d] FL=%.5f Wb (%5.1f%% demag, %-8s) | spd=%.0f rpm | trq=%.1f Nm | Ia=%.2f A\n', row, total_samples, noisy_psim, demag_pct, label, noisy_speed, noisy_torque, Ia_rms);

                row = row + 1;
            end

        end

    end

end

%% ---- Export CSV ----
header = {'Speed', 'Torque', 'Ia_RMS', 'Iq_RMS', 'Id_RMS', 'Peak_Current', 'MechPower', 'CopperLoss', 'Torque_Iq_Ratio', 'Id_Iq_Ratio', 'Flux_Linkage', 'Demag_Pct', 'Demag_Class'};

T_num = array2table(data, 'VariableNames', header(1:12));
T_class = table(class_label, 'VariableNames', header(13));
T = [T_num, T_class];

writetable(T, 'IPMSM_Dataset.csv');

fprintf('Dataset saved : IPMSM_Dataset.csv\n');
fprintf('Total rows    : %d\n', total_samples);
