function plot_backup_soc_vs_distance(log_dir)
if nargin < 1 || strlength(string(log_dir)) == 0
    log_dir = "/home/tmo/ros2_ws/log/energy_backup_20260401_195404";
end

[distance_km, avg_soc, truck_soc, reached_50] = read_backup_soc(log_dir);

fig = figure('Color', 'w', 'Position', [120 120 1100 650], ...
    'Name', 'SOC vs Distance (Backup Log)', 'NumberTitle', 'off');
ax = axes(fig);
hold(ax, 'on');
grid(ax, 'on');

plot(ax, distance_km, avg_soc, '-', 'LineWidth', 2.6, 'Color', [0.00 0.00 0.00], ...
    'DisplayName', 'Fleet Avg SOC');
plot(ax, distance_km, truck_soc(:, 1), '-', 'LineWidth', 1.8, 'Color', [0.85 0.33 0.10], ...
    'DisplayName', 'Truck0 SOC');
plot(ax, distance_km, truck_soc(:, 2), '-', 'LineWidth', 1.8, 'Color', [0.47 0.67 0.19], ...
    'DisplayName', 'Truck1 SOC');
plot(ax, distance_km, truck_soc(:, 3), '-', 'LineWidth', 1.8, 'Color', [0.00 0.45 0.74], ...
    'DisplayName', 'Truck2 SOC');

xlabel(ax, 'Distance (km)');
ylabel(ax, 'SOC (%)');
if reached_50
    title(ax, 'SOC vs Distance Until Fleet Avg SOC Reaches 50%');
else
    title(ax, 'SOC vs Distance (0-120 km window, Fleet Avg SOC 50% not reached)');
end
xlim(ax, [0 120]);
ylim(ax, [40 100]);
legend(ax, 'Location', 'southwest');

hold(ax, 'off');
end


function [distance_km, avg_soc, truck_soc, reached_50] = read_backup_soc(log_dir)
truck_tables = cell(1, 3);
for truck_id = 0:2
    csv_path = fullfile(log_dir, sprintf('truck%d_energy.csv', truck_id));
    if ~isfile(csv_path)
        error('Missing CSV: %s', csv_path);
    end
    truck_tables{truck_id + 1} = readtable(csv_path);
end

min_rows = min(cellfun(@height, truck_tables));
if min_rows == 0
    error('No rows found in backup CSV files.');
end

truck_distance_m = zeros(min_rows, 3);
truck_soc = zeros(min_rows, 3);

for idx = 1:3
    T = truck_tables{idx}(1:min_rows, :);
    t = T.time_sec;
    v = max(0.0, T.velocity_ms);
    dt = [0; diff(t)];
    dt(dt < 0) = 0;
    truck_distance_m(:, idx) = cumsum(v .* dt);
    truck_soc(:, idx) = min(100.0, max(0.0, T.soc));
end

distance_m = mean(truck_distance_m, 2);
avg_soc = mean(truck_soc, 2);
distance_m = max(0.0, distance_m);
avg_soc = min(100.0, max(0.0, avg_soc));

cut_idx = find(avg_soc <= 50.0, 1, 'first');
distance_km = distance_m / 1000.0;
idx_120km = find(distance_km <= 120.0, 1, 'last');
if isempty(idx_120km)
    idx_120km = 1;
end

reached_50 = ~isempty(cut_idx);
if reached_50
    final_idx = min(cut_idx, idx_120km);
else
    final_idx = idx_120km;
end

distance_km = distance_km(1:final_idx);
truck_soc = truck_soc(1:final_idx, :);
avg_soc = avg_soc(1:final_idx);
end
