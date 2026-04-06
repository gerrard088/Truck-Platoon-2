function plot_backup_soc_bar_at_truck0_40(log_dir)
if nargin < 1 || strlength(string(log_dir)) == 0
    log_dir = "/home/tmo/ros2_ws/log/energy_backup_20260401_195404";
end

[soc_values, hit_index] = read_soc_at_truck0_40(log_dir);

fig = figure('Color', 'w', 'Position', [160 160 820 560], ...
    'Name', 'SOC Bar at Truck0 40 Percent', 'NumberTitle', 'off');
ax = axes(fig);

bar_plot = bar(ax, soc_values, 0.55, 'FaceColor', 'flat');
bar_plot.CData = [
    0.85 0.33 0.10
    0.47 0.67 0.19
    0.00 0.45 0.74
];

grid(ax, 'on');
xlabel(ax, 'Truck');
ylabel(ax, 'SOC (%)');
title(ax, sprintf('Truck SOC When Truck0 First Reaches 40%% (row %d)', hit_index));
ax.XTick = 1:3;
ax.XTickLabel = {'truck0', 'truck1', 'truck2'};
ylim(ax, [20 100]);
end


function [soc_values, hit_index] = read_soc_at_truck0_40(log_dir)
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

truck_soc = zeros(min_rows, 3);
for idx = 1:3
    T = truck_tables{idx}(1:min_rows, :);
    truck_soc(:, idx) = min(100.0, max(0.0, T.soc));
end

hit_index = find(truck_soc(:, 1) <= 40.0, 1, 'first');
if isempty(hit_index)
    error('Truck0 SOC never reaches 40%% in %s', log_dir);
end

soc_values = truck_soc(hit_index, :);
end
