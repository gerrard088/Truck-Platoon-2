function plot_per_lap_energy(log_dir, refresh_sec)
if nargin < 1 || strlength(string(log_dir)) == 0
    log_dir = "/home/tmo/ros2_ws/log/energy";
end

if nargin < 2 || isempty(refresh_sec)
    refresh_sec = 1.0;
end

fig = figure('Color', 'w', 'Position', [100 100 1200 700], ...
    'Name', 'Live Energy Monitor', 'NumberTitle', 'off');
tiledlayout(fig, 4, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

ax1 = nexttile;
grid(ax1, 'on');
hold(ax1, 'on');
xlabel(ax1, 'Distance (km)');
ylabel(ax1, 'Fleet Avg SOC (%)');
title(ax1, 'Fleet Average SOC vs Distance');
line_avg_soc = plot(ax1, NaN, NaN, '-', 'LineWidth', 2.0, ...
    'Color', [0.00 0.45 0.74]);
hold(ax1, 'off');
ylim(ax1, [0 100]);

ax2 = nexttile;
grid(ax2, 'on');
xlabel(ax2, 'Truck');
ylabel(ax2, 'SOC (%)');
title(ax2, 'Live Truck SOC');
bar_soc = bar(ax2, [0 0 0], 0.55, 'FaceColor', 'flat');
bar_soc.CData = [
    0.85 0.33 0.10
    0.47 0.67 0.19
    0.00 0.45 0.74
];
ax2.XTick = 1:3;
ax2.XTickLabel = {'truck0', 'truck1', 'truck2'};
ylim(ax2, [0 100]);

ax3 = nexttile;
grid(ax3, 'on');
hold(ax3, 'on');
xlabel(ax3, 'Time (sec)');
ylabel(ax3, 'Speed (km/h)');
title(ax3, 'Fleet Average Speed vs Time');
line_avg_speed = plot(ax3, NaN, NaN, '-', 'LineWidth', 2.0, ...
    'Color', [0.64 0.08 0.18]);
hold(ax3, 'off');
ylim(ax3, [0 90]);

ax4 = nexttile;
grid(ax4, 'on');
hold(ax4, 'on');
xlabel(ax4, 'Time (sec)');
ylabel(ax4, 'Throttle');
title(ax4, 'Fleet Average Throttle vs Time');
line_avg_throttle = plot(ax4, NaN, NaN, '-', 'LineWidth', 2.0, ...
    'Color', [0.49 0.18 0.56]);
hold(ax4, 'off');
ylim(ax4, [-1.1 1.1]);

status_text = annotation(fig, 'textbox', [0.60 0.94 0.38 0.04], ...
    'String', 'Waiting for data...', 'EdgeColor', 'none', ...
    'HorizontalAlignment', 'right', 'FontSize', 11);

while isvalid(fig)
    try
        [time_sec, distance_km, avg_soc, latest_soc, avg_speed_kmh, avg_throttle, has_throttle] = read_live_energy(log_dir);
        if isempty(distance_km)
            status_text.String = "CSV exists, but no valid rows yet.";
        else
            update_plots(ax1, line_avg_soc, ax2, bar_soc, ax3, line_avg_speed, ...
                ax4, line_avg_throttle, time_sec, distance_km, avg_soc, latest_soc, ...
                avg_speed_kmh, avg_throttle, has_throttle, status_text);
        end
    catch err
        status_text.String = "Read failed: " + string(err.message);
    end

    drawnow;
    pause(refresh_sec);
end
end


function [time_sec, distance_km, avg_soc, latest_soc, avg_speed_kmh, avg_throttle, has_throttle] = read_live_energy(log_dir)
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
    time_sec = [];
    distance_km = [];
    avg_soc = [];
    latest_soc = [0 0 0];
    avg_speed_kmh = [];
    avg_throttle = [];
    has_throttle = false;
    return;
end

truck_distance_km = zeros(min_rows, 3);
truck_soc = zeros(min_rows, 3);
truck_speed_kmh = zeros(min_rows, 3);
truck_throttle = NaN(min_rows, 3);
has_throttle = true;

for idx = 1:3
    T = truck_tables{idx}(1:min_rows, :);
    t = T.time_sec;
    v = T.velocity_ms;
    dt = [0; diff(t)];
    dt(dt < 0) = 0;
    truck_distance_km(:, idx) = cumsum(v .* dt) / 1000.0;
    truck_soc(:, idx) = T.soc;
    truck_speed_kmh(:, idx) = v * 3.6;
    if ismember("throttle", string(T.Properties.VariableNames))
        truck_throttle(:, idx) = T.throttle;
    else
        has_throttle = false;
    end
end

raw_time_sec = truck_tables{1}.time_sec(1:min_rows);
time_sec = raw_time_sec - raw_time_sec(1);
distance_km = mean(truck_distance_km, 2);
avg_soc = mean(truck_soc, 2);
latest_soc = truck_soc(end, :);
avg_speed_kmh = mean(truck_speed_kmh, 2);
avg_throttle = mean(truck_throttle, 2, 'omitnan');
end


function update_plots(ax1, line_avg_soc, ax2, bar_soc, ax3, line_avg_speed, ...
    ax4, line_avg_throttle, time_sec, distance_km, avg_soc, latest_soc, ...
    avg_speed_kmh, avg_throttle, has_throttle, status_text)
set(line_avg_soc, 'XData', distance_km, 'YData', avg_soc);
xmin = min(distance_km);
xmax = max(distance_km);
if xmin == xmax
    xmax = xmin + 0.01;
end
xlim(ax1, [xmin xmax]);
ylim(ax1, [0 100]);

set(bar_soc, 'YData', latest_soc);
ylim(ax2, [0 100]);

set(line_avg_speed, 'XData', time_sec, 'YData', avg_speed_kmh);
tmin = min(time_sec);
tmax = max(time_sec);
if tmin == tmax
    tmax = tmin + 0.01;
end
xlim(ax3, [tmin tmax]);
ylim(ax3, [0 90]);

if has_throttle
    set(line_avg_throttle, 'XData', time_sec, 'YData', avg_throttle);
    xlim(ax4, [tmin tmax]);
    ylim(ax4, [-1.1 1.1]);
    throttle_suffix = sprintf(' | Fleet Avg Throttle %.3f', avg_throttle(end));
else
    set(line_avg_throttle, 'XData', NaN, 'YData', NaN);
    xlim(ax4, [tmin tmax]);
    ylim(ax4, [-1.1 1.1]);
    throttle_suffix = ' | Fleet Avg Throttle unavailable (missing CSV column)';
end

status_text.String = sprintf(['Distance %.3f km | Fleet Avg SOC %.3f %% | ' ...
    'Fleet Avg Speed %.3f km/h | T0 %.3f %% | T1 %.3f %% | T2 %.3f %%%s'], ...
    distance_km(end), avg_soc(end), avg_speed_kmh(end), ...
    latest_soc(1), latest_soc(2), latest_soc(3), throttle_suffix);
end
