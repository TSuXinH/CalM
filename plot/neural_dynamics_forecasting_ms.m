clear; close all; clc;

f_calm_in  = "session_corr_calm_heldin.mat";
f_calm_out = "session_corr_calm_heldout.mat";
f_poco_in  = "session_corr_poco_heldin.mat";
f_poco_out = "session_corr_poco_heldout.mat";

assert(isfile(f_calm_in)  && isfile(f_calm_out) && isfile(f_poco_in) && isfile(f_poco_out), ...
    "Cannot find one or more .mat files in current folder.");

[calm_in_sid,  calm_in_corr]  = load_calm_corr(f_calm_in);
[poco_in_sid,  poco_in_corr]  = load_poco_corr(f_poco_in);
[common_in, ia, ib] = intersect(calm_in_sid, poco_in_sid, 'stable');
calm_in_corr = calm_in_corr(ia);
poco_in_corr = poco_in_corr(ib);

[calm_out_sid, calm_out_corr] = load_calm_corr(f_calm_out);
[poco_out_sid, poco_out_corr] = load_poco_corr(f_poco_out);
[common_out, ja, jb] = intersect(calm_out_sid, poco_out_sid, 'stable');
calm_out_corr = calm_out_corr(ja);
poco_out_corr = poco_out_corr(jb);

fprintf("[INFO] Held-in sessions used:  %d\n", numel(common_in));
fprintf("[INFO] Held-out sessions used: %d\n", numel(common_out));

dx   = 0.26;   
gap  = 0.55;   
x1 = 1.00; x2 = x1 + dx;
x3 = x2 + gap; x4 = x3 + dx;
xpos = [x1 x2 x3 x4];

labels = ["POCO","CALM","POCO","CALM"];

colors = [ ...
    0.90 0.25 0.25;  % POCO
    0.45 0.65 0.95;  % CALM
    0.90 0.25 0.25;  % POCO
    0.45 0.65 0.95;  % CALM
];

box_data = {poco_in_corr(:), calm_in_corr(:), poco_out_corr(:), calm_out_corr(:)};
means = cellfun(@(v) mean(v,'omitnan'), box_data).';


fig = figure('Color','w','Position',[120,120,1100,520]);
set(fig,'Renderer','painters');        
set(fig,'InvertHardcopy','off');       

ax = axes('Position',[0.08 0.22 0.90 0.72], 'Color','w');
hold(ax,'on');

b = bar(ax, xpos, means, 'FaceColor','flat', 'EdgeColor','none', 'BarWidth',0.88);
b.CData = colors;

allvals = vertcat(box_data{:});
ymax = max(allvals(~isnan(allvals)));
ylim(ax, [0, max(0.55, ymax*1.10)]);

boxW = 0.18;  
for k = 1:4
    draw_box_outline(ax, xpos(k), box_data{k}, boxW);
end

yl = ylim(ax);
y_text = yl(1) + 0.035*(yl(2)-yl(1));
for k = 1:4
    if ~isnan(means(k))
        text(ax, xpos(k), y_text, sprintf('%.2f%%', means(k)*100), ...
            'HorizontalAlignment','center', ...
            'Color','w', 'FontWeight','bold', 'FontSize',10);
    end
end

ax.FontName = 'Times New Roman';
ax.FontSize = 12;
ax.XTick = xpos;
ax.XTickLabel = labels;
ax.XTickLabelRotation = 0;

ylabel(ax, "Correlation", 'FontSize',13, 'FontWeight','bold');
title(ax, "CALM vs POCO: Held-in / Held-out (bar=mean, box=distribution)", 'FontSize',13);

grid(ax,'off'); ax.XGrid='off'; ax.YGrid='off';
box(ax,'on'); ax.TickDir='out';

xlim(ax, [x1-0.35, x4+0.35]);

add_group_label(fig, ax, mean(xpos(1:2)), "Held-in");
add_group_label(fig, ax, mean(xpos(3:4)), "Held-out");

hold(ax,'off');

print(fig, 'calm_vs_poco_corr_vector.pdf', '-dpdf', '-painters');

function [sid, corr_vals] = load_calm_corr(matfile)
    S = load(matfile);
    if isfield(S,'base_session_id')
        sid = string(S.base_session_id(:));
    elseif isfield(S,'npz_name')
        sid = string(S.npz_name(:));
    else
        error("CALM mat missing base_session_id/npz_name: %s", matfile);
    end
    sid = erase(sid, ".npz");
    corr_vals = double(S.corr(:));
    n = min(numel(sid), numel(corr_vals));
    sid = sid(1:n); corr_vals = corr_vals(1:n);
    ok = ~isnan(corr_vals) & (strlength(sid) > 0);
    sid = sid(ok); corr_vals = corr_vals(ok);
end

function [sid, corr_vals] = load_poco_corr(matfile)
    S = load(matfile);
    if ~isfield(S,'session_id'); error("POCO mat missing session_id: %s", matfile); end
    sid = string(S.session_id(:));
    sid = erase(sid, ".npz");
    if ~isfield(S,'mean_corr'); error("POCO mat missing mean_corr: %s", matfile); end
    corr_vals = double(S.mean_corr(:));
    n = min(numel(sid), numel(corr_vals));
    sid = sid(1:n); corr_vals = corr_vals(1:n);
    ok = ~isnan(corr_vals) & (strlength(sid) > 0);
    sid = sid(ok); corr_vals = corr_vals(ok);
end

function draw_box_outline(ax, x, y, w)
    y = y(~isnan(y));
    if isempty(y); return; end

    q = quantile(y, [0.25 0.50 0.75]);
    q1 = q(1); med = q(2); q3 = q(3);
    iqr = q3 - q1;

    lowFence = q1 - 1.5*iqr;
    highFence = q3 + 1.5*iqr;
    lw = min(y(y >= lowFence));
    uw = max(y(y <= highFence));

    xL = x - w/2; xR = x + w/2;
    line(ax, [xL xR], [q1 q1], 'Color','k', 'LineWidth',1.1);
    line(ax, [xL xR], [q3 q3], 'Color','k', 'LineWidth',1.1);
    line(ax, [xL xL], [q1 q3], 'Color','k', 'LineWidth',1.1);
    line(ax, [xR xR], [q1 q3], 'Color','k', 'LineWidth',1.1);

    line(ax, [xL xR], [med med], 'Color','k', 'LineWidth',1.1);

    line(ax, [x x], [q3 uw], 'Color','k', 'LineWidth',1.1);
    line(ax, [x x], [q1 lw], 'Color','k', 'LineWidth',1.1);

    capW = w*0.55;
    line(ax, [x-capW/2 x+capW/2], [uw uw], 'Color','k', 'LineWidth',1.1);
    line(ax, [x-capW/2 x+capW/2], [lw lw], 'Color','k', 'LineWidth',1.1);
end

function add_group_label(fig, ax, x_center_data, txt)
    axPos = ax.Position;
    xl = xlim(ax);
    x_norm = axPos(1) + axPos(3) * (x_center_data - xl(1)) / (xl(2)-xl(1));
    y_norm = axPos(2) - 0.08;

    annotation(fig, 'textbox', [x_norm-0.06, y_norm, 0.12, 0.05], ...
        'String', txt, 'EdgeColor','none', ...
        'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
        'FontName','Times New Roman', 'FontSize',12, 'FontWeight','bold', ...
        'Color','k');
end
