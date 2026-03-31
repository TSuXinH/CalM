clear; close all; clc;


f_calm_in  = "metrics_heldin_calm.mat";
f_calm_out = "metrics_heldout_calm.mat";
f_poyo_in  = "metrics_heldin_poyo.mat";
f_poyo_out = "metrics_heldout_poyo.mat";

assert(isfile(f_calm_in)  && isfile(f_calm_out) && isfile(f_poyo_in) && isfile(f_poyo_out), ...
    "Cannot find one or more .mat files. Put them in current folder or edit paths.");

[calm_in_sid,  calm_in]  = load_calm_metrics(f_calm_in,  "heldin");
[calm_out_sid, calm_out] = load_calm_metrics(f_calm_out, "heldout");

[poyo_in_sid,  poyo_in]  = load_poyo_metrics(f_poyo_in);
[poyo_out_sid, poyo_out] = load_poyo_metrics(f_poyo_out);

[calm_in, poyo_in, ~, how_in]     = align_by_sid_or_order(calm_in_sid,  calm_in,  poyo_in_sid,  poyo_in,  "held-in");
[calm_out, poyo_out, ~, how_out] = align_by_sid_or_order(calm_out_sid, calm_out, poyo_out_sid, poyo_out, "held-out");

fprintf("[INFO] Held-in align:  %s, n=%d\n", how_in,  size(calm_in.corr,1));
fprintf("[INFO] Held-out align: %s, n=%d\n", how_out, size(calm_out.corr,1));

dx   = 0.26;
gap  = 0.55;
x1 = 1.00; x2 = x1 + dx;
x3 = x2 + gap; x4 = x3 + dx;

xpos = [x1 x2 x3 x4];

labels = ["POYO","CALM","POYO","CALM"];

colors = [ ...
    0.90 0.25 0.25;  % POYO
    0.45 0.65 0.95;  % CALM
    0.90 0.25 0.25;  % POYO
    0.45 0.65 0.95;  % CALM
];

dim_names     = ["VX","VY","VZ"];
metric_names  = ["corr","r2","mse_trial_c"];
metric_ylabel = ["Correlation","R^2","MSE (trial\_c)"];

fig = figure('Color','w','Position',[80, 80, 1400, 980]);
set(fig,'Renderer','painters');
set(fig,'InvertHardcopy','off');
set(fig,'PaperPositionMode','auto');

t = tiledlayout(fig, 3, 3, "TileSpacing","compact", "Padding","compact");
sgtitle(t, "CALM vs POYO: Held-in / Held-out (bar=mean, box=distribution)", ...
    "FontName","Times New Roman", "FontSize",14, "FontWeight","bold");

boxW = 0.18;

for r = 1:3
    for c = 1:3
        ax = nexttile(t, (r-1)*3 + c);
        hold(ax,'on');

        [v_in_calm,  v_out_calm] = pick_metric_dim(calm_in, calm_out, metric_names(r), c);
        [v_in_poyo,  v_out_poyo] = pick_metric_dim(poyo_in, poyo_out, metric_names(r), c);

        box_data = {v_in_poyo(:), v_in_calm(:), v_out_poyo(:), v_out_calm(:)};
        means = cellfun(@(v) mean(v,'omitnan'), box_data).';

        b = bar(ax, xpos, means, 'FaceColor','flat', 'EdgeColor','none', 'BarWidth',0.88);
        b.CData = colors;

        allvals = vertcat(box_data{:});
        allvals = allvals(~isnan(allvals));
        if isempty(allvals)
            ymax = 1;
        else
            ymax = max(allvals);
        end
        ytop = max(0.55, ymax*1.10);
        
        ybot = 0;
        
        ylim(ax, [ybot, ytop]);
        for k = 1:4
            draw_box_outline(ax, xpos(k), box_data{k}, boxW);
        end

        yl = ylim(ax);
        y_anchor = max(0, yl(1));
        y_text   = y_anchor + 0.035*(yl(2)-y_anchor);
        for k = 1:4
            if ~isnan(means(k))
                if metric_names(r) == "mse_trial_c"
                    txt = sprintf('%.3f', means(k));
                else
                    txt = sprintf('%.2f%%', means(k)*100);
                end
                text(ax, xpos(k), y_text, txt, ...
                    'HorizontalAlignment','center', ...
                    'Color','w', 'FontWeight','bold', 'FontSize',9, ...
                    'FontName','Times New Roman');
            end
        end

        ax.FontName = 'Times New Roman';
        ax.FontSize = 11;
        ax.XTick = xpos;                 
        ax.XTickLabel = labels;
        ax.XTickLabelRotation = 0;
        grid(ax,'off'); ax.XGrid='off'; ax.YGrid='off';
        box(ax,'on'); ax.TickDir='out';
        xlim(ax, [x1-0.35, x4+0.35]);

        if r == 1
            title(ax, dim_names(c), "FontWeight","bold");
        end
        if c == 1
            ylabel(ax, metric_ylabel(r), "FontWeight","bold");
        end

        if r == 3
            yl = ylim(ax);
            y_group = yl(1) - 0.12*(yl(2)-yl(1));
            text(ax, mean(xpos(1:2)), y_group, "Held-in", ...
                'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
                'FontName','Times New Roman', 'FontSize',11, 'FontWeight','bold', ...
                'Color','k', 'Clipping','off');
            text(ax, mean(xpos(3:4)), y_group, "Held-out", ...
                'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
                'FontName','Times New Roman', 'FontSize',11, 'FontWeight','bold', ...
                'Color','k', 'Clipping','off');
        end

        hold(ax,'off');
    end
end

out_pdf = "calm_vs_poyo_heldin_heldout_3x3_vector.pdf";
print(fig, out_pdf, '-dpdf', '-painters', '-bestfit');
fprintf("[OK] Saved: %s\n", out_pdf);

function [sid, M] = load_calm_metrics(matfile, which)
    S = load(matfile);
    if ~isfield(S, which)
        error("CALM mat missing field '%s': %s", which, matfile);
    end
    T = S.(which);

    M.corr        = to_Nx3(double(T.corr));
    M.r2          = to_Nx3(double(T.r2));
    M.mse_trial_c = to_Nx3(double(T.mse_trial_c));

    sid = strings(size(M.corr,1), 1);
    if isfield(T,'base_session_id')
        sid = string(T.base_session_id(:));
    elseif isfield(T,'npz_name')
        sid = string(T.npz_name(:));
    elseif isfield(T,'session_id')
        sid = string(T.session_id(:));
    else
        sid = string((1:size(M.corr,1))');
        fprintf("[WARN] CALM '%s' has no base_session_id/session_id/npz_name -> will align by order.\n", which);
    end
    sid = erase(sid, ".npz");
end

function [sid, M] = load_poyo_metrics(matfile)
    S = load(matfile);

    req = ["corr","r2","mse_trial_c"];
    for k = 1:numel(req)
        if ~isfield(S, req(k))
            error("POYO mat missing '%s': %s", req(k), matfile);
        end
    end

    M.corr        = to_Nx3(double(S.corr));
    M.r2          = to_Nx3(double(S.r2));
    M.mse_trial_c = to_Nx3(double(S.mse_trial_c));

    n = size(M.corr,1);
    sid = strings(n,1);
    if isfield(S,'base_session_id')
        sid = string(S.base_session_id(:));
    elseif isfield(S,'session_id')
        sid = string(S.session_id(:));
    elseif isfield(S,'npz_name')
        sid = string(S.npz_name(:));
    else
        sid = string((1:n)');
        fprintf("[WARN] POYO has no base_session_id/session_id/npz_name -> will align by order.\n");
    end
    sid = erase(sid, ".npz");
end

function A = to_Nx3(X)
    if isempty(X)
        A = nan(0,3);
        return;
    end
    if isvector(X), X = X(:); end
    if size(X,2) == 3
        A = X;
    elseif size(X,1) == 3 && size(X,2) ~= 3
        A = X.';
    else
        error("Metric matrix must be N×3 (or 3×N). Got %dx%d.", size(X,1), size(X,2));
    end
end

function [A2, B2, sid_used, how] = align_by_sid_or_order(sidA, A, sidB, B, tag)
    sidA = normalize_sid(sidA);
    sidB = normalize_sid(sidB);

    haveA = ~isempty(sidA) && any(strlength(sidA) > 0);
    haveB = ~isempty(sidB) && any(strlength(sidB) > 0);

    if haveA && haveB
        [common, ia, ib] = intersect(sidA, sidB, 'stable');
        if ~isempty(common)
            A2 = index_metrics(A, ia);
            B2 = index_metrics(B, ib);
            sid_used = common;
            how = "sid";
            fprintf("[INFO] %s aligned by session_id: %d common sessions\n", tag, numel(common));
            return;
        else
            warning("[WARN] %s no common session ids -> will align by order.", tag);
        end
    else
        warning("[WARN] %s missing/empty session ids -> will align by order.", tag);
    end

    nA = size(A.corr,1);
    nB = size(B.corr,1);
    n  = min(nA, nB);

    if nA ~= nB
        warning("[WARN] %s session counts differ -> trunc to %d", tag, n);
    end

    A2 = index_metrics(A, 1:n);
    B2 = index_metrics(B, 1:n);
    sid_used = string((1:n)).';
    how = "order";
end

function sid = normalize_sid(sid)
    if nargin==0 || isempty(sid)
        sid = strings(0,1);
        return;
    end
    sid = string(sid(:));
    sid = erase(sid, ".npz");
    sid = strtrim(sid);
    sid(sid == "<missing>" | sid == "nan" | sid == "NaN") = "";
end

function M2 = index_metrics(M, idx)
    M2 = M;
    M2.corr        = M.corr(idx, :);
    M2.r2          = M.r2(idx, :);
    M2.mse_trial_c = M.mse_trial_c(idx, :);
end

function [v_in, v_out] = pick_metric_dim(M_in, M_out, metric_name, dim_idx)
    switch metric_name
        case "corr"
            v_in = M_in.corr(:, dim_idx);
            v_out = M_out.corr(:, dim_idx);
        case "r2"
            v_in = M_in.r2(:, dim_idx);
            v_out = M_out.r2(:, dim_idx);
        case "mse_trial_c"
            v_in = M_in.mse_trial_c(:, dim_idx);
            v_out = M_out.mse_trial_c(:, dim_idx);
        otherwise
            error("Unknown metric: %s", metric_name);
    end
end

function draw_box_outline(ax, x, y, w)
    y = y(~isnan(y));
    if isempty(y), return; end

    q = quantile(y, [0.25 0.50 0.75]);
    q1 = q(1); med = q(2); q3 = q(3);
    iqr = q3 - q1;

    lowFence = q1 - 1.5*iqr;
    highFence = q3 + 1.5*iqr;
    lw = min(y(y >= lowFence));
    uw = max(y(y <= highFence));

    xL = x - w/2; xR = x + w/2;

    line(ax, [xL xR], [q1 q1], 'Color','k', 'LineWidth',1.0);
    line(ax, [xL xR], [q3 q3], 'Color','k', 'LineWidth',1.0);
    line(ax, [xL xL], [q1 q3], 'Color','k', 'LineWidth',1.0);
    line(ax, [xR xR], [q1 q3], 'Color','k', 'LineWidth',1.0);

    line(ax, [xL xR], [med med], 'Color','k', 'LineWidth',1.0);

    line(ax, [x x], [q3 uw], 'Color','k', 'LineWidth',1.0);
    line(ax, [x x], [q1 lw], 'Color','k', 'LineWidth',1.0);

    capW = w*0.55;
    line(ax, [x-capW/2 x+capW/2], [uw uw], 'Color','k', 'LineWidth',1.0);
    line(ax, [x-capW/2 x+capW/2], [lw lw], 'Color','k', 'LineWidth',1.0);
end
