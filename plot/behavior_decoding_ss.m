clear; close all; clc;

method_names  = ["GLM","RRR","TCN","POYO+","Ours-Linear","Ours-Nonlinear"];
session_names = ["sub6-0716-visA-L23","sub10-0922-PM-L23","sub10-1113-AM-L23"];

nM = numel(method_names);
nS = numel(session_names);

corr_pitch = [...
    0.878200, 0.833600, 0.771300; ... % GLM
    0.878200, 0.833600, 0.766100; ... % RRR
    0.810600, 0.806300, 0.644500; ... % TCN
    0.871512, 0.677521, 0.831908; ... % POYO+
    0.900398, 0.866051, 0.799972; ... % Ours-Linear
    0.921037, 0.901522, 0.849327; ... % Ours-Nonlinear
];
corr_roll = [...
    0.807000, 0.864000, 0.878100; ... % GLM
    0.807000, 0.864000, 0.878400; ... % RRR
    0.752400, 0.843300, 0.817300; ... % TCN
    0.806116, 0.837372, 0.864569; ... % POYO+
    0.817837, 0.873580, 0.903797; ... % Ours-Linear
    0.860685, 0.913595, 0.919302; ... % Ours-Nonlinear
];
corr_yaw = [...
    0.866600, 0.894100, 0.892700; ... % GLM
    0.866600, 0.894100, 0.893900; ... % RRR
    0.780200, 0.898500, 0.873900; ... % TCN
    0.888056, 0.900768, 0.926104; ... % POYO+
    0.889320, 0.910877, 0.909943; ... % Ours-Linear
    0.889816, 0.937127, 0.920560; ... % Ours-Nonlinear
];

r2_pitch = [...
    0.692000, 0.655200, 0.527300; ... % GLM
    0.692000, 0.655200, 0.518800; ... % RRR
    0.579100, 0.611900, 0.288000; ... % TCN
    0.740536, 0.421134, 0.661548; ... % POYO+
    0.789580, 0.725198, 0.590292; ... % Ours-Linear
    0.833400, 0.783606, 0.697569; ... % Ours-Nonlinear
];
r2_roll = [...
    0.442400, 0.665400, 0.625800; ... % GLM
    0.442400, 0.665400, 0.628400; ... % RRR
    0.238700, 0.626900, 0.390700; ... % TCN
    0.545633, 0.588324, 0.685887; ... % POYO+
    0.558808, 0.702502, 0.717286; ... % Ours-Linear
    0.615744, 0.782349, 0.795960; ... % Ours-Nonlinear
];
r2_yaw = [...
    0.527600, 0.650100, 0.551400; ... % GLM
    0.527600, 0.650100, 0.551600; ... % RRR
    0.060400, 0.562300, 0.306400; ... % TCN
    0.652909, 0.609394, 0.750911; ... % POYO+
    0.600834, 0.698576, 0.424213; ... % Ours-Linear
    0.568339, 0.717839, 0.661696; ... % Ours-Nonlinear
];

% ---- MSE ----
mse_pitch = [...
    0.232500, 0.295200, 0.389800; ... % GLM
    0.232500, 0.295200, 0.398800; ... % RRR
    0.327600, 0.333700, 0.591700; ... % TCN
    0.193036, 0.473987, 0.291242; ... % POYO+
    0.153721, 0.233186, 0.329042; ... % Ours-Linear
    0.122910, 0.184693, 0.244569; ... % Ours-Nonlinear
];
mse_roll = [...
    0.231300, 0.232300, 0.101200; ... % GLM
    0.231300, 0.232300, 0.101300; ... % RRR
    0.329500, 0.260300, 0.170300; ... % TCN
    0.202668, 0.117816, 0.219799; ... % POYO+
    0.188541, 0.228235, 0.076975; ... % Ours-Linear
    0.166665, 0.166224, 0.055867; ... % Ours-Nonlinear
];
mse_yaw = [...
    0.132600, 0.152300, 0.076200; ... % GLM
    0.132600, 0.152300, 0.076200; ... % RRR
    0.264000, 0.163200, 0.126100; ... % TCN
    0.115940, 0.0611145, 0.120126; ... % POYO+
    0.110880, 0.151230, 0.080067; ... % Ours-Linear
    0.116051, 0.144582, 0.060489; ... % Ours-Nonlinear
];

colors = [...
    0.75 0.75 0.75;  % GLM: light gray
    0.55 0.55 0.55;  % RRR: dark gray
    0.62 0.52 0.82;  % TCN: purple
    0.45 0.65 0.95;  % POYO+: blue
    0.98 0.62 0.25;  % Ours-Linear: orange
    0.90 0.25 0.25;  % Ours-Nonlinear: red
];

dim_titles    = ["Pitch (dim0 / VX)", "Roll (dim1 / VY)", "Yaw (dim2 / VZ)"];
metric_titles = ["Correlation", "R^2", "MSE"];

data_corr = {corr_pitch, corr_roll, corr_yaw};
data_r2   = {r2_pitch,   r2_roll,   r2_yaw};
data_mse  = {mse_pitch,  mse_roll,  mse_yaw};

fig = figure('Color','w','Position',[60,60,1750,900]);
t = tiledlayout(3,3,'Padding','compact','TileSpacing','compact');

for c = 1:3
    ax = nexttile(t);
    plot_bar_box_on_ax(ax, data_corr{c}, method_names, colors, "Correlation", dim_titles(c));
end
for c = 1:3
    ax = nexttile(t);
    plot_bar_box_on_ax(ax, data_r2{c}, method_names, colors, "R^2", dim_titles(c));
end
for c = 1:3
    ax = nexttile(t);
    plot_bar_box_on_ax(ax, data_mse{c}, method_names, colors, "MSE", dim_titles(c));
end

out_pdf = "ss_decoding_3x3_vector.pdf";
print(fig, out_pdf, '-dpdf', '-painters', '-bestfit');
fprintf("[OK] Saved: %s\n", out_pdf);

function plot_bar_box_on_ax(ax, scores, labels, colors, ylab, ttl)
    axes(ax);
    cla(ax);
    hold(ax,'on');

    nModels = size(scores,1);
    nRuns   = size(scores,2);  

    means = mean(scores,2,'omitnan');

    b = bar(ax, 1:nModels, means, 'FaceColor','flat', 'EdgeColor','none', 'BarWidth',0.62);
    b.CData = colors;
    if isprop(b,'FaceAlpha'); b.FaceAlpha = 0.78; end

    if exist("boxchart","file") == 2
        x = repmat((1:nModels)', nRuns, 1);  
        y = scores(:);

        bc = boxchart(ax, x, y, ...
            'BoxFaceColor','w', 'BoxFaceAlpha',0.12, ...
            'WhiskerLineColor','k', ...
            'LineWidth',1.1);

        if isprop(bc,'BoxWidth'); bc.BoxWidth = 0.30; end

        if isprop(bc,'BoxEdgeColor')
            bc.BoxEdgeColor = 'k';
        elseif isprop(bc,'EdgeColor')
            bc.EdgeColor = 'k';
        end
        if isprop(bc,'MedianLineColor'); bc.MedianLineColor = 'k'; end

        if isprop(bc,'MarkerStyle'); bc.MarkerStyle = 'none'; end

    else
        ax2 = axes('Position', ax.Position, 'Color','none');
        hold(ax2,'on');

        boxplot(ax2, scores', 'Positions', 1:nModels, ...
            'Colors','k', 'Symbol','', 'Widths',0.30);

        boxes = findobj(ax2, 'Tag', 'Box');
        for j = 1:numel(boxes)
            xd = get(boxes(j),'XData');
            yd = get(boxes(j),'YData');
            patch('XData', xd, 'YData', yd, ...
                  'FaceColor','w', 'FaceAlpha',0.12, ...
                  'EdgeColor','k', 'LineWidth',1.1, ...
                  'Parent', ax2);
        end

        set(ax2,'XTick',[],'YTick',[],'Box','off');
        linkaxes([ax ax2],'xy');
        uistack(ax,'top');
    end

    yl = ylim(ax);
    y_text = yl(1) + 0.05*(yl(2)-yl(1));
    for i = 1:nModels
        text(ax, i, y_text, sprintf('%.4f', means(i)), ...
            'HorizontalAlignment','center', 'Color','w', ...
            'FontWeight','bold', 'FontSize',9);
    end

    hold(ax,'off');

    ax.FontName = 'Times New Roman';
    ax.FontSize = 11;
    ax.XTick = 1:nModels;
    ax.XTickLabel = labels;
    ax.XTickLabelRotation = 25;

    ylabel(ax, ylab, 'FontSize',12, 'FontWeight','bold');
    title(ax, ttl, 'FontSize',13);

    grid(ax,'off');
    ax.GridAlpha = 0.22;
    ax.XGrid = 'off';
    box(ax,'on');
end
