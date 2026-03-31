clear; close all; clc;

method_names = ["iTransformer","PatchTST","TCN","Ours","POCO"];

colors = [...
    0.45 0.65 0.95;  % iTransformer: blue
    0.75 0.75 0.75;  % PatchTST: light gray
    0.62 0.52 0.82;  % TCN: purple
    0.45 0.65 0.95;  % POCO: blue-ish
    0.90 0.25 0.25;  % Ours: red

];

corr_scores = [...
    0.07292695603120111, 0.1243450907814602, 0.0494023753844623;  % iTransformer
    0.14986654983749934, 0.2507231114560083, 0.1746659174512619;  % PatchTST
    0.15557523071765900, 0.2558723986148834, 0.14477145671844482; % TCN
    0.2647,              0.3780,             0.2277;              % POCO 
    0.304976,            0.466727,           0.435723;            % Ours 

];

fig = figure('Color','w','Position',[80,80,1350,520]);
ax = axes('Position',[0.05 0.16 0.93 0.78]);
plot_bar_box_on_ax(ax, corr_scores, method_names, colors, "Correlation", "Ours vs POCO (Pearson Corr)");

out_pdf = "ss_forecasting_real.pdf";
print(fig, out_pdf, '-dpdf', '-painters', '-bestfit');
fprintf("[OK] Saved: %s\n", out_pdf);

function plot_bar_box_on_ax(ax, scores, labels, colors, ylab, ttl)
    axes(ax); 
    cla(ax);
    hold(ax,'on');

    nModels = size(scores,1);
    nRuns   = size(scores,2);   
    means   = mean(scores,2,'omitnan');


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
        ax2 = axes('Position', ax.Position, 'Color','none'); %#ok<LAXES>
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
        text(ax, i, y_text, sprintf('%.2f%', means(i)*100), ...
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

    grid(ax,'on');
    ax.GridAlpha = 0.22;
    ax.YGrid = 'off';      
    ax.XGrid = 'off';      

    box(ax,'on');
end
