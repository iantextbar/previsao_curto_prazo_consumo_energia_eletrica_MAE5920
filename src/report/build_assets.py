"""Figuras e tabelas do relatório.

Lê os dados brutos (src/data) e as previsões exportadas pelo notebook (outputs/) e grava:
- figures/*.pdf (LaTeX) e figures/*.png (Word);
- tables/*.tex (incluídas em relatorio.tex por \\input).

Uso, na raiz do repositório:
    uv run --with matplotlib python src/report/build_assets.py

Os números das dobras (validação com janelas expansivas), do custo computacional e do Chronos-2
não estão nos CSVs e foram copiados das saídas impressas do notebook (execução no Colab de
2026-11-12); a célula de origem está indicada em cada bloco.
"""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'src' / 'data' / 'energydata_complete.csv'
OUTPUTS = ROOT / 'outputs'
HERE = Path(__file__).resolve().parent
FIGURES = HERE / 'figures'
TABLES = HERE / 'tables'

VAL_START, TEST_START = pd.Timestamp('2016-04-16'), pd.Timestamp('2016-05-07')
FOLD_BLOCKS = [('2016-03-05', '2016-03-26'), ('2016-03-26', '2016-04-16'), ('2016-04-16', '2016-05-07')]
HORIZON_STEPS = {10: 1, 60: 6}

# Paleta (dataviz, tema categórico padrão): a cor acompanha o modelo em todas as figuras.
INK, INK_2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
COLORS = {'gru': '#2a78d6', 'mlp_res': '#eb6834', 'arx': '#1baf7a', 'persistencia': '#8f8e89', 'chronos': '#eda100'}
LABELS = {'persistencia': 'Persistência', 'arx': 'ARX', 'mlp_res': 'MLP (residual)', 'gru': 'GRU'}

plt.rcParams.update({
    'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'legend.fontsize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.edgecolor': INK_2, 'axes.linewidth': 0.6,
    'axes.labelcolor': INK, 'xtick.color': INK_2, 'ytick.color': INK_2, 'text.color': INK,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03, 'axes.formatter.use_locale': False,
})
WIDTH = 6.3  # polegadas, largura do texto em A4 com margens de 2,5 cm


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f'{name}.pdf')
    fig.savefig(FIGURES / f'{name}.png', dpi=300)
    plt.close(fig)


def br(x, digits=1):
    """Número com vírgula decimal."""
    return f'{x:.{digits}f}'.replace('.', ',')


# ---------------------------------------------------------------------------- dados

def load_data():
    raw = pd.read_csv(DATA, parse_dates=['date']).set_index('date')
    return raw


def split_of(index):
    return np.where(index < VAL_START, 'train', np.where(index < TEST_START, 'val', 'test'))


def persistence_predictions(raw, reference):
    """Persistência nas mesmas origens de um modelo de referência (as origens comuns a todos os modelos)."""
    current = raw['Appliances']
    origins = reference[reference['seed'] == reference['seed'].min()]
    return origins.assign(modelo='persistencia', seed=0,
                          y_pred=current.reindex(origins['origem']).to_numpy())


def load_predictions(key, rodada):
    return pd.read_csv(OUTPUTS / f'preds_{key}_rodada{rodada}.csv', parse_dates=['origem', 'timestamp'])


def metrics(preds):
    """MAE e RMSE por semente, depois média e DP entre sementes."""
    err = preds.assign(ae=(preds['y_true'] - preds['y_pred']).abs(), se=(preds['y_true'] - preds['y_pred']) ** 2)
    per_seed = err.groupby(['split', 'horizonte_min', 'seed']).agg(MAE=('ae', 'mean'), MSE=('se', 'mean'))
    per_seed['RMSE'] = np.sqrt(per_seed['MSE'])
    grouped = per_seed.groupby(['split', 'horizonte_min'])
    return grouped[['MAE', 'RMSE']].mean().join(grouped[['MAE']].std().rename(columns={'MAE': 'MAE_dp'}))


# ---------------------------------------------------------------------------- figuras

def fig_eda(raw):
    """Perfil horário por dia da semana (mapa de calor) e distribuição do consumo."""
    y = raw['Appliances']
    profile = y.groupby([y.index.day_of_week, y.index.hour]).mean().unstack()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH, 2.25), gridspec_kw={'width_ratios': [1.65, 1]})

    im = ax1.imshow(profile.to_numpy(), aspect='auto', cmap='Blues', interpolation='nearest')
    ax1.set_yticks(range(7), ['seg', 'ter', 'qua', 'qui', 'sex', 'sáb', 'dom'])
    ax1.set_xticks(range(0, 24, 3), [f'{h}h' for h in range(0, 24, 3)])
    ax1.grid(False)
    ax1.set_xlabel('Hora do dia')
    ax1.set_title('(a) Consumo médio por hora e dia da semana', loc='left')
    bar = fig.colorbar(im, ax=ax1, fraction=0.04, pad=0.02)
    bar.set_label('Wh', fontsize=7.5)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7)

    bins = np.logspace(np.log10(y.min()), np.log10(y.max()), 40)
    ax2.hist(y, bins=bins, color=COLORS['gru'], edgecolor='white', linewidth=0.5)
    ax2.set_xscale('log')
    ax2.set_xlabel('Consumo em 10 min (Wh, escala log)')
    ax2.set_ylabel('Frequência')
    ax2.set_title('(b) Distribuição do consumo', loc='left')
    med, p95 = y.median(), y.quantile(0.95)
    top = ax2.get_ylim()[1] * 1.35
    ax2.set_ylim(0, top)
    for value, text, side in ((med, f'mediana\n{med:.0f} Wh', 'right'), (p95, f'p95\n{p95:.0f} Wh', 'left')):
        ax2.axvline(value, color=INK_2, linewidth=0.8)
        x = value * (0.93 if side == 'right' else 1.07)
        ax2.text(x, top * 0.97, text, fontsize=7, color=INK_2, va='top', ha=side)
    fig.tight_layout(w_pad=1.5)
    save(fig, 'fig1_eda')


def fig_protocol(raw):
    """Temperaturas diárias com os conjuntos e os blocos das dobras: divisão, dobras e mudança de distribuição."""
    rooms = [f'T{i}' for i in (1, 2, 3, 4, 5, 7, 8, 9)]
    daily = pd.DataFrame({'Interna (média dos cômodos)': raw[rooms].mean(axis=1), 'Externa (estação)': raw['T_out']})
    daily = daily.resample('D').mean()

    fig, ax = plt.subplots(figsize=(WIDTH, 2.5))
    start, end = raw.index.min(), raw.index.max()
    spans = [(start, VAL_START, 'treino', '#f3f2ee'), (VAL_START, TEST_START, 'validação', '#e6e5df'),
             (TEST_START, end, 'teste', '#d6d5cd')]
    for a, b, name, shade in spans:
        ax.axvspan(a, b, color=shade, linewidth=0, zorder=0)
    lines = {'Interna (média dos cômodos)': COLORS['mlp_res'], 'Externa (estação)': COLORS['gru']}
    for col, color in lines.items():
        ax.plot(daily.index, daily[col], color=color, linewidth=1.6, label=col, zorder=3)

    # faixa de temperatura interna do treino, para mostrar a extrapolação no teste
    train_max = daily.loc[daily.index < VAL_START, 'Interna (média dos cômodos)'].max()
    ax.axhline(train_max, color=COLORS['mlp_res'], linewidth=0.8, alpha=0.7, zorder=2)
    ax.text(start + pd.Timedelta(days=1), train_max + 0.5, 'máximo da média diária interna no treino', fontsize=7,
            color=INK_2)

    ymin, ymax = -4, 31
    ax.set_ylim(ymin, ymax)
    for a, b, name, _ in spans:
        ax.text(a + (b - a) / 2, ymin + 1.0, name, ha='center', fontsize=7.5, color=INK_2)
    for k, (a, b) in enumerate(FOLD_BLOCKS, start=1):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        y0 = ymax - 1.2
        ax.annotate('', xy=(a, y0), xytext=(b, y0),
                    arrowprops=dict(arrowstyle='|-|', color=INK, linewidth=0.8, shrinkA=0, shrinkB=0,
                                    mutation_scale=2.5))
        ax.text(a + (b - a) / 2, y0 + 0.5, f'bloco {k}', ha='center', fontsize=7, color=INK)
    ax.set_ylabel('Temperatura média diária (°C)')
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=7.5, borderaxespad=0.2)
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m/%Y'))
    ax.margins(x=0)
    fig.tight_layout()
    save(fig, 'fig2_protocolo')


def fig_by_hour(preds):
    """MAE no teste por hora do instante previsto, modelos finais."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.3), sharex=True)
    order = ['persistencia', 'arx', 'gru']
    for ax, minutes in zip(axes, (10, 60)):
        for key in order:
            part = preds[(preds['modelo'] == key) & (preds['split'] == 'test') & (preds['horizonte_min'] == minutes)]
            mae = (part['y_true'] - part['y_pred']).abs().groupby(part['timestamp'].dt.hour).mean()
            ax.plot(mae.index, mae.to_numpy(), color=COLORS[key], linewidth=1.6 if key == 'gru' else 1.2,
                    label=LABELS[key], marker='o' if key == 'gru' else None, markersize=2.5)
        ax.set_title(f'Horizonte de {minutes} min', loc='left')
        ax.set_xlabel('Hora do instante previsto')
        ax.set_xticks(range(0, 24, 3), [f'{h}h' for h in range(0, 24, 3)])
    axes[0].set_ylabel('MAE no teste (Wh)')
    axes[1].legend(loc='upper left')
    fig.tight_layout(w_pad=1.5)
    save(fig, 'fig3_mae_hora')


def fig_test_day(preds, raw):
    """Um dia do teste com pico: observado, GRU, ARX e intervalo de 80% do Chronos-2 (zero-shot, 7 dias), 60 min."""
    q = pd.read_csv(OUTPUTS / 'quantis_chronos2_zs1008_rodada3.csv', parse_dates=['origem', 'timestamp'])
    q = q[(q['split'] == 'test') & (q['horizonte_min'] == 60)].set_index('timestamp').sort_index()

    test = raw.loc[raw.index >= TEST_START, 'Appliances']
    by_day = test.groupby(test.index.date)
    full_days = by_day.size()[by_day.size() == 144].index
    evening_peak = test[test.index.hour.isin(range(16, 20))]
    evening_peak = evening_peak.groupby(evening_peak.index.date).max().loc[full_days]
    day = pd.Timestamp(evening_peak.sort_values().index[-2])  # dia completo com pico alto no fim da tarde
    lo, hi = day, day + pd.Timedelta(days=1)

    def mean_pred(key):
        part = preds[(preds['modelo'] == key) & (preds['split'] == 'test') & (preds['horizonte_min'] == 60)]
        return part.groupby('timestamp')['y_pred'].mean().loc[lo:hi]

    fig, ax = plt.subplots(figsize=(WIDTH, 2.4))
    band = q.loc[lo:hi]
    ax.fill_between(band.index, band['0.1'], band['0.9'], color=COLORS['chronos'], alpha=0.35, linewidth=0,
                    label='Chronos-2 zero-shot, intervalo de 80%')
    obs = raw.loc[lo:hi, 'Appliances']
    ax.plot(obs.index, obs.to_numpy(), color=INK, linewidth=1.0, label='Observado')
    ax.plot(*zip(*mean_pred('arx').items()), color=COLORS['arx'], linewidth=1.4, label='ARX')
    ax.plot(*zip(*mean_pred('gru').items()), color=COLORS['gru'], linewidth=1.6, label='GRU (média das sementes)')
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Hh'))
    ax.set_ylabel('Consumo (Wh)')
    ax.set_xlabel(f'Instante previsto, {day:%d/%m/%Y} (previsões feitas 60 min antes)')
    ax.legend(loc='upper left', ncol=2)
    fig.tight_layout()
    save(fig, 'fig4_dia_teste')
    return day


# ---------------------------------------------------------------------------- tabelas

def write(name, text):
    TABLES.mkdir(exist_ok=True)
    (TABLES / f'{name}.tex').write_text(text, encoding='utf-8')


# Saídas impressas do notebook (execução no Colab de 2026-11-12): células 99 (rodada 1), 126 (ARX + MLP),
# 139 (GRU, LSTM) e 142 (teste de todas as variantes). Formato: {split: (MAE 10, MAE 60)}, DP do teste
# entre sementes e RMSE de teste. O treino e a validação das variantes residuais não foram impressos
# (None); a validação do MLP (residual) vem do texto da célula 146, com uma casa decimal.
PRINTED = {
    'persistencia': {'train': (30.833, 60.285), 'val': (26.849, 53.290), 'test': (26.510, 46.024),
                     'test_dp': (None, None), 'rmse': (66.463, 101.289)},
    'arx': {'train': (31.646, 50.129), 'val': (27.446, 42.939), 'test': (27.083, 40.270),
            'test_dp': (None, None), 'rmse': (59.440, 78.386)},
    'mlp_r1': {'train': (28.901, 40.432), 'val': (26.783, 38.142), 'test': (36.526, 52.376),
               'test_dp': (3.280, 5.593), 'rmse': (66.073, 86.363)},
    'cnn_r1': {'train': (33.843, 41.615), 'val': (35.040, 43.114), 'test': (83.555, 91.135),
               'test_dp': (18.515, 11.723), 'rmse': (122.131, 130.203)},
    'mlp_res': {'train': None, 'val': (24.6, 37.6), 'test': (24.191, 35.352),
                'test_dp': (0.163, 0.832), 'rmse': (60.065, 81.133)},
    'cnn_res': {'train': None, 'val': None, 'test': (25.006, 37.284),
                'test_dp': (0.244, 0.782), 'rmse': (60.937, 83.133)},
    'arx_mlp': {'train': (25.906, 35.160), 'val': (25.237, 38.097), 'test': (24.552, 35.549),
                'test_dp': (0.396, 0.754), 'rmse': (59.290, 81.705)},
    'lstm': {'train': (27.761, 39.421), 'val': (24.552, 36.732), 'test': (24.245, 35.395),
             'test_dp': (0.127, 1.159), 'rmse': (61.067, 83.670)},
    'gru': {'train': (27.716, 40.764), 'val': (24.256, 36.324), 'test': (23.698, 35.028),
            'test_dp': (0.301, 0.834), 'rmse': (60.634, 81.140)},
}


def fill_from_csv(key, m):
    """Completa as lacunas de PRINTED com um CSV de previsões, se ele reproduzir o MAE de teste impresso."""
    row = PRINTED[key]
    got = m.loc[[('test', 10), ('test', 60)], 'MAE'].to_numpy()
    if np.abs(got - row['test']).max() >= 0.01:
        return False
    for split in ('train', 'val'):
        if row[split] is None or split == 'val' and key == 'mlp_res':
            row[split] = tuple(m.loc[[(split, 10), (split, 60)], 'MAE'])
    return True


def table_results():
    """MAE em treino/validação/teste e RMSE no teste, rodadas 1 e 2."""
    blocks = [
        ('Rodada 1', [('persistencia', 'Persistência'), ('arx', 'ARX'), ('mlp_r1', 'MLP'), ('cnn_r1', 'CNN 1D')]),
        ('Rodada 2', [('mlp_res', 'MLP (residual)'), ('cnn_res', 'CNN 1D (residual)'), ('arx_mlp', 'ARX + MLP'),
                      ('lstm', 'LSTM'), ('gru', '\\textbf{GRU}')]),
    ]
    lines = [
        r'\begin{tabular}{@{}lcccccccc@{}}', r'\toprule',
        r' & \multicolumn{2}{c}{MAE treino} & \multicolumn{2}{c}{MAE validação} & \multicolumn{2}{c}{MAE teste}'
        r' & \multicolumn{2}{c}{RMSE teste} \\',
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}',
        r'Modelo & 10 & 60 & 10 & 60 & 10 & 60 & 10 & 60 \\', r'\midrule',
    ]
    for title, rows in blocks:
        lines.append(rf'\multicolumn{{9}}{{@{{}}l}}{{\emph{{{title}}}}} \\')
        for key, label in rows:
            row = PRINTED[key]
            vals = []
            for split in ('train', 'val'):
                vals += ['---', '---'] if row[split] is None else [br(x) for x in row[split]]
            for mean, sd in zip(row['test'], row['test_dp']):
                vals.append(br(mean) if sd is None else f'{br(mean)} ({br(sd)})')
            vals += [br(x) for x in row['rmse']]
            lines.append(f'{label} & ' + ' & '.join(vals) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('tab_resultados', '\n'.join(lines) + '\n')


# Saídas impressas do notebook: célula 113 (representações, MASE por dobra), célula 119 (variantes,
# média das dobras por horizonte) e célula 125 (ARX + MLP, MASE por dobra).
FOLD_REPR = {  # (representação, modelo, horizonte): MASE nas dobras 1, 2, 3
    ('níveis', 'ARX', 10): (1.057, 0.889, 0.890), ('níveis', 'ARX', 60): (0.787, 0.707, 0.712),
    ('níveis', 'MLP', 10): (1.440, 1.058, 0.911), ('níveis', 'MLP', 60): (0.996, 0.761, 0.657),
    ('variações', 'ARX', 10): (1.097, 0.912, 0.900), ('variações', 'ARX', 60): (0.835, 0.736, 0.744),
    ('variações', 'MLP', 10): (1.308, 2.321, 0.954), ('variações', 'MLP', 60): (0.874, 2.490, 0.671),
}
FOLD_VARIANTS = {  # variante: (MASE médio 10 min, MASE médio 60 min)
    'MLP': (1.136, 0.805), 'MLP (24 h)': (1.173, 0.796), 'MLP (residual)': (0.863, 0.677),
    'MLP (residual + 24 h)': (0.864, 0.680),
    'CNN 1D': (1.860, 1.048), 'CNN 1D (24 h)': (1.772, 1.011), 'CNN 1D (residual)': (0.878, 0.705),
    'CNN 1D (residual + 24 h)': (0.882, 0.704),
}
FOLD_ARX_MLP = {'ARX': {10: (1.057, 0.889, 0.890), 60: (0.787, 0.707, 0.712)},
                'ARX + MLP': {10: (1.011, 0.875, 0.870), 60: (0.939, 0.634, 0.654)}}


def table_folds():
    """MASE médio nas três dobras: representações (rodada 1 → 2) e variantes da rodada 2."""
    def fmt(a, b, bold=False):
        cells = [f'{a:.3f}', f'{b:.3f}', f'{(a + b) / 2:.3f}']
        cells = [c.replace('.', ',') for c in cells]
        if bold:
            cells[-1] = rf'\textbf{{{cells[-1]}}}'
        return ' & '.join(cells)

    lines = [r'\begin{tabular}{@{}llccc@{}}', r'\toprule',
             r'Decisão & Configuração & 10 min & 60 min & Média \\', r'\midrule']
    repr_means = {}
    for rep in ('níveis', 'variações'):
        for model in ('ARX', 'MLP'):
            a, b = (np.mean(FOLD_REPR[(rep, model, h)]) for h in (10, 60))
            repr_means.setdefault(rep, []).extend([a, b])
            lines.append(rf'Representação & {model}, {rep} & {fmt(a, b)} \\')
    lines.append(r'\addlinespace')
    for name, (a, b) in FOLD_VARIANTS.items():
        bold = name in ('MLP (residual)', 'CNN 1D (residual)')
        lines.append(rf'Variante & {name} & {fmt(a, b, bold)} \\')
    lines.append(r'\addlinespace')
    for name, d in FOLD_ARX_MLP.items():
        a, b = np.mean(d[10]), np.mean(d[60])
        lines.append(rf'Correção do ARX & {name} & {fmt(a, b, name == "ARX + MLP")} \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('tab_dobras', '\n'.join(lines) + '\n')
    return {rep: float(np.mean(v)) for rep, v in repr_means.items()}


# Célula 145 (custo computacional, 5 sementes; GRU e LSTM em GPU, demais em CPU).
COST = [
    ('ARX (Ridge)', 'defasagens $y_{t-11..t}$, $y_{t+h-144}$, exógenas em $t$', 'linear, um modelo por horizonte', '---', '---', '---'),
    ('MLP', '49 variáveis (12 defasagens em $\\log(1+y)$, estat. móveis, sensores, calendário)', '2 camadas ocultas de 128, dropout 0,2', '23\\,686', '7,3', '1,0'),
    ('CNN 1D', 'janela de 6 passos $\\times$ 36 canais', '2 conv.\\ (32 filtros, kernel 3), dropout 0,2', '7\\,750', '8,3', '2,0'),
    ('ARX + MLP', 'entradas do MLP + previsões do ARX', 'MLP sobre os resíduos do ARX (perda L1)', '23\\,426', '7,7', '0,9'),
    ('GRU', 'janela de 72 passos (12 h) $\\times$ 30 variáveis + calendário do alvo', '2 camadas de 128, dropout 0,1', '160\\,786', '10,0', '22,9'),
    ('LSTM', 'idem GRU', 'idem GRU', '214\\,290', '11,5', '28,7'),
]


def table_models():
    lines = [r'\begin{tabular}{@{}lp{4.6cm}p{3.6cm}rrr@{}}', r'\toprule',
             r'Modelo & Entrada & Arquitetura & Parâm. & Treino (s) & Inf. (ms) \\', r'\midrule']
    for row in COST:
        lines.append(' & '.join(row) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('tab_modelos', '\n'.join(lines) + '\n')


# Célula 160 (Chronos-2, teste; execução única).
CHRONOS = [
    ('Zero-shot, 12 h', 26.511, 41.098, 0.779, 0.760, 0.231, 0.365),
    ('Zero-shot, 7 dias', 23.991, 34.465, 0.761, 0.779, 0.209, 0.302),
    ('LoRA, 7 dias', 23.611, 33.267, 0.791, 0.792, 0.204, 0.287),
    ('Zero-shot, 7 dias + clima futuro$^\\dagger$', 24.175, 34.458, 0.769, 0.776, 0.210, 0.302),
]


def table_chronos():
    lines = [r'\begin{tabular}{@{}lcccccc@{}}', r'\toprule',
             r' & \multicolumn{2}{c}{MAE (Wh)} & \multicolumn{2}{c}{Cobertura 80\%} & \multicolumn{2}{c}{WQL} \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}',
             r'Chronos-2 & 10 & 60 & 10 & 60 & 10 & 60 \\', r'\midrule']
    for name, *v in CHRONOS:
        cells = [br(v[0]), br(v[1])] + [f'{x * 100:.0f}\\%' for x in v[2:4]] + [br(x, 3) for x in v[4:]]
        lines.append(f'{name} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('tab_chronos', '\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------- principal

def main():
    raw = load_data()
    keys = {'arx': ('arx', 2), 'mlp_r1': ('mlp', 1), 'cnn_r1': ('cnn_1d', 1), 'mlp_res': ('mlp_res', 2),
            'cnn_res': ('cnn_1d_res', 2), 'arx_mlp': ('arx_mlp', 2), 'gru': ('gru', 2), 'lstm': ('lstm', 2)}
    preds = {k: load_predictions(*v) for k, v in keys.items() if (OUTPUTS / f'preds_{v[0]}_rodada{v[1]}.csv').exists()}
    preds['persistencia'] = persistence_predictions(raw, preds['arx'])

    # Conferência: só usamos um CSV se ele reproduz o MAE de teste impresso no notebook. CSVs de uma
    # execução reduzida (smoke) ou antiga não passam e ficam fora das figuras.
    valid = []
    for key, p in preds.items():
        m = metrics(p)
        got = m.loc[[('test', 10), ('test', 60)], 'MAE'].to_numpy()
        ok = fill_from_csv(key, m)
        valid += [key] if ok else []
        print(f'{"OK " if ok else "---"} {key:13s} CSV {got[0]:8.3f} / {got[1]:8.3f}   '
              f'notebook {PRINTED[key]["test"][0]:.3f} / {PRINTED[key]["test"][1]:.3f}')
    for key in ('persistencia', 'arx', 'gru'):
        assert key in valid, f'{key}: CSV não confere com o notebook; as figuras 3 e 4 dependem dele'

    long = pd.concat([preds[k].assign(modelo=k) for k in ('persistencia', 'arx', 'gru')])
    fig_eda(raw)
    fig_protocol(raw)
    fig_by_hour(long)
    day = fig_test_day(long, raw)

    table_results()
    repr_means = table_folds()
    table_models()
    table_chronos()

    missing = [k for k, row in PRINTED.items() if row['train'] is None or row['val'] is None]
    print(f'MASE médio nas dobras por representação: {repr_means}')
    print(f'Dia da figura 4: {day:%Y-%m-%d}')
    if missing:
        print(f'Sem treino/validação na tabela (copie os CSVs do Colab para outputs/): {missing}')


if __name__ == '__main__':
    main()
