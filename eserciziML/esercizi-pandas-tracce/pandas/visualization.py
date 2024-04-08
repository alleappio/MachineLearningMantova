import joypy  # e.g., conda install -c conda-forge joypy | pip install joypy
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import warnings

warnings.filterwarnings(action='once')

large = 22
med = 16
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")

# Version
print(mpl.__version__)  # > 3.0.0
print(sns.__version__)  # > 0.9.0


def plot1():
    # Asse Y secondario ####################################################################################################
    # Importo i dati
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/economics.csv")
    x = df['date']
    y1 = df['psavert']
    y2 = df['unemploy']

    # Disegno Line1 (asse Y primario)
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 9), dpi=80)
    ax1.plot(x, y1, color='tab:red')

    # Disegno Line2 (asse Y secondario)
    ax1.plot(x, y2, color='tab:blue')

    # Personalizzo il grafico
    ax1.set_xlabel('Year', fontsize=20)
    ax1.tick_params(axis='x', rotation=0, labelsize=12)
    ax1.set_ylabel('Personal Savings Rate', color='tab:red', fontsize=20)
    ax1.grid(alpha=.4)

    plt.show()


def plot2():
    # Istogrammi marginali #################################################################################################
    # Importo i dati
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/mpg_ggplot2.csv")

    # Creo la Fig e le gridspec
    fig = plt.figure(figsize=(16, 10), dpi=80)
    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

    # Definisco gli assi
    ax_main = fig.add_subplot(grid[:-1, :-1])
    ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
    ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

    # Scatterplot
    ax_main.scatter('displ', 'hwy', s=df.cty * 4, c=df.manufacturer.astype('category').cat.codes, alpha=.9, data=df,
                    cmap="tab10", edgecolors='gray', linewidths=.5)

    # Istorgramma in basso
    ax_bottom.hist(df.displ, 40, histtype='stepfilled', orientation='vertical', color='deeppink')
    ax_bottom.invert_yaxis()

    # Istogramma a destra
    ax_right.hist(df.hwy, 40, histtype='stepfilled', orientation='horizontal', color='deeppink')

    # Personalizzo il grafico
    ax_main.set(title='Scatterplot with Histograms \n displ vs hwy', xlabel='displ', ylabel='hwy')
    ax_main.title.set_fontsize(20)
    for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
        item.set_fontsize(14)
    xlabels = ax_main.get_xticks().tolist()
    ax_main.set_xticklabels(xlabels)
    plt.show()


# Aggiungo il numero di oggetti dentro le box
def add_n_obs(df, group_col, y):
    medians_dict = {grp[0]: grp[1][y].median() for grp in df.groupby(group_col)}
    xticklabels = [x.get_text() for x in plt.gca().get_xticklabels()]
    n_obs = df.groupby(group_col)[y].size().values
    for (x, xticklabel), n_ob in zip(enumerate(xticklabels), n_obs):
        plt.text(x, medians_dict[xticklabel] * 1.01, "#obs : " + str(n_ob), horizontalalignment='center',
                 fontdict={'size': 14}, color='white')

def plot3():
    # Box plot #############################################################################################################
    # Importo i dati
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")

    # Disegno il grafico
    plt.figure(figsize=(13, 10), dpi=80)

    add_n_obs(df, group_col='class', y='hwy')

    # Personalizzo il grafico
    plt.title('Box Plot of Highway Mileage by Vehicle Class', fontsize=22)
    plt.ylim(10, 40)
    plt.show()


def plot4():
    # Categorical plot #####################################################################################################
    # Importo i dati
    titanic = sns.load_dataset("titanic")

    fig = plt.figure(figsize=(16, 10), dpi=80)

    # Disegno i grafici
    #g = sns.catplot("alive", col="deck", col_wrap=4,
    #                data=titanic[titanic.deck.notnull()],
    #                kind="count", height=3.5, aspect=.8,
    #                palette='tab20')

    sns.catplot(x="age", y="embark_town",
                hue="sex", col="class",
                data=titanic[titanic.embark_town.notnull()],
                orient="h", height=5, aspect=1, palette="tab10",
                kind="violin", dodge=True, cut=0, bw=.2)

    # Personalizzo il grafico
    fig.suptitle('sf')
    plt.show()


def plot5():
    # Diagramma delle aree #################################################################################################
    # Importo i dati
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/economics.csv")

    # Preparo i dati
    x = df['date'].values.tolist()
    y1 = df['psavert'].values.tolist()
    y2 = df['uempmed'].values.tolist()
    mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive']
    columns = ['psavert', 'uempmed']

    # Disegno i grafici
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=80)
    ax.plot(x, y1, label=columns[1], alpha=0.5, color=mycolors[1], linewidth=2)
    ax.plot(x, y2, label=columns[0], alpha=0.5, color=mycolors[0], linewidth=2)

    # Personalizzo il grafico
    ax.set_title('Personal Savings Rate vs Median Duration of Unemployment', fontsize=18)
    ax.set(ylim=[0, 30])
    ax.legend(loc='best', fontsize=12)
    plt.xticks(x[::50], fontsize=10, horizontalalignment='center')
    plt.yticks(np.arange(2.5, 30.0, 2.5), fontsize=10)
    plt.xlim(-10, x[-1])

    # disegno le linee piu' spesse
    for y in np.arange(2.5, 30.0, 2.5):
        plt.hlines(y, xmin=0, xmax=len(x), colors='black', alpha=0.3, linestyles="--", lw=0.5)

    # Schiarisco i bordi
    plt.gca().spines["top"].set_alpha(0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(0)
    plt.gca().spines["left"].set_alpha(.3)
    plt.show()


def plot6():
    # Matrice di correlazione ##############################################################################################
    # Importo i dati
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mtcars.csv")

    # Disegno i grafici
    plt.figure(figsize=(12, 10), dpi=80)
    sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='RdYlGn', center=0,
                 annot=True)
    #sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, center=0)

    # Personalizzo il grafico
    plt.title('Correlogram of mtcars', fontsize=22)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


def plot7():
    # Grafici delle distribuzioni ##########################################################################################
    # Importo i dati
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")

    # Disegno i grafici
    plt.figure(figsize=(13, 10), dpi=80)
    sns.distplot(df.loc[df['class'] == 'compact', "cty"], color="dodgerblue", label="Compact", hist_kws={'alpha': .7},
                 kde_kws={'linewidth': 3})
    sns.distplot(df.loc[df['class'] == 'suv', "cty"], color="orange", label="SUV", hist_kws={'alpha': .7},
                 kde_kws={'linewidth': 3})
    plt.ylim(0, 0.35)

    # Decoration
    plt.title('Density Plot of City Mileage by Vehicle Type', fontsize=22)
    plt.legend()
    plt.show()


def plot8():
    # Diegramma a torta ####################################################################################################
    # Importo i dati
    df_raw = pd.read_csv("https://github.com/selva86/datasets/raw/master/mpg_ggplot2.csv")

    # Preparo i dati
    df = df_raw.groupby('class').size().reset_index(name='counts')
    data = df['counts']
    categories = df['class']

    # Disegno il grafico
    explode = [0, 0, 0, 0, 0, 0.1, 0]
    fig, ax = plt.subplots(figsize=(12, 7), subplot_kw=dict(aspect="equal"), dpi=80)

    def func(pct, allvals):
        absolute = int(pct / 100. * np.sum(allvals))
        return "{:.1f}% ({:d} )".format(pct, absolute)


    wedges, texts, autotexts = ax.pie(data,
                                      autopct=lambda pct: func(pct, data),
                                      textprops=dict(color="w"),
                                      colors=plt.cm.Dark2.colors,
                                      startangle=140,
                                      explode=explode)

    # Personalizzo il grafico
    ax.legend(wedges, categories, title="Vehicle Class", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=10, weight=700)
    ax.set_title("Class of Vehicles: Pie Chart")
    plt.show()


def plot9():
    # Diagramma delle divergenze ###########################################################################################
    # Importo i dati
    df = pd.read_csv("https://github.com/selva86/datasets/raw/master/mtcars.csv")

    # Preparo i dati
    x = df.loc[:, ['mpg']]
    df['mpg_z'] = (x - x.mean()) / x.std()
    df['colors'] = 'black'

    # Imposto il colore delle FIAT
    df.loc[df.cars == 'Fiat X1-9', 'colors'] = 'darkorange'
    df.sort_values('mpg_z', inplace=True)
    df.reset_index(inplace=True)

    # Disegno il grafico
    plt.figure(figsize=(14, 16), dpi=80)
    plt.hlines(y=df.index, xmin=0, xmax=df.mpg_z, color=df.colors, alpha=0.4, linewidth=1)
    plt.scatter(df.mpg_z, df.index, color=df.colors, s=[600 if x == 'Fiat X1-9' else 300 for x in df.cars], alpha=0.6)
    plt.yticks(df.index, df.cars)
    plt.xticks(fontsize=12)

    # Annoto il grafico
    plt.annotate('Mercedes Models', xy=(0.0, 11.0), xytext=(1.0, 11), xycoords='data',
                 fontsize=15, ha='center', va='center',
                 bbox=dict(boxstyle='square', fc='firebrick'),
                 arrowprops=dict(arrowstyle='-[, widthB=2.0, lengthB=1.5', lw=2.0, color='steelblue'), color='white')

    # Aggiungo i rettangoli
    p1 = patches.Rectangle((-2.0, -1), width=.3, height=3, alpha=.2, facecolor='red')
    p2 = patches.Rectangle((1.5, 27), width=.8, height=5, alpha=.2, facecolor='green')
    plt.gca().add_patch(p1)
    plt.gca().add_patch(p2)

    # Personalizzo il grafico
    plt.title('Diverging Bars of Car Mileage', fontdict={'size': 20})
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()


if __name__ == '__main__':
    # 3, 7
    plot7()

