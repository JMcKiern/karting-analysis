import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def reorderLegend(ax, new_order):
    handles, labels = ax.get_legend_handles_labels()
    legend_dict = dict(zip(labels, handles))
    handles, labels = zip(*[(legend_dict[label], label) for label in new_order])
    ax.legend(list(handles), list(labels), loc='center left', bbox_to_anchor=(1, 0.5))

def getResults(df):
    df = df.groupby('Driver')['Lap Time'].agg(['count', 'sum']).rename(columns={'count': 'No. Laps', 'sum': 'Total Time'})
    df = df.sort_values(['No. Laps', 'Total Time'], ascending=[False, True])
    df['Position'] = range(1, 1 + len(df))
    df = df.reset_index()
    return df

def printSummary(df):
    print('Data Sample:')
    print(df.head())
    print('')

    print('Mean Lap Time by Driver:')
    print(df.groupby('Driver')['Lap Time'].mean().sort_values())
    print('')

    print('Fastest 10 Laps:')
    print(df.sort_values('Lap Time').head(10).to_string(index=False))
    print('')

    print('Slowest 10 Laps')
    print(df.sort_values('Lap Time', ascending=False).head(10).to_string(index=False))
    print('')

def plotLapTimeDist(df, fig_num, plot=None):
    if plot is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot
    ax.set_title('Fig {}: Lap Time Distribution by Driver'.format(fig_num))

    sns.set_style('whitegrid')
    sns.boxplot(x='Lap Time', y='Driver', data=df, sym='', saturation=0.3, ax=ax)
    sns.stripplot(x='Lap Time', y='Driver', data=df, linewidth=0.5, ax=ax)

    ax.set_axisbelow(True)
    ax.xaxis.grid(True)
    ax.set_yticks(np.arange(-0.5, len(ax.get_yticks()) + 0.5, 1), minor=True)
    ax.tick_params(axis='y', left=False)
    ax.yaxis.grid(True, which='minor')

def plotPosUsingFastestNLaps(df, fig_num, plot=None):
    # Plot driver positions using n fastest laps
    df_nlap_pos = pd.DataFrame(columns=['Driver', 'Max Laps', 'No. Laps', 'Position'])
    for num_laps in range(1, 20):
        df_nlap = df.groupby('Driver')['Lap Time'].nsmallest(num_laps).reset_index()[['Driver', 'Lap Time']]
        df_res = getResults(df_nlap)
        df_res['Max Laps'] = num_laps
        df_nlap_pos = df_nlap_pos.append(df_res)
    df_nlap_pos = df_nlap_pos.reset_index()

    if plot is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot
    ax.set_title('Fig {}: Race Positions Taking Only Each Driver\'s N Fastest Laps'.format(fig_num))
    sns.lineplot(x='Max Laps', y='Position', hue='Driver', style='Driver', markers=True, dashes=False, markersize=8, data=df_nlap_pos)

    drivers_sorted = df_nlap_pos[df_nlap_pos['Max Laps'] == df_nlap_pos['Max Laps'].max()].sort_values('Position')['Driver'].tolist()
    reorderLegend(ax, drivers_sorted)

    ax.invert_yaxis()
    ax.xaxis.set_ticks(range(1, df_nlap_pos['Max Laps'].nunique() + 1))
    ax.yaxis.set_ticks(range(1, df_nlap_pos['Driver'].nunique() + 1))

def plotPosOverRace(df, fig_num, grid_positions, plot=None):
    df_grid = pd.DataFrame(grid_positions, columns=['Driver'])
    df_grid['No. Laps'] = 0
    df_grid['Position'] = range(1, 1 + len(df_grid))

    # Plot driver positions over race
    df_lap_pos = pd.DataFrame(columns=['Driver', 'No. Laps', 'Position'])
    df_lap_pos = df_lap_pos.append(df_grid)
    for num_laps in range(1, 20):
        df_lap = df[df['Lap Number'] <= num_laps]
        df_res = getResults(df_lap)
        df_res['No. Laps'] = num_laps
        df_lap_pos = df_lap_pos.append(df_res)
    df_lap_pos = df_lap_pos.reset_index()

    if plot is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot
    ax.set_title('Fig {}: Race Positions Over Race'.format(fig_num))
    sns.lineplot(x='No. Laps', y='Position', hue='Driver', style='Driver', markers=True, dashes=False, markersize=8, data=df_lap_pos)

    drivers_sorted = df_lap_pos[df_lap_pos['No. Laps'] == df_lap_pos['No. Laps'].max()].sort_values('Position')['Driver'].tolist()
    reorderLegend(ax, drivers_sorted)

    ax.invert_yaxis()
    ax.xaxis.set_ticks(range(df_lap_pos['No. Laps'].nunique()))
    ax.yaxis.set_ticks(range(1,df_lap_pos['Driver'].nunique() + 1))

def plotIntervalOverRace(df, fig_num, final_positions, plot=None):
    # Plot intervals over race
    df_lap_pos = pd.DataFrame(columns=['Driver', 'No. Laps', 'Position'])
    for num_laps in range(1, 20):
        df_lap = df[df['Lap Number'] <= num_laps]
        df_res = getResults(df_lap)
        # df_res['No. Laps'] = num_laps
        df_res['Total Time'] = df_res.apply(lambda x: x['Total Time'] if x['No. Laps'] == num_laps else x['Total Time'], axis=1)
        df_lap_pos = df_lap_pos.append(df_res)
    df_lap_pos = df_lap_pos.reset_index()

    # Get interval from leader
    df_lap_pos['Interval'] = df_lap_pos.groupby('No. Laps')['Total Time'].transform(lambda x: x - x.min())

    if plot is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot
    ax.set_title('Fig {}: Interval Over Race'.format(fig_num))
    sns.lineplot(x='No. Laps', y='Interval', hue='Driver', style='Driver', markers=True, dashes=False, markersize=8, data=df_lap_pos)

    reorderLegend(ax, final_positions)

    ax.invert_yaxis()
    ax.xaxis.set_ticks(range(1,df_lap_pos['No. Laps'].nunique() + 1))

def plotLapTimeOverRace(df, fig_num, plot=None):
    # Plot driver lap times over race
    if plot is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot
    ax.set_title('Fig {}: Lap Times Over Race'.format(fig_num))
    sns.lineplot(x='Lap Number', y='Lap Time', hue='Driver', style='Driver', markers=True, dashes=False, markersize=8, data=df)

    ax.xaxis.set_ticks(range(1,df['Lap Number'].nunique() + 1))

def main():
    parser = argparse.ArgumentParser(description='Analyse karting lap times')
    parser.add_argument('filepath', help='filepath of the csv with lap times')
    args = parser.parse_args()
    df = pd.read_csv(args.filepath)

    grid_positions = df[df['Mode'] == 'Qualifying'].groupby('Driver')['Lap Time'].mean().sort_values().index.tolist()

    # Filter to only race data
    df = df[df['Mode'] == 'Race']

    # TODO: Add units to columns and axis labels
    # TODO: Consider orientation of the graphs
    # TODO: Generalise the fig, ax params
    # TODO: Add comments to graphing functions

    # TODO: Sort data by fastest to slowest driver

    final_positions = df.groupby('Driver')['Lap Time'].mean().sort_values().index.tolist()

    printSummary(df)
    plotLapTimeDist(df, 1)
    plotLapTimeOverRace(df, 2)
    plotIntervalOverRace(df, 3, final_positions)
    plotPosOverRace(df, 4, grid_positions)
    plotPosUsingFastestNLaps(df, 5)
    plt.show()

if __name__ == '__main__':
    main()
