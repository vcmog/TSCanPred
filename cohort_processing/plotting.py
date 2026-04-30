import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_population_pyramid(df, title=None):
    """
    Given dataframe of attributes relating to patients, plot a population pyramid.

    Args:
        df (DataFrame): Dataframe with columns including:
            - 'AGE_AT_INDEX':
    """
    total_pop = len(df)
    # make age groups
    bins = [i for i in range(0, 120, 5)]
    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins) - 1)]

    dems = df[["AGE_AT_INDEX", "GNDR_CD"]]
    female_age_groups = pd.cut(
        dems[dems["GNDR_CD"] == 1]["AGE_AT_INDEX"],
        bins=bins,
        labels=labels,
        right=False,
    )
    male_age_groups = pd.cut(
        dems[dems["GNDR_CD"] == 0]["AGE_AT_INDEX"],
        bins=bins,
        labels=labels,
        right=False,
    )

    female_age_counts = 100 * (female_age_groups.value_counts() / total_pop)
    male_age_counts = 100 * (male_age_groups.value_counts() / total_pop)

    dem_counts = (
        female_age_counts.to_frame()
        .join(male_age_counts.to_frame(), lsuffix="_Female", rsuffix="_Male")
        .reset_index()
    )
    dem_counts.rename(
        columns={"count_Female": "Female", "count_Male": "Male"}, inplace=True
    )
    dem_counts["Male"] = -1 * dem_counts["Male"]

    if title:
        title = title
    else:
        title = "Distribution of age by gender in the cohort"

    cols = sns.color_palette("mako")
    pop_pyramid = sns.barplot(
        x="Female", data=dem_counts, y="AGE_AT_INDEX", label="Female", color=cols[5]
    )
    pop_pyramid = sns.barplot(
        x="Male", data=dem_counts, y="AGE_AT_INDEX", label="Male", color=cols[3]
    )
    pop_pyramid.set(title=title, xlabel="% of population", ylabel="Age Band")

    max_pop = math.ceil(max(female_age_counts.max(), abs(male_age_counts.min())))
    xticks = np.arange(-max_pop, max_pop + 1, 1)
    xlabels = [str(abs(x)) for x in xticks]
    plt.xticks(xticks, xlabels)
    plt.show()


def plot_case_control_age(
    df, save_path=None, title="Density of ages within the case and control populations"
):
    """
    Plot KDE plots of age distribution for cases and controls.

    Args:
        df (DataFrame):         Dataframe with columns:
            - 'OUTCOME':        Binary, where 1=case and 0=control.
            - 'AGE_AT_INDEX':   Numeric, age of patient.
        save_path (str or None):Path where plot should be saved, or None. If None, plot is not saved.
        title (str):            Title for plot.

    """
    # split into case and control subsets
    case_atts = df[df["OUTCOME"] == 1]
    control_atts = df[df["OUTCOME"] == 0]

    # create figure and plot a KDE plot for each subset
    plt.figure(figsize=(8, 5))
    sns.kdeplot(case_atts["AGE_AT_INDEX"], fill=True, color="blue", label="Cases")
    sns.kdeplot(
        control_atts["AGE_AT_INDEX"], fill=True, color="orange", label="Controls"
    )
    plt.title(title)
    plt.legend()

    if save_path:
        plt.savefig(save_path)
