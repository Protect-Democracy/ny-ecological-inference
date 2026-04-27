# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pyei==1.1.1",
#     "pymc==5.16.2",
# ]
# ///

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyei.r_by_c import RowByColumnEI


with Path.open("results.json", "r") as f:
    data = json.load(f)

for race in data["results"]["ballotItems"]:
    if "President" in race["name"]:
        for candidate in race["ballotOptions"]:
            if "harris" in candidate["name"].lower():
                for precinct in candidate["precinctResults"]:
                    vote_count = precinct["voteCount"]
                    if (
                        "ramapo" in precinct["name"].lower()
                        and "98" in precinct["name"].lower()
                    ):
                        print(
                            "Harris votes in ",
                            precinct["name"].split("(")[0].lower().strip(),
                            "",
                            vote_count,
                        )
    if "United States Senator" in race["name"]:
        for candidate in race["ballotOptions"]:
            if "gillibrand" in candidate["name"].lower():
                for precinct in candidate["precinctResults"]:
                    vote_count = precinct["voteCount"]
                    if (
                        "ramapo" in precinct["name"].lower()
                        and "98" in precinct["name"].lower()
                    ):
                        print(
                            "Senate D votes in ",
                            precinct["name"].split("(")[0].lower().strip(),
                            "",
                            vote_count,
                        )

candidate["name"]


def extract_df_from_results(data=data):
    # The keys will be precinct names
    # and the values will be dictionaries of vote counts
    precinct_data = {}

    # Extract the list of ballot items (races)
    ballot_items = data["results"]["ballotItems"]

    # Iterate over each race in the election
    for race in ballot_items:
        race_name = race["name"]
        # Iterate over each candidate in the race
        for candidate in race["ballotOptions"]:
            candidate_name_full = candidate["name"]
            party = candidate["politicalParty"]

            # Determine a clean column name based on the race and candidate
            if "President" in race_name:
                if "harris" in candidate_name_full.lower():
                    col_name = "Votes_Harris"
                elif "trump" in candidate_name_full.lower():
                    col_name = "Votes_Trump"
                else:
                    col_name = "Votes_President_Other"
            elif "United States Senator" in race_name:
                if party == "Democratic":
                    col_name = "Votes_Senate_D"
                elif party == "Republican":
                    col_name = "Votes_Senate_R"
                else:
                    col_name = "Votes_Senate_Other"
            else:
                # Fallback for other races
                col_name = f"Votes_{race['id']}_{candidate['id']}"

            # Iterate over each precinct's results for the current candidate
            for precinct in candidate["precinctResults"]:
                precinct_name = precinct["name"].split("(")[0].lower().strip()
                precinct_id = precinct["id"]
                vote_count = precinct["voteCount"]

                # If this is the first time we're seeing this precinct,
                # initialize its dictionary
                if precinct_name not in precinct_data:
                    precinct_data[precinct_name] = {}
                    precinct_data[precinct_name]["precinct_id"] = (
                        precinct_id  # Add metadata
                    )

                # Add the vote count for the current candidate to the precinct's data
                if col_name not in precinct_data[precinct_name].keys():
                    precinct_data[precinct_name][col_name] = vote_count
                else:
                    precinct_data[precinct_name][col_name] += vote_count

    # Convert the dictionary to a pandas DataFrame, with precinct names as the index
    results = pd.DataFrame.from_dict(precinct_data, orient="index")

    # Reorder columns for clarity
    desired_order = [
        "precinct_id",
        "Votes_Harris",
        "Votes_Trump",
        "Votes_President_Other",
        "Votes_Senate_D",
        "Votes_Senate_R",
        "Votes_Senate_Other",
    ]
    # Filter for columns that exist in the dataframe to avoid errors
    existing_columns = [col for col in desired_order if col in results.columns]
    results = results[existing_columns]

    return results


results = extract_df_from_results()

registered_voters = pd.read_csv("registered_voters.csv", index_col=0)
registered_voters["lastname"] = registered_voters["lastname"].astype(int)
r = results.merge(registered_voters, left_index=True, right_index=True, how="outer")

group_fractions = r[["Votes_Harris", "Votes_Trump", "Votes_President_Other"]].astype(
    int
).to_numpy() / np.repeat(r["lastname"].to_numpy().reshape(307, 1), axis=1, repeats=3)

group_fractions = group_fractions.T
group_fractions = np.concat(
    [group_fractions, 1.0 - np.sum(group_fractions, axis=0).reshape(1, -1)], axis=0
)

votes_fractions = r[["Votes_Senate_D", "Votes_Senate_R", "Votes_Senate_Other"]].astype(
    int
).to_numpy() / np.repeat(r["lastname"].to_numpy().reshape(307, 1), axis=1, repeats=3)
votes_fractions = votes_fractions.T
votes_fractions = np.concat(
    [votes_fractions, 1.0 - np.sum(votes_fractions, axis=0).reshape(1, -1)], axis=0
)

registered_voters

ei = RowByColumnEI(model_name="multinomial-dirichlet")
ei.fit(
    group_fractions,
    votes_fractions,
    registered_voters["lastname"].to_numpy(),
    ["Harris", "Trump", "POTUS Other", "No Vote"],
    ["Senate_D", "Senate_R", "Senate Other", "No Vote"],
    progressbar=True,
)


# Print results
means = pd.DataFrame(
    ei.posterior_mean_voting_prefs,
    columns=["Harris", "Trump", "POTUS Other", "No Vote"],
    index=["Senate_D", "Senate_R", "Senate Other", "No Vote"],
)
print(means)

precinct_level_samples = np.transpose(
    ei.sim_trace["posterior"]["b"].stack(all_draws=["chain", "draw"]).values,
    axes=(3, 0, 1, 2),
)

outliers = [
    "Ramapo 109",
    "Ramapo 111",
    "Ramapo 117",
    "Ramapo 118",
    "Ramapo 21",
    "Ramapo 35",
    "Ramapo 45",
    "Ramapo 55",
    "Ramapo 58",
    "Ramapo 95",
    "Ramapo 97",
    "Ramapo 98",
]

outliers = [
    "clarkstown 1",
    "clarkstown 10",
    "stony point 5",
    "stony point 6",
    "stony point 7",
]

bins = np.linspace(0, 1.0, 50)
for outlier in outliers:
    idx = np.where(r.index == outlier.lower())[0]
    print(idx)
    x = precinct_level_samples[:, idx, :, :].reshape(-1, 4, 4)
    fig = plt.figure(figsize=[6, 10])
    plt.subplot(311)
    plt.hist(x[:, 0, 0], bins=bins, density=True, label="Harris %")
    plt.hist(x[:, 0, 1], bins=bins, density=True, label="Trump %")
    plt.hist(x[:, 0, 2], bins=bins, density=True, label="Other %")
    plt.hist(x[:, 0, 3], bins=bins, density=True, label="No Vote %")
    plt.title("Senate D")
    plt.legend()
    plt.subplot(312)
    plt.hist(x[:, 1, 0], bins=bins, density=True, label="Harris %")
    plt.hist(x[:, 1, 1], bins=bins, density=True, label="Trump %")
    plt.hist(x[:, 1, 2], bins=bins, density=True, label="Other %")
    plt.hist(x[:, 1, 3], bins=bins, density=True, label="No Vote %")
    plt.title("Senate R")
    plt.legend()
    plt.subplot(313)
    plt.hist(x[:, 3, 0], bins=bins, density=True, label="Harris %")
    plt.hist(x[:, 3, 1], bins=bins, density=True, label="Trump %")
    plt.hist(x[:, 3, 2], bins=bins, density=True, label="Other %")
    plt.hist(x[:, 3, 3], bins=bins, density=True, label="No Vote %")
    plt.title("Senate No Vote")
    plt.legend()
    plt.suptitle(outlier)
    plt.tight_layout()
    plt.show()

r
r.iloc[290].T

r.iloc[idx].T
