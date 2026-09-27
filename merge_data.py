"""Match IPL player names and create the manually approved merged dataset."""

from pathlib import Path

import pandas as pd
from thefuzz import process, fuzz


# Build paths relative to this script so it works from any current directory.
BASE_DIR = Path(__file__).resolve().parent
STATS_FILE = BASE_DIR / "data" / "ipl_stats.csv"
AUCTION_FILE = BASE_DIR / "data" / "ipl_auction.csv"
REVIEW_FILE = BASE_DIR / "data" / "matches_to_review.csv"
CONFIDENT_FILE = BASE_DIR / "data" / "matches_confident.csv"


print("Step 1: Loading the IPL statistics and auction CSV files...")
stats_df = pd.read_csv(STATS_FILE)
auction_df = pd.read_csv(AUCTION_FILE)
print(f"Loaded ipl_stats.csv with shape: {stats_df.shape}")
print(f"Loaded ipl_auction.csv with shape: {auction_df.shape}")

required_stats_columns = {"player"}
required_auction_columns = {"Player"}
missing_stats_columns = required_stats_columns - set(stats_df.columns)
missing_auction_columns = required_auction_columns - set(auction_df.columns)
if missing_stats_columns or missing_auction_columns:
	raise ValueError(
		"Missing required columns. "
		f"Stats: {sorted(missing_stats_columns)}; "
		f"Auction: {sorted(missing_auction_columns)}"
	)

print("\nStep 2: Getting unique player names from the auction file...")
auction_player_names = (
	auction_df["Player"]
	.dropna()
	.astype(str)
	.str.strip()
	.loc[lambda names: names != ""]
	.drop_duplicates()
	.tolist()
)
stats_player_names = (
	stats_df["player"]
	.dropna()
	.astype(str)
	.str.strip()
	.loc[lambda names: names != ""]
	.drop_duplicates()
	.tolist()
)
print(f"Found {len(auction_player_names)} unique auction player names.")
print(f"Found {len(stats_player_names)} unique stats player names to search.")

print("\nStep 3: Matching each auction player to the best stats name...")
match_rows = []
for auction_player_name in auction_player_names:
	best_match, match_score = process.extractOne(
		auction_player_name,
		stats_player_names,
		scorer=fuzz.token_sort_ratio,
	)
	match_rows.append(
		{
			"auction_player_name": auction_player_name,
			"matched_stats_player_name": best_match,
			"match_score": match_score,
		}
	)

match_results = pd.DataFrame(
	match_rows,
	columns=[
		"auction_player_name",
		"matched_stats_player_name",
		"match_score",
	],
)
print(f"Completed fuzzy matching for {len(match_results)} players.")

print("\nStep 4: Splitting matches by score for manual review...")
confident_matches = match_results[match_results["match_score"] >= 90].copy()
review_needed = match_results[
	match_results["match_score"].between(60, 89)
].copy()
no_match = match_results[match_results["match_score"] < 60].copy()

print(f"Confident matches (score >= 90): {len(confident_matches)}")
print(f"Matches needing review (score 60-89): {len(review_needed)}")
print(f"No match above score 60: {len(no_match)}")
print("Scores below 90 are not automatically accepted.")

print("\nStep 5: Saving the two review files...")
if REVIEW_FILE.exists() and "is_correct" in pd.read_csv(REVIEW_FILE, nrows=0).columns:
	print(
		f"Preserving the manually reviewed file at {REVIEW_FILE}; "
		"its is_correct values will be used below."
	)
else:
	review_needed.to_csv(REVIEW_FILE, index=False)
	print(f"Saved matches needing review to: {REVIEW_FILE}")
confident_matches.to_csv(CONFIDENT_FILE, index=False)
print(f"Saved confident matches to: {CONFIDENT_FILE}")

print("\nStep 6: Loading trusted and manually approved matches...")
confident_matches = pd.read_csv(CONFIDENT_FILE)
reviewed_matches = pd.read_csv(REVIEW_FILE)
if "is_correct" not in reviewed_matches.columns:
	raise ValueError(
		f"The review file must contain an 'is_correct' column: {REVIEW_FILE}"
	)

approved_review_matches = reviewed_matches[
	reviewed_matches["is_correct"] == "Yes"
]
final_matches = pd.concat(
		[
			confident_matches,
			approved_review_matches,
		],
		ignore_index=True,
	)[["auction_player_name", "matched_stats_player_name"]]
print(f"Trusted matches loaded: {len(confident_matches)}")
print(f"Manually approved matches loaded: {len(approved_review_matches)}")
print(f"Total matches used for the final merge: {len(final_matches)}")

print("\nStep 7: Merging auction rows with the approved name matches...")
auction_with_matches = auction_df.merge(
	final_matches,
	left_on="Player",
	right_on="auction_player_name",
	how="inner",
)
merged_df = auction_with_matches.merge(
	stats_df,
	left_on="matched_stats_player_name",
	right_on="player",
	how="inner",
)

original_auction_rows = len(auction_df)
retained_auction_rows = len(merged_df)
retained_percentage = (
	(retained_auction_rows / original_auction_rows) * 100
	if original_auction_rows
	else 0
)
data_loss_percentage = 100 - retained_percentage
print(f"Final merged DataFrame shape: {merged_df.shape}")
print(
	 f"Auction rows retained: {retained_auction_rows} of "
	 f"{original_auction_rows} ({retained_percentage:.2f}%)"
)
print(f"Auction row data loss: {data_loss_percentage:.2f}%")

print("\nStep 8: Validating the final merged matches...")
# Keep match scores in this temporary view so the saved dataset keeps its
# existing columns while suspicious rows can still be reported with scores.
match_details = pd.concat(
	[
		confident_matches,
		approved_review_matches,
	],
	ignore_index=True,
)[
	["auction_player_name", "matched_stats_player_name", "match_score"]
]
validation_df = merged_df.merge(
	match_details,
	left_on=["Player", "matched_stats_player_name"],
	right_on=["auction_player_name", "matched_stats_player_name"],
	how="left",
)
validation_df["auction_initial"] = (
	validation_df["Player"].astype(str).str.strip().str.split().str[0].str[:1].str.upper()
)
validation_df["stats_initial"] = (
	validation_df["matched_stats_player_name"]
	.astype(str)
	.str.strip()
	.str.split()
	.str[0]
	.str[:1]
	.str.upper()
)
suspicious_mismatches = validation_df[
	validation_df["auction_initial"] != validation_df["stats_initial"]
]
print(f"Suspicious initial mismatches found: {len(suspicious_mismatches)}")
if suspicious_mismatches.empty:
	print("No suspicious initial mismatches found.")
else:
	print("SUSPICIOUS_MISMATCH rows (manual review required):")
	print(
		suspicious_mismatches[
			["Player", "matched_stats_player_name", "match_score"]
		].to_string(index=False)
	)

players_per_stats_name = validation_df.groupby(
	"matched_stats_player_name"
)["Player"].nunique()
shared_stats_names = players_per_stats_name[players_per_stats_name >= 2]
print(
	"\nStats names matched to two or more different auction players: "
	f"{len(shared_stats_names)}"
)
if shared_stats_names.empty:
	print("No shared stats-name collisions found.")
else:
	print("Potential duplicate-player collisions (manual review required):")
	for stats_name in shared_stats_names.index:
		auction_players = (
			validation_df.loc[
				validation_df["matched_stats_player_name"] == stats_name,
				"Player",
			]
			.drop_duplicates()
			.tolist()
		)
		print(f"{stats_name}: {auction_players}")

print("\nStep 9: Saving the final merged dataset...")
merged_df.to_csv(BASE_DIR / "data" / "ipl_merged.csv", index=False)
print("Saved final merged data to: data/ipl_merged.csv")
print("Final merged columns:")
print(list(merged_df.columns))
