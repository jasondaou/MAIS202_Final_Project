import pandas as pd

dataset = pd.read_csv('SpotifyAudioFeaturesNov2018.csv', header=0)
artistPopularitycsv = pd.read_csv('output1.csv', header=0)
dataset.insert(3, 'Artist Popularity', '0')
artistPopularityDict = {}
artistNames = dataset.iloc[1:-1, 0]
# dataset.insert(2, "Artist Popularity", 0)
for index, row in artistPopularitycsv.iterrows():
    artistPopularityDict[row["Artist"]] = row["Popularity"]

i = 1;
for index, row in dataset.iterrows():
    name = row["artist_name"]
    if name in artistPopularityDict:
        dataset.at[index, "Artist Popularity"] = artistPopularityDict[name]
    print(index, "\r")

dataset.to_csv('DatasetWithArtistPopularity.csv', index=False)
