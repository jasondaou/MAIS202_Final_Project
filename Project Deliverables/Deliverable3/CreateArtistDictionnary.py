from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import pandas as pd

client_credentials_manager = SpotifyClientCredentials(client_id='e6c6ff809f24475cbcf4280718a78889', client_secret='238f8d6a6f7d4ea3a6b2d09cb1cd65a0')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


#Create a dictionnary of Artist name to popularity, add popularity as column to csv, then use popularity as feature
csv_input = pd.read_csv('SpotifyAudioFeaturesNov2018.csv', header=0)
artistPopularityDict = {}
artistNames = csv_input.iloc[1:-1, 0]
csv_input.insert(2, "Artist Popularity", 0)
index = 1;
for name in artistNames:
    try:
        if name not in artistPopularityDict:
            results = sp.search(q='artist:' + name, type='artist')
            id = results["artists"]["items"][0]["id"]
            artist = sp.artist(id)
            artistPopularityDict[name] = artist["popularity"]
            csv_input.at[index, "Artist Popularity"] = artist["popularity"]
        print(index, "\r")
    except:
        continue
    finally:
        index += 1
        if index % 500 == 0:
            popularities = pd.DataFrame.from_dict({"Artist" : list(artistPopularityDict.keys()), "Popularity" : list(artistPopularityDict.values())})
            popularities.to_csv('output1.csv', index=False)

popularities = pd.DataFrame(artistPopularityDict, columns=["Artist" , "Popularity"])
popularities.to_csv('output1.csv', index = False)
