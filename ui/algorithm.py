# Aux_Cord Recommendation Algorithm
#
# Flynn Richardson

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import matplotlib.pyplot as plt
import os
import pydotplus

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


CLIENT_ID = '29a9fed336c0425aa09a96c840d88a3f'
CLIENT_SECRET = '7aae807829bd4b27aed0bc546b4200f2'
client_credentials_manager = SpotifyClientCredentials(CLIENT_ID,CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


attribute_order = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                   'liveness','mode', 'speechiness', 'valence']


def list_tracks(compilation):
    '''
    Receives a track compilation (album or playlist) and generates a list of its tracks' ids
    
    Inputs:
    compilation (dictionary): album or playlist object, contains tracks
    
    Outputs:
    tracklist (list): a list of track ids
    '''
    if compilation['type'] == 'album':
        track_block = sp.album_tracks(compilation['id'])
    elif compilation['type'] == 'playlist':
        track_block = sp.playlist(compilation['id'], fields = 'tracks')['tracks']

    tracks = track_block['items']

    while track_block['next']:
        track_block = sp.next(track_block)
        tracks += track_block['items']

    if compilation['type'] == 'album':
        tracklist = list(map(lambda t: t['id'], tracks))
    elif compilation['type'] == 'playlist':
        tracklist = list(map(lambda t: t['track']['id'], tracks))

    return tracklist


def stats(compilation, is_playlist=True):
    '''
    Generates 8-column pandas dataframe of audio attributes from each track object
    contained in the track compilation (tracklist or playlist).
    Dataframe is indexed by track id.

    Inputs:
    compilation (dictionary or list): A playlist dictionary or a list of track ids
    is_playlist (boolean): specify if compilation is a list or playlist

    Outputs:
    stat_array (pandas dataframe): n x 8 dataframe of the audio features
        acousticness, danceability, energy, instrumentalness, liveness, mode,
        speechiness, and valence. Indexed by track id
    '''
    stat_list = []

    if is_playlist:
        tracklist = list_tracks(compilation)
    else:
        tracklist = compilation
    
    batches = len(tracklist) // 50

    if len(tracklist) % 50 != 0:
        batches += 1
    
    for batch in range(batches):
        stat_list += sp.audio_features(tracklist[batch * 50 : batch * 50 + 50])
    
    stat_array = pd.DataFrame(stat_list)
    stat_array.set_index('id', inplace=True)
    stat_array.drop(columns = ['type', 'uri', 'track_href', 'analysis_url',
                               'tempo', 'time_signature', 'duration_ms',
                               'loudness', 'key'], inplace = True)
    stat_array = stat_array[attribute_order]

    return stat_array


def int_stats(stat_array):
    '''
    Sorts non-mode audio features into 0.1-width bins

    Inputs:
    stat_array (pandas dataframe): n x 8 dataframe indexed by track id.

    Outputs:
    None, operates on given dataframe
    '''
    for column_name in attribute_order:
        if column_name != 'mode':
            stat_array[column_name] = pd.cut(stat_array[column_name],
                      [-0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


def find_max(int_df):
    '''
    For given categorical dataframe, counts the proportion per bin for each audio 
    attribute and stores counts in a 1-column dataframe.
    Returns a list of dataframes of all bins with non-zero counts.

    Inputs:
    int_df (dataframe): categorical dataframe where non-mode audio attributes are 
    sorted in to bins

    Outputs:
    attributes (list): list of 1-column dataframes for nonzero audio attribute bin 
    counts
    '''
    attributes = []
    for column in int_df:
        int_col = pd.DataFrame(int_df[column].value_counts())
        attributes.append(int_col[int_col[column] != 0] / len(int_df))

    return attributes


def top_intervals(atr_col):
    '''
    Finds up to two highest-density intervals for each audio attribute, approximates the average
    value within each interval, and returns a list of interval - average value pairs. For the
    audio attribute mode, returns the highest proportion mode.

    Inputs:
    atr_col (pandas dataframe): audio attribute column of interval counts

    Outputs:
    list of interval, average value pairs
    '''
    top_ints = []
    cumulative_p  = 0

    for index, row in atr_col.iterrows():
        if atr_col.columns[0] == 'mode':
            return [[index, index]]

        cumulative_p += row[0]
        top_ints.append([index, row[0], (index.left + index.right) / 2 * row[0]])

        if cumulative_p >= 0.5:
            break    

    top_ints.sort()
    index = 0

    while index + 1 < len(top_ints):
        if top_ints[index][0].right == top_ints[index + 1][0].left:
            top_ints[index][0] = pd.Interval(left=top_ints[index][0].left,
                                     right=top_ints[index + 1][0].right,
                                     closed='right')
            top_ints[index][1] += top_ints[index + 1][1]
            top_ints[index][2] += top_ints[index + 1][2]
            del top_ints[1]
        else:
            index += 1

    for interval in top_ints:
        interval[2] /= interval[1]
        del interval[1]

    return sorted(top_ints, key = lambda int: int[1], reverse = True)[:2]


def interval_choices(playlist):
    '''
    Lists top_intervals choices for each audio attribute given a playlist

    Inputs:
    playlist (dictionary): a dictionary containing tracks

    Outputs:
    max_list (list): list of lists of interval, average value pairs for each audio attribute
    '''
    stats_df = stats(playlist)
    int_stats(stats_df)
    max_list = find_max(stats_df)

    for atr_index in range(len(max_list)):
        max_list[atr_index] = (max_list[atr_index].rename_axis('interval')
                                                  .groupby('interval').sum())
        max_list[atr_index].sort_values(ascending=False, inplace=True,
                                        by=attribute_order[atr_index])
        max_list[atr_index] = top_intervals(max_list[atr_index])

    return max_list


def seq_generator(max_list, index):
    '''
    Takes max_list and uses average values as target criteria for each audio attribute.
    Recursively generates different target criteria given a list of top_intervals choices.
    (for max 2 intervals per audio attribute except mode, generates a max of 2^7 search sequences)

    Inputs:
    max_list (list): list of lists of interval, average value pairs
    index (int): counter keeping track of recursion count

    Outputs:
    choices (list): list of lists of target criteria permutations
    '''
    choices = []

    if index == 7:
        for i in max_list[index]:
            choices.append([i[1]])

    else:
        for i in max_list[index]:
            for item in seq_generator(max_list, index + 1):
                choices.append([i[1]] + item)

    return choices


def find_recommendations(playlist, target_genre):
    '''
    Given a playlist, recommends songs from a user-inputted genre, new releases, and
    from a seed of 5 tracks in the playlist.

    Inputs:
    playlist (dictionary): dictionary containing tracks
    target_genre (string): a genre recognized by Spotify API

    Outputs:
    rec_list (list): a list of track ids representing recommended songs
    '''
    seqs = seq_generator(interval_choices(playlist), 0)
    perms = len(seqs)

    if perms == 1:
        cap = 100
    else:
        cap = 128 // perms

    rec_list = []
    rec_list += sp.recommendations(seed_genres=target_genre + ['new_release'],
                                   target_acousticness=seqs[0][0],
                                   target_danceability=seqs[0][1],
                                   target_energy=seqs[0][2],
                                   target_intrumentalness=seqs[0][3],
                                   target_liveness=seqs[0][4],
                                   target_mode=seqs[0][5],
                                   target_speechiness=seqs[0][6],
                                   target_valence=seqs[0][7], limit=100)['tracks']

    for seq in seqs:
        rec_list += sp.recommendations(seed_tracks=list_tracks(playlist)[:5],
                                       target_acousticness=seq[0],
                                       target_danceability=seq[1],
                                       target_energy=seq[2],
                                       target_intrumentalness=seq[3],
                                       target_liveness=seq[4],
                                       target_mode=seq[5],
                                       target_speechiness=seq[6],
                                       target_valence=seq[7], limit=cap)['tracks']
    
    rec_list = list(map(lambda t: t['id'], rec_list))
    
    return rec_list


def predict_like_songs(playlist1,playlist2,d_genres,l_genre):
    '''
    Given two playlist objects, user 1's dislike genre and user 2's prefered genre,
    apply the machine learning algorithm and return a list of recommended songs.

    Inputs:
        playlist1: user 1 playlist object
        playlist2: user 2 playlist object
        d_geres: (list) user 1's disliked genres
        l_genre: (list) user 2's preffered genre

    Output:
        A list of list which contains the information of each track
    '''
    disliked_song_ids = []

    for i in range(4):
        tracks = sp.recommendations(seed_genres = d_genres, limit = 100)["tracks"]

        for track in tracks:
            disliked_song_ids.append(track["id"])

    disliked_stats = stats(disliked_song_ids, is_playlist = False)
    disliked_stats["like"] = 0

    liked_stats = stats(playlist1)
    liked_stats["like"] = 1

    training_data = pd.concat([liked_stats, disliked_stats], axis = 0)

    x_train = training_data[attribute_order]
    y_train = training_data["like"]
    recommended_songs = find_recommendations(playlist2, l_genre)
    x_test = stats(recommended_songs, is_playlist = False)

    c = DecisionTreeClassifier(min_samples_split=100)
    tr = c.fit(x_train, y_train)
    
    dot_data = tree.export_graphviz(c, out_file=None,
                                    feature_names=attribute_order,
                                    impurity=False,
                                    rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png("./static/tree.png")
    
    predictions = c.predict(x_test)

    x_test["predictions"] = predictions
    predicted_songs = x_test[x_test["predictions"] == 1]
    predicted_ids = set(predicted_songs.index.tolist())
    ids = list(predicted_ids)

    predicted_tracks = []
    batches = len(ids) // 50

    if len(ids) % 50 != 0:
        batches += 1

    for batch in range(batches):
        predicted_tracks += sp.tracks(ids[batch * 50: batch * 50 + 50])['tracks'] 

    output_list = []

    if len(predicted_tracks) > 0:
        for track in predicted_tracks[:30]:
            song_list = (track['name'], track['artists'][0]['name'], 
                          track['album']['name'], track['preview_url'])
            output_list.append(song_list)

    return output_list

def plot_histogram(pl1,pl2,features):
    '''
    Given two playlist objects and a list of audio features, plot and save 
    a histogram for each feature.

    Inputs:
        pl1 and pl2: playlist objects
        features: (list) a list of features to plot

    No output
    '''
    stats1 = stats(pl1)
    stats2 = stats(pl2)

    for feature in features:
        fig = plt.figure()
        x = stats1[feature]
        y = stats2[feature]
        ax = fig.add_subplot(111)
        ax.hist([x,y], 
                bins = [0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], 
                alpha = 0.5, density = True, 
                label = ["user 1","user 2"], 
                color = ["c","m"])
        ax.legend(loc = "upper right")
        ax.set_xlim([0,1])
        ax.get_yaxis().set_visible(False)
        ax.set_title(feature, fontsize=20)
        strfile = "./static/"+feature+"_plot.png"

        if os.path.isfile(strfile):
             os.remove(strfile)

        plt.savefig(strfile)


def gen_playlist(args):
    '''
    Generate a new playlist based on the given arguments from users

    Input:
        args: (dictionary) input information from user

    Output:
        a list of songs, with each song being a list of information regaridng 
        that song
    '''
    pl1 = sp.playlist(args['pl1'])
    pl2 = sp.playlist(args['pl2'])

    d_genres = args['d_genre']
    l_genre_1 = args['l_genre_1']
    l_genre_2 = args['l_genre_2']
    features = args['features']

    playlist = set()
    retry = 0

    while len(playlist) <= 5:
        for song in predict_like_songs(pl1,pl2,d_genres,l_genre_2):
            playlist.add(song)

        for song in predict_like_songs(pl2,pl1,d_genres,l_genre_1):
            playlist.add(song)

        retry += 1

        if retry == 4:
            break

    playlist = list(playlist)

    if features != []:
        plot_histogram(pl1, pl2, features)

    return playlist