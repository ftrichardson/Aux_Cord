import json
import traceback
import sys
import csv
import os

from functools import reduce
from operator import and_

from django.shortcuts import render
from django import forms

from algorithm import gen_playlist

NOPREF_STR = 'choose a genre'
RES_DIR = os.path.join(os.path.dirname(__file__),'res')

def _load_column(filename, col=0):
    '''Load single column from csv file.'''
    with open(filename) as f:
        col = list(zip(*csv.reader(f)))[0]
        return list(col)


def _load_res_column(filename, col=0):
    '''Load column from resource directory.'''
    return _load_column(os.path.join(RES_DIR, filename), col=col)


def _build_dropdown(options):
    '''Convert a list to (value, caption) tuples.'''
    return [(x, x) if x is not None else ('', NOPREF_STR) for x in options]


GENRE = _build_dropdown([None] + _load_res_column('genre.csv'))
FEATURES = _build_dropdown(_load_res_column('feature.csv'))


class SearchForm(forms.Form):
    pl1 = forms.CharField(label='User 1 Spotify Playlist URI',
                          help_text='e.g. spotify:playlist:37i9dQZF1DWWMOmoXKqHTD',
                          required=False)

    l_genre_1 = forms.ChoiceField(label='User 1 Preferred Genre',
                                  choices=GENRE,
                                  required=False)

    d_genre_1 = forms.ChoiceField(label='User 1 Disliked Genre 1',
                                  choices=GENRE,
                                  required=False)

    d_genre_2 = forms.ChoiceField(label='User 1 Disliked Genre 2 (Optional)',
                                  choices=GENRE,
                                  required=False)


    pl2 = forms.CharField(label='User 2 Spotify Playlist URI',
                          help_text='e.g. spotify:playlist:37i9dQZF1DWZeKCadgRdKQ',
                          required=False)

    l_genre_2 = forms.ChoiceField(label='User 2 Preferred Genre',
                                  choices=GENRE,
                                  required=False)

    d_genre_3 = forms.ChoiceField(label='User 2 Disliked Genre 1',
                                  choices=GENRE,
                                  required=False)

    d_genre_4 = forms.ChoiceField(label='User 2 Disliked Genre 2 (Optional)',
                                  choices=GENRE,
                                  required=False)
    
    features = forms.MultipleChoiceField(label='Visualize Audio Features',
                                    choices=FEATURES,
                                    widget=forms.CheckboxSelectMultiple,
                                    required=False)

    show_tree = forms.BooleanField(label='Visualize Decision Tree',
                                   required=False)
    

def home(request):
    context = {}
    res = None
    if request.method == 'GET':
        # create a form instance and populate it with data from the request:
        form = SearchForm(request.GET)
        # check whether it's valid:
        if form.is_valid():
            # Convert form data to an args dictionary for find_courses
            args = {}
            args['pl1'] = form.cleaned_data['pl1']
            args['pl2'] = form.cleaned_data['pl2']
            args['d_genre'] = []
            for i in range(4):
                g = form.cleaned_data['d_genre_'+str(i+1)]
                if g != '':
                    args['d_genre'].append(g)
            args['l_genre_1'] = [form.cleaned_data['l_genre_1']]
            args['l_genre_2'] = [form.cleaned_data['l_genre_2']]
            args['features'] = form.cleaned_data['features']
            args['show_tree'] = form.cleaned_data['show_tree']
            complete = True
            for key, value in args.items():
                if value == '' or value == []:
                    if key != 'features':
                        complete = False
            if complete == True:
                res = gen_playlist(args)

    else:
        form = SearchForm()
    if res == None:
        context['result'] = None
        context['features'] = None
        context['show_tree'] = False
    else:
        context['result'] = res
        context['columns'] = ['Track', 'Artist','Album','30s Preview URL']
        context['features'] = True
        for feature in args['features']:
            context[feature] = True
        context['show_tree'] = args['show_tree']

    context['form'] = form
    return render(request, 'index.html', context)


