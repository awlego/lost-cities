"""Challenger player for autoresearch-style iterative improvement.

This is the ONLY file the AI agent should modify. Started as a copy of
Committer.  The goal is to iteratively improve win rate vs Committer.
"""

from classes import *
from utils import *

class Challenger(Player):
    @classmethod
    def get_name(cls):
        return 'challenger'


    '''
    heurstics to add: 
    -[ ] open suits that you have length, weighted by the sum of the value of your cards.
    EV = multiplier * (20 - sum(value of cards + expected value of draws of that suit)) (probably need to make a function to estimate the expected value of draws of that suit)
    -[ ] number of cards in hand that you need/want to play so that you can correctly stall by drawing duds end game if your hand has lots of plays and you want to drag the game out
    -[ ] don't discard obvious plays for your opponents
    -[ ] create a heuristic for obvious discards in your hand that are safe discards (cards that are in a suit that an opponent has already committed a higher number to)
    -[ ] prefer to not open new suits if there is another good play
    -[ ]
    '''
    def play(self, r):
        me = r.whose_turn

        draw = 'deck'
        possible_draws = playable_draws(r.flags, me)
        cards = r.h[me].cards
        playable_cards = [c for c in cards
                             if is_playable(c, r.flags[c[0]].played[me])]
    

        if playable_cards:

            
            
        else:
            discard = // todo 
            draw = // todo
            return discard, True, draw
