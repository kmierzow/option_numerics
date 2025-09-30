import math
from typing import List, Tuple, Dict

players_and_elos = {
    'A': 2000,
    'B': 1900,
    'C': 1800,
    'D': 1700,
    'E': 1600,
    'F': 1500,
    'G': 1400,
    'H': 1300
}

def A_beats_B(A: str, B: str) -> float:
    elo_a = players_and_elos[A]
    elo_b = players_and_elos[B]
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def win_probability(player: str, matchups: List[Tuple[str, str]], bracket: List[Tuple[str, str]]) -> float:
    if not matchups:  # Reached the end (winner)
        return 1.0
    
    # Find the player's current match
    for i, (p1, p2) in enumerate(matchups):
        if p1 == player or p2 == player:
            opponent = p2 if p1 == player else p1
            win_prob = A_beats_B(player, opponent)
            break
    else:
        return 0.0  # Player not in this round
    
    next_matchups = matchups[:i] + matchups[i + 1:]
    
    next_opponent_idx = i ^ 1  # Pair matches (0 with 1, 2 with 3, etc.)
    if next_opponent_idx < len(next_matchups):
        opp1, opp2 = next_matchups[next_opponent_idx]
        # Probability of facing each opponent
        opp1_prob = A_beats_B(opp1, opp2)
        opp2_prob = A_beats_B(opp2, opp1)
        # Create next round matchups (pair winners)
        next_round = []
        for j in range(0, len(next_matchups), 2):
            if j == next_opponent_idx:
                next_round.append((player, opp1))
            elif j + 1 < len(next_matchups):
                next_round.append((next_matchups[j][0], next_matchups[j + 1][0]))
        # Recurse for both possible opponents
        prob1 = win_probability(player, next_round, bracket) * win_prob * opp1_prob
        next_round[-1] = (player, opp2)
        prob2 = win_probability(player, next_round, bracket) * win_prob * opp2_prob
        return prob1 + prob2
    else:
        # Last match in round (e.g., semifinals to final)
        next_round = [(player, next_matchups[0][0])]
        return win_probability(player, next_round, bracket) * win_prob

# Example usage
matchups = [('A', 'B'), ('C', 'D'), ('E', 'F'), ('G', 'H')]
target_player = 'A'
probability = win_probability(target_player, matchups, matchups)
print(f"Probability of {target_player} winning: {probability:.4f}")
