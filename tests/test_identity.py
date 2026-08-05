from footy.stages.identity import IdentityResolver


def test_majority_vote_needs_minimum_readings():
    assert IdentityResolver.majority_vote([7, 7], min_votes=5) is None


def test_majority_vote_picks_dominant_number():
    assert IdentityResolver.majority_vote([7, 7, 7, 1, 7, 9], min_votes=5) == 7


def test_majority_vote_abstains_when_split():
    assert IdentityResolver.majority_vote([7, 1, 9, 3, 4, 5], min_votes=5) is None
