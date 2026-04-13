from __future__ import annotations


EMOTION_LEXICON = [
    "happy", "joyful", "cheerful", "content", "satisfied", "delighted", "elated", "blissful", "playful", "amused",
    "grateful", "hopeful", "optimistic", "inspired", "motivated", "energized", "excited", "enthusiastic", "curious", "interested",
    "relieved", "calm", "peaceful", "serene", "relaxed", "safe", "secure", "confident", "proud", "accomplished",
    "loving", "affectionate", "caring", "compassionate", "empathetic", "warm", "tender", "fond", "trusting", "admiring",
    "respectful", "awed", "reverent", "nostalgic", "reflective", "fulfilled", "fulfilled", "brooding", "pensive", "thoughtful",
    "vulnerable", "open", "gentle", "patient", "forgiving", "accepting", "docile", "obedient", "humble", "bashful",
    "shy", "timid", "nervous", "anxious", "worried", "uneasy", "tense", "afraid", "fearful", "terrified",
    "panicked", "alarmed", "startled", "surprised", "shocked", "confused", "uncertain", "doubtful", "skeptical", "suspicious",
    "guarded", "defensive", "withdrawn", "lonely", "isolated", "sad", "sorrowful", "grieving", "melancholic", "gloomy",
    "downcast", "disappointed", "discouraged", "defeated", "hopeless", "despairing", "desperate", "helpless", "trapped", "ashamed",
    "embarrassed", "guilty", "remorseful", "regretful", "self-conscious", "inadequate", "inferior", "resentful", "envious", "jealous",
    "bitter", "frustrated", "irritated", "annoyed", "angry", "furious", "hostile", "spiteful", "indignant", "outraged",
    "disgusted", "repulsed", "contemptuous", "judgmental", "harsh", "cold", "detached", "apathetic", "numb", "tired",
    "drained", "exhausted", "fatigued", "overwhelmed", "burdened", "pressured", "stressed", "urgent", "restless", "fidgety",
    "impatient", "stubborn", "determined", "resolute", "defiant", "rebellious", "aggressive", "protective", "concerned", "sympathetic",
    "moved", "touched", "earnest", "sincere", "dutiful", "responsible", "professional", "measured", "composed", "balanced",
    "centered", "grounded", "brokenhearted", "heartsick", "yearning", "longing", "wistful", "devastated", "miserable", "agitated",
    "apprehensive", "cautious", "hesitant", "resigned", "subdued", "listless", "exuberant", "radiant", "gleeful", "ecstatic",
]


def get_emotion_list(count: int) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in EMOTION_LEXICON:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    if count > len(deduped):
        raise ValueError(f"Requested {count} emotions but only {len(deduped)} are available")
    return deduped[:count]
