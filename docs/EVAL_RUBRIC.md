# Game review rubric

Prompt for one reviewer agent per game from `scripts/eval.sh`. It is written for `game.example.yml`
(Spanish, medium, unaware killer); adjust the first paragraph for other configs. Keeping it unchanged
between batches is what makes their scores comparable.

You review one generated murder mystery party game (Spanish, difficulty: medium, killer does not know
they are the killer). The game is played at home: each guest reads only their own character sheet, then
in Act 2 the host plays the detective, reveals clues one by one and guides interrogations, and finally
reads the solution.

Extract the text of every PDF in the game folder with `pdftotext -layout <file> -` (ignore images and
layout; only judge the text). Read the solution first to know the truth, then read everything else as
the people at the table would.

Judge only these dimensions. For each, give a score 1–5 and list every concrete problem with a short
quote of the conflicting text and the document it comes from.

1. **Contradictions**: two documents disagree on a fact players can compare (time, place, object,
   substance, who did what, relationship, gender), or a suspect's own sheet contradicts itself (e.g.
   they observe something from a place they had already left). Count them.
2. **Solvability (fair play)**: can a careful group deduce the killer from the clues plus the
   observations in the suspects' sheets? Does the solution rely on any fact players never had? Can every
   red herring be cleared with a clue or an observation?
3. **Difficulty**: medium means two or three clues plus observations from at least two players are
   needed. Can the killer be named from a single clue, a single observation, or two pieces without
   comparing testimonies? Do several suspects all report the same incriminating observation?
4. **Red herrings**: do innocent suspects look plausible (means or opportunity, not only motive)?
5. **Spoilers**: does anything players see before the reveal (clue titles, clue texts, guiding
   questions, Act 2 intro, the killer's own sheet) single out the killer?
6. **Killer's sheet**: could the killer's player tell from their own sheet's shape or content that they
   are the killer?

End your report with this summary block, exactly:

```
SUMMARY
contradictions_count: N
spoilers_count: N
scores: contradictions=X solvability=X difficulty=X red_herrings=X spoilers=X killer_sheet=X
overall: X/5
one_line: <one sentence verdict>
```

Do not suggest fixes to the generator code; only review the game.
