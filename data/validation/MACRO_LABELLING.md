# Labelling the economy-talk sample

Two files, both labelled by people only. No scorer output is shown, so you can't be nudged.

| File | Posts | Who |
|---|---|---|
| `macro_sample.csv` | 45 | Labeller C rows 1–15, Labeller A rows 16–30, Labeller B rows 31–45 |
| `macro_overlap.csv` | the same first 15 posts again | **Labeller A**, without opening rows 1–15 of `macro_sample.csv` or asking Labeller C |

The overlap measures whether two people give the same label to the same post. If they often
disagree, the labels can't be used to judge the scorer. Please work separately.

## The question for every post
**Is this post presenting the economy as getting worse, better, or neither?**

Write one word in `human_label`:

- **bad**: says things are getting worse. Examples: rising inflation, rate hikes hurting people, recession or crisis warnings, job losses, markets falling on bad news.
- **good**: says things are getting better. Examples: inflation easing, growth, strong jobs, rate cuts welcomed, markets rising on good news.
- **neutral**: no direction. Examples: questions, plain data with no spin, mixed or balanced pieces, or off-topic posts that only mention a keyword in passing.

Judge what the post itself says, not what you personally think about the economy or what it means for gold.
Put your name in `labelled_by`, and save as **CSV UTF-8**. Don't edit any other column.
