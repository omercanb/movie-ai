A project that aims to train an LLM to speak like a movie character.\
There are further goals to make the LLM generate stage directions as well making the experience liek a movie.

It was need to create a new dataset with categorized movie dialogue/stage directions instead of the Cornell Movie-Dialogs Corpus for learning purposes.
The Internet Movie Script Database (IMSDb) was used to create the dataset. It has some scripts that are formatted such that the lines can be categorized based on whitespace.
For more data, the formatted scripts were used to train a CNN which was used to categogrize the unformatted scripts' lines. (A transformer model may also be used in the future)
