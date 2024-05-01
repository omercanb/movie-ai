A project that aims to eventually enable you to speak to any movie character like you were a part of the movie.

The plan is to train the text generation model on dialogue and stage directions so it can format the conversation like a movie script.
Thus, it was need to create my own dataset with categorized movie dialogue/stage directions and cannot use the Cornell Movie-Dialogs Corpus.
The Internet Movie Script Database (IMSDb) was used to create the dataset. It has some scripts that are formatted such that the lines can be categorized based on whitespace.
For more data, the formatted scripts were used to train a CNN which was used to categogrize the unformatted scripts' lines. (A transformer model may also be used in the future)
Currently at this stage: Now a pretrained transformer will be fine tuned to generate dialogue as a starting point.
Another model will be fine tuned to generate stage directions and a router will choose which model to use. (PEFT methods will probabbly used instead of having two large transformer model)
Finally, based on what movie the user wants a final training will occur with that movie specifically. 
(Whether the user will be talking to a specific character and if so, how it will work is not decided yet.)
