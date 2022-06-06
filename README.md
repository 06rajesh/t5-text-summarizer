# t5-text-summarizer

implementation of t5 huggingface model for summarizing texts using pytorch and pytorch lighting.
<br/>`ipynb` file. Ready to run on google colab.

T5 huggingface: https://huggingface.co/docs/transformers/model_doc/t5 <br/>
Kaggle News Summary Dataset: https://www.kaggle.com/datasets/sunnysai12345/news-summary

### Example:
####Text:
```
The Hungarian Grand Prix could yet stand as one of the defining moments of the 2017 Formula One season.
Under the unyielding sun of a Budapest summer the two title protagonists, who have been equally unforgiving on 
track in a battle that has swayed gloriously between them, finally showed their hands as to how that fight would 
proceed in the second half of the season.Sebastian Vettel won what was hardly a classic race but Lewis Hamilton, 
in standing by his team-mate, admitted he might have dropped points that could ultimately cost him the world 
championship.Vettel won at the Hungaroring with an impressive drive from pole position. That he did so while 
nursing a damaged steering wheel made it all the more of an achievement but also put the Scuderia on the spot.
His team-mate Kimi Raikkonen was quicker on track but they chose not to order Vettel to allow him to pass. 
That the German is their only title contender could not have been clearer.In their wake Hamilton was on a charge 
and in turn faster than his team?mate Valtteri Bottas. With passing at the Hungaroring all but impossible this 
year, he asked Mercedes to tell Bottas to let him through to chase down the leading Ferraris, agreeing that he 
would give the place back should he fail to overtake them. Bottas let him through and, unable to pass Raikkonen 
and seven seconds up the road from his team-mate, Hamilton duly gave back the place on the final lap. Raikkonen 
held second, with Bottas in third and Hamilton fourth.The British driver now trails Vettel by 14 points in the 
world championship and ceding the place cost him three. With the title fight likely to go to the wire it might 
yet prove to have been the sportsmanlike gesture that could deny him a fourth drivers? title this year.?I want 
to win the championship the right way,? he said. ?I don?t know whether that will come back to bite me on the 
backside or not but I said at the beginning of the year, I want to win it the right way. I do think today was 
the right way to do things.?Ferrari have a long history of imposing team orders and designating a No1 driver 
but the Mercedes executive director, Toto Wolff, agreed with Hamilton and drew a stark comparison with how his 
team and the Scuderia went racing.?We have seen the backlash from decisions that were ruthless and cold blooded 
and the effect that it had on the brand,? he said. ?You could say: ?Screw it, it still won them the championship, 
who cares? They are in the history books.? But I do not think that is the right spin and if the purpose of us 
being here is to do the right things and win in the right way ? sometimes doing it the right way and standing 
by your values is tough and it was today, believe me.?Giving the place back was no easy feat. Bottas was only 
just in front of the Red Bull of Max Verstappen and there was a real danger the manoeuvre could have cost 
Hamilton two places. The team had debated whether it should be done but were clear Hamilton was not instructed 
to do so and that he was merely sticking to his word.?It was very sportsmanlike behaviour,? said Wolff, who 
also acknowledged that it could cost Hamilton the title. ?These values won us six championships and it will 
make us win more championships in years to come. It cost us three points and could cost us the championship 
and we are perfectly conscious of that. But in terms of how the drivers and this team operates, we stick to 
what we say and if the consequences are as much as losing the championship, then we will take it.?There was 
little doubt that Vettel deserved the 25 points, both he and Raikkonen having a pace advantage in the early 
stage over Mercedes. But he revealed he was wrestling with his damaged car from the off. ?I felt there was 
something not right on the grid and for the formation lap the steering wheel was not straight,? he said.?Then 
it got worse and towards the end of the stint it was more difficult. We spoke on the radio about it and I was 
told to avoid the kerbs but on this track you use them on every corner, so you lose speed. It was good that it 
is tricky to overtake here. It felt like a very long race.?Ferrari have not won the constructors? championship 
since 2008 and have not taken the drivers? title since Raikkonen claimed it in 2007. Their step forward to go 
head-to head with Mercedes this season gives them a real opportunity to end the drought.Raikkonen might have 
been passed by Hamilton, which would have cost them constructor points, but they opted to gamble on the British 
driver being unable to do so. It paid off and they can consider their strategy a success but it indicated 
that they were willing to risk the team championship in favour of backing Vettel for the title.Maurizio 
Arrivabene, Ferrari?s team principal, praised Raikkonen for being a ?true team player? and Sergio Marchionne, 
the Ferrari chief executive, admitted the result had come as a relief. ?It was a much?needed win,? he said. 
?The great thing is we earned it, this was a tough, tough race and we almost lost it and we got it back.
?Vettel holds the lead going into the summer break and he has earned it but Hamilton can also rest easy 
knowing that he held an honourable line in Hungary. ?The team were in a difficult position but I think 
today really shows, hopefully, that I am a man of my word,? he said.

```

####Output
```
The Hungarian Grand Prix could yet stand as one of the defining moments of the 2017 Formula One season. 
Lewis Hamilton won what was hardly a classic race but said he might have dropped points that could 
ultimately cost him the world championship. Ferrari's Kimi Raikkonen, who is their only title contender, 
had passed at the Hungaroring all but impossible this year.
```