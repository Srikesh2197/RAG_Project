# Stage 5: example gold sentences for each failure type

Written by `python scripts/stage05_sweep.py --write`. Ranks are of the first chunk that overlaps the gold sentence, in `chunk-recursive-128` (dense), `retr-sparse` (BM25) and `retr-hybrid-rrf` (hybrid). Overlap is the share of the sentence's terms that the question also uses.

## Dense misses what BM25 finds

BM25 has the sentence in its top 10; dense does not have it in its top 50. 50 of 1079 gold sentences.

**1. q_0129, inference.** Dense not in 50, BM25 1, hybrid 13. Overlap 28%.

- Question: Who is the individual whose criminal trial, involving allegations of directing a $14 billion misuse of customer funds and self-enrichment through fraud, is reported by both TechCrunch and Cnbc | World Business News Leader, and is also accused of planning a board with experts for a company while excluding investors as directors?
- Gold sentence: During cross-examination, Huang said Paradigm pressed Bankman-Fried on the board issue and was told he didn't want investors as directors but he did plan on having a board with experts.

**2. q_0594, inference.** Dense not in 50, BM25 1, hybrid 20. Overlap 31%.

- Question: Which company, featured in multiple TechCrunch articles, is not only responsible for introducing GPT-4 Turbo and planning to make GPT-4 with vision available but is also predicted to dominate the AI tools platform market, overshadowing competitors like Hugging Face, and is known for creating the popular generative AI, ChatGPT?
- Gold sentence: The “app store for AI” will be pushed hard as the platform to get your AI toys and tools from, and never mind Hugging Face or any open source models.

**3. q_1010, temporal.** Dense not in 50, BM25 1, hybrid 9. Overlap 33%.

- Question: Did the reporting style on players achieving first downs in Sporting News articles change between the article featuring Anthony Hankerson on October 7, 2023, and the one highlighting A.J. Dillon on December 3, 2023?
- Gold sentence: 7:06 p.m. – Anthony Hankerson gets two first downs for the Buffs on a four-yard run and a 12-yard run.

**4. q_1308, temporal.** Dense not in 50, BM25 1, hybrid 24. Overlap 60%.

- Question: After TechCrunch reported on the shutdown of Pebble, the Twitter alternative previously known as T2, on October 24, 2023, did the role of social media as highlighted by FOX News - Health in the "Save Lucas" campaign by The Goeller family on December 9, 2023, remain consistent or inconsistent with the importance of social media's role as mentioned in the TechCrunch article?
- Gold sentence: The Goellers are now seeking support again, launching the "Save Lucas" campaign across social media.

**5. q_2237, inference.** Dense not in 50, BM25 1, hybrid 2. Overlap 23%.

- Question: Which team, known for being a favorite and having previously suffered defeats in Christchurch and Sydney, attempted a strategic play while having a numerical advantage on the field during a controversial and dramatic final, as discussed in articles from 'The Roar | Sports Writers Blog'?
- Gold sentence: It comes despite Argentina knocking over the All Blacks last year in Christchurch, as well as a maiden defeat in 2020 in Sydney.

## BM25 misses what dense finds

Dense has the sentence in its top 10; BM25 does not have it in its top 50. 31 of 1079 gold sentences.

**1. q_2147, inference.** Dense 1, BM25 not in 50, hybrid 16. Overlap 21%.

- Question: Who, after a period of injury affecting his participation in the MLS playoffs with Inter Miami, is also recognized for influencing the integration of emerging talents into Argentina's forward line as observed in a World Cup qualifying match covered by Sporting News?
- Gold sentence: He is helping to usher in a youth movement up front for Argentina, with Julian Alvarez and Alejandro Garnacho considered the future of the forward line.

**2. q_2237, inference.** Dense 1, BM25 not in 50, hybrid 12. Overlap 0%.

- Question: Which team, known for being a favorite and having previously suffered defeats in Christchurch and Sydney, attempted a strategic play while having a numerical advantage on the field during a controversial and dramatic final, as discussed in articles from 'The Roar | Sports Writers Blog'?
- Gold sentence: Sensing an opportunity to strike against 14 men, the All Blacks kicked for the corner on a couple occasions out wide.

**3. q_1214, comparison.** Dense 2, BM25 not in 50, hybrid 15. Overlap 0%.

- Question: Does the TechCrunch article suggest that social networks are controlled by large corporations in a similar way to how The Age article implies that DeepMind was a target for acquisition by major tech companies?
- Gold sentence: By the end of 2012, Google and Facebook were angling to acquire the London lab, according to three people familiar with the matter.

**4. q_1805, inference.** Dense 2, BM25 not in 50, hybrid 11. Overlap 19%.

- Question: Who is the player recognized by Sporting News as the top wide receiver for Week 14 and might face challenges in reaching 2,000-plus receiving yards in a single season due to the strong pass defenses of his team's remaining opponents?
- Gold sentence: The league's leading receiver, Tyreek Hill (vs. Titans in Week 14), stands as the unquestioned WR1 for Week 14 after torching the Commanders to the tune of five catches, 157 yards, and two TDs.

**5. q_2379, temporal.** Dense 2, BM25 not in 50, hybrid 15. Overlap 11%.

- Question: After the CBSSports.com report on Brock Purdy's performance published on October 4, 2023, and the subsequent CBSSports.com analysis of his play under pressure published on October 12, 2023, was there a change in the assessment of Brock Purdy's performance?
- Gold sentence: However, Purdy's been at his worst when pressured (like most quarterbacks), completing 50% of his throws for 6.7 yards per attempt with a gaudy 15.9% off-target rate.

## Both miss

Neither retriever has the sentence in its top 50. Lowest overlap first. 133 of 1079 gold sentences.

**1. q_0083, temporal.** Dense not in 50, BM25 not in 50, hybrid not in 50. Overlap 0%.

- Question: Has the portrayal of Sam Bankman-Fried's legal situation in TechCrunch articles changed between the report published on October 2, 2023, and the one published on October 7, 2023?
- Gold sentence: SBF, as he’s known, has pleaded not-guilty to some seven charges of fraud and conspiracy.

**2. q_0466, temporal.** Dense not in 50, BM25 not in 50, hybrid not in 50. Overlap 0%.

- Question: Between the Sporting News report on NBA betting sites and apps published on October 2, 2023, and the Sporting News report on point spread betting published on November 1, 2023, was the reporting on how sportsbooks adjust their betting lines consistent?
- Gold sentence: From there, you can claim your welcome bonus, which can come in many forms (be sure to read the requirements of any welcome bonus before using it).

**3. q_1445, inference.** Dense not in 50, BM25 not in 50, hybrid not in 50. Overlap 0%.

- Question: Which company, covered by both Engadget and Polygon, is set to release an upgraded version of its product with an improved screen, enhanced battery life, and several minor physical upgrades, with availability starting on November 16th?
- Gold sentence: Valve has announced a new Steam Deck and — double surprise — we’ve already reviewed it.

**4. q_2217, inference.** Dense not in 50, BM25 not in 50, hybrid not in 50. Overlap 0%.

- Question: What team was eliminated from European competitions after a loss at Old Trafford, as reported by both 'The Independent - Sports' and 'Sporting News'?
- Gold sentence: United were playing in quite a controlled way when they were going to eventually have to go for it, but Bayern were still getting through that with relative ease.

**5. q_2245, comparison.** Dense not in 50, BM25 not in 50, hybrid not in 50. Overlap 0%.

- Question: Do 'The Verge' and 'Engadget' articles both suggest that 'Consumers' can find guidance or deals on tech products, while the 'TechCrunch' article proposes a different interest of 'Consumers' in the realm of social networking?
- Gold sentence: And if you want to do even more research before making a buying decision, we’ve put together guides to the best wireless earbuds and best noise-canceling headphones, which can help you determine which pair is right for you.

## Fusion loses what one retriever had

One retriever has the sentence in its top 10 and the other not in its top 50; the fused list puts it below rank 20, outside a 2,000-token prompt. 32 of 1079 gold sentences.

**1. q_0961, temporal.** Dense not in 50, BM25 9, hybrid not in 50. Overlap 47%.

- Question: Has the focus of the European Commission's involvement reported by TechCrunch changed from addressing competition concerns in Amazon's iRobot purchase to facilitating dialogue and assessing issues with Meta's ad-free subscription service to probing Elon Musk's X over illegal content risks and moderation practices?
- Gold sentence: The process also loops in the European Commission to help facilitate dialogue, assess issues and bring pressure to bear on unfair practices.

**2. q_1331, inference.** Dense 4, BM25 not in 50, hybrid not in 50. Overlap 8%.

- Question: Which type of establishments, as reported by Sporting News, are known to modify betting odds to manage their financial risk, may return wagers in certain weather-related interruptions, profit from betting outcomes regardless of the event's result, and alter specific award-related betting lines based on new information?
- Gold sentence: If a significant amount of money is being placed on one team or participant, sportsbooks might adjust the odds to balance their liability.

**3. q_2397, comparison.** Dense 9, BM25 not in 50, hybrid not in 50. Overlap 8%.

- Question: Does the TechCrunch article suggest that Sam Bankman-Fried's motivation for alleged fraudulent activities was for personal gain, while the Fortune article focuses on the jury's role in determining the truthfulness of Sam Bankman-Fried's actions, without attributing a specific motive?
- Gold sentence: The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve great wealth, power and influence, while the defense countered that the FTX founder acted in good faith, never meant to commit fraud or steal and basically got in over his head.

**4. q_0278, inference.** Dense not in 50, BM25 10, hybrid 45. Overlap 44%.

- Question: Which NFL wide receiver, who is recognized as the top player at his position for Week 14 by Sporting News, also requires an average of almost 153 yards over his final three games according to CBSSports.com, but faces strong pass defenses in his remaining games, which might hinder his pursuit of 2,000 receiving yards for the season, despite having scored two touchdowns and accumulated 157 receiving yards in a recent victory over the Washington Commanders as reported by The Guardian?
- Gold sentence: Tyreek Hill had two touchdowns among his 157 receiving yards to help the Miami Dolphins rout the Washington Commanders (4-9).

**5. q_2045, inference.** Dense 9, BM25 not in 50, hybrid 44. Overlap 18%.

- Question: Which rugby team, featured in articles from 'The Independent - Sports' and 'The Roar | Sports Writers Blog', faced home defeats to Ireland, South Africa, and Argentina, aimed to utilize a numerical advantage by kicking for the corner, and has players striving to conclude their careers on a high note, while also having lost to Argentina both in Christchurch and previously in Sydney?
- Gold sentence: Sensing an opportunity to strike against 14 men, the All Blacks kicked for the corner on a couple occasions out wide.

