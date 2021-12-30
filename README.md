# Hate Speech Classification using NLP

- **Defining the problem** - With such a massive growth in the number of users on social media platforms, the communication outreach has revolutionised extremely. While most of the time these platforms tend to provide a positive or neutral atmosphere across the online community, there has also been an increase of hateful activities that exploit such infrastructure. For instance, on Twitter, hateful tweets commonly have abusive speech targeting individuals or particular groups. Identifying such negative content is
crucial for understanding public sentiment of a crowd towards another crowd.

- **Why does it matter** - As a result of this quick spread of hate speech content on social media platforms, a person or a group gets disparaged on the basis of their race, gender, sexual orientation, nationality, religion, or other characteristics.

- **Intention/Idea** - <!---TODO: Update the content here)-->
- **How does it help/ why does it matter** - From the practical usage point of view, this task would help many big organizations in reducing their hateful content. For example, social media companies need to flag up hateful content for moderation, while law enforcement needs to identify hateful messages and their nature as forensic evidence.

## Dataset Selection
For this project, I am using the [T-Davidson](https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data) dataset. The properties of this dataset are as follows:

- The data are stored as a CSV. Each data file contains 5 columns:

    - count = number of CrowdFlower users who coded each tweet (min is 3, sometimes more users coded a tweet when judgments were determined to be unreliable by CF).

    - hate_speech = number of CF users who judged the tweet to be hate speech.

    - offensive_language = number of CF users who judged the tweet to be offensive.

    - neither = number of CF users who judged the tweet to be neither offensive nor non-offensive.

    - class = class label for majority of CF users. 0 - hate speech 1 - offensive language 2 - neither
