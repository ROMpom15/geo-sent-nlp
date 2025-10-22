# si425project #
## Description: ##
    Our goal is to compare viewpoint differences/sentiment between American and Chinese news outlets. For instance, we will be comparing what kinds of opinions American and Chinese citizens have about the Taiwan Strait, South China Sea, the American and Chinese militaries, American and Chinese navies, shipbuilding in both countries, and economies in both countries. Our general approach to achieving this is to use a news source's data set card to access the contents of the news articles.

    We will then attempt to extract the sentences/paragraphs surrounding the terms/topics listed above and evaluate the perspectives of the respective nationalities. Our sources of data are CNN's DailyMail data set taken from HuggingFace (for English articles) and THUCNews taken from figshare (for Chinese articles).

    This is an interesting topic because it's pretty relevant to the US Navy and Marine Corps and the current geopolitical tensions between the US and China, so it would be cool to analyze what is actually being said on a lower level closer to the general public as opposed to what senior leaders throw at each other. 

    1. Sentiment analysis

        Compare sentiment distributions (means, variances) across countries and topics.

    2. Topic-specific linguistic associations

        A user inputs a “hot-button” topic (e.g., South China Sea).
        Using document embeddings and vector search, we extract top co-occurring words and multi-word expressions.
        We then cluster and compare common phrase associations to identify differing frames.

        Example:
        Chinese articles: defense, territorial waters, sovereignty
        American articles: freedom of navigation, military maneuvers, fire hoses

    3. Summaries

        Use summarization models to generate short (1–2 sentence) summaries per topic cluster.
        Compare resulting summaries for tone, emphasis, and framing differences.

        Example:
        U.S. summary: “China seeks to assert power in territorial waters through aggressive maneuvers.”
        Chinese summary: “China defends domestic fishing shoals against foreign incursions.”

        If we had time and availability, we’d like to find objective statements common to both language clusters and attempt to remove bias or subjective perspectives. 

Data:
    Our CNN DailyMail data will be taken from Hugging Face. There are over 300,000 new articles with associated "highlights" or summary bullet points. The highlights serve as target summaries of the full article. The average article length is relatively long (hundreds of tokens) and highlights are much shorter.
        https://huggingface.co/datasets/abisee/cnn_dailymail/tree/main/3.0.0
    Our THUCNews data will be taken from figshare. There are 840,000 Chinese news documents organized into 14 classes. It was built from historical RSS feeds of Sina News through filtering. Because the contents are Chinese text, it will be necessary to preprocess and segment words. This data is relatively "clean" in the sense the it has clear category labels, making it a good data set for baseline classification experiments in Chinese. 
        https://figshare.com/articles/dataset/THUCNews_Chinese_News_Text_Classification_Dataset/28279964?file=51924092
        https://dqwang122.github.io/projects/CNewSum/
        https://www.kaggle.com/datasets/ceshine/yet-another-chinese-news-dataset
    Chinese Sentiment Analysis
    https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment