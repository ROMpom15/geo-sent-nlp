# si425project #
## Description: ##
    Our goal is to compare viewpoint differences/sentiment between American and Chinese news outlets. For instance, we compare American and Chinese opinions about the Taiwan Strait, South China Sea, the American and Chinese militaries, American and Chinese navies, shipbuilding in both countries, and economies in both countries. Our general approach to achieving this is to use a news source's data set card to access the contents of the news articles.

    We  then attempt to extract the sentences/paragraphs surrounding the terms/topics listed above and evaluate the perspectives of the respective nationalities. Our sources of data are CNN's DailyMail data set taken from HuggingFace (for English articles) and THUCNews taken from figshare (for Chinese articles).

    1. topic analysis src/topic_ana

        # James

    2. Topic-specific linguistic summaries

        # Elijah

        A user inputs a “hot-button” topic (e.g., South China Sea).
        Using document embeddings and vector search, we extract top co-occurring words and multi-word expressions.
        We then cluster and compare common phrase associations to identify differing frames.

        Example:
        Chinese articles: defense, territorial waters, sovereignty
        American articles: freedom of navigation, military maneuvers, fire hoses
         Use summarization models to generate short (1–2 sentence) summaries per topic cluster.
        Compare resulting summaries for tone, emphasis, and framing differences.

        Example:
        U.S. summary: “China seeks to assert power in territorial waters through aggressive maneuvers.”
        Chinese summary: “China defends domestic fishing shoals against foreign incursions.”

        If we had time and availability, we’d like to find objective statements common to both language clusters and attempt to remove bias or subjective perspectives. 

    3. Translation Analysis src/trans/ana
        Compare the difference in translated Chinese news articles using cosine similarity.
        saves avg and count of articles compared to geo-sent-nlp/reports/scores.txt
       
       To run: python3 geo-sent-nlp/src/trans_ana/trans_analysis.py

Data:
    English
    Our CNN DailyMail data will be taken from Hugging Face. There are over 300,000 new articles with associated "highlights" or summary bullet points. The highlights serve as target summaries of the full article. The average article length is relatively long (hundreds of tokens) and highlights are much shorter.
        https://huggingface.co/datasets/abisee/cnn_dailymail/tree/main/3.0.0

    Chinese - CNewSum
    The second dataset is called CNewSum, which contains Chinese news summaries and improves on older datasets like LCSTS and THUCNews. It was from major Chinese online news portals such as Sina, Sohu, and NetEase. It was also filtered and cleaned according to human written summary test sets. This data was stored in jsonl format, having over 304,000 articles.
        https://dqwang122.github.io/projects/CNewSum/

    # Chinese Sentiment Analysis
    # https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment

Models:
    Translation:
    ALMA 13-b
    https://huggingface.co/haoranxu/ALMA-13B
    Implemented in utils/batch_trans.py

    Word-embeddings:
    jina
    https://huggingface.co/jinaai/jina-embeddings-v2-base-zh
    Implemented in /src/trans_ana/trans_analysis.py



